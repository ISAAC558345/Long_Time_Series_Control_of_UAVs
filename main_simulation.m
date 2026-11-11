%% main_simulation.m
% 海面浮标运动预测与无人机 ISAC 感知功率、水平位置联合优化
% MATLAB R2022b 单脚本仿真。不定义函数，不依赖工具箱。
%
% 模型要点（与论文一致，未额外引入三维水动力、海杂波或多普勒）：
%   深水 Airy 线性波；浮标仅在 x-z 平面运动，y 固定；
%   统一等效 Morison 方程；短时窗口内浸没长度恒定；
%   距离-方向余弦观测 + Airy-Morison-EKF；
%   4x4 UPA 归一化阵列响应；优化阶段 |omega|^2 = 1；
%   通信评估使用真实位置与预测位置的实际波束匹配；
%   上行信道只含一次 1/d 自由空间衰减；
%   固定位置下的感知功率分配由原 SDP 的 Schur 补等价问题经两层二分精确求解，
%   再与无人机候选网格搜索组成联合优化。不是启发式替代算法。
%
% 公式核对说明（不隐藏）：
%   观测函数为
%       h = [-kappa_a*dx/d; -kappa_a*dy/d; 2*d/c0]
%   对 z 求导得到
%       dh2/dz = -kappa_a*dy*dz/d^3
%   文中印刷的 H(2,2) = -kappa_a*(dx^2+dy^2)/d^3 与该观测函数矛盾。
%   当浮标与无人机 y 坐标相同时，h2 恒为 0，真实偏导为 0，印刷式却不为 0。
%   EKF 与信息矩阵必须使用 dh/ds，因此程序采用解析导数，并在启动时
%   用中心差分同时打印解析值、差分值和印刷值。观测函数本身不改。
%
% 第二轮修正（不改观测函数、不改信道、不改 H(2,2)）：
%   第 1 帧停在 q_U0，只优化功率；n>=2 才做粗/细网格；
%   0.1% 内的近似最优解优先选更靠近上一位置的点；
%   位置约束使用 d_s-0.30 m，评估仍用真实波束和一次路径损耗；
%   Monte Carlo 按 M0 抽初始误差，真实轨迹为 Morison 加与 Qs 一致的过程噪声。

clear; clc; close all;
format compact;

if exist('OCTAVE_VERSION', 'builtin')
    more off;
    graphics_toolkit('qt');
    set(0, 'DefaultFigureVisible', 'off');
    fig_font = 'WenQuanYi Micro Hei';
else
    fig_font = 'Times New Roman';
    font_list = {};
    try
        font_list = listfonts;
    catch
        font_list = {};
    end
    preferred_fonts = {'Microsoft YaHei', 'SimHei', 'SimSun', ...
        'Noto Sans CJK SC', 'Source Han Sans SC', 'Times New Roman'};
    for fi = 1:numel(preferred_fonts)
        if ~isempty(font_list) && any(strcmpi(font_list, preferred_fonts{fi}))
            fig_font = preferred_fonts{fi};
            break;
        end
    end
end
set(0, 'DefaultAxesFontName', fig_font);
set(0, 'DefaultTextFontName', fig_font);
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultTextFontSize', 11);
set(0, 'DefaultFigureColor', 'w');
set(0, 'DefaultAxesLineWidth', 0.8);
set(0, 'DefaultLineLineWidth', 1.3);

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
out_dir = script_dir;

rng(2026);

%% 1. 参数初始化
a = 0.35;
T_wave = 5;
g = 9.81;
water_depth = 50;
rho_water = 1025;

K = 3;
D_e = 0.8;
L_w = 0.8;
m_b = 200;
C_M = 2.0;
C_D = 1.2;
U_c = 0.03;
z0 = -1.0;
zeta = 0.7;

x_buoy0 = [0, 15, 30];
y_buoy = [-10, 0, 10];
z_buoy0 = -1.0;

c0 = 3e8;
fc = 5.8e9;
NantSide = 4;
sigma_r2 = 1e-8;
chi = 0.01;

P_b = 0.1;
sigma_c2 = 1e-10;
R_s = 4.0;
P_U = 6.0;

Ts = 0.25;
N = 60;
Nmc = 100;
sigma_ax = 0.03;
sigma_az = 0.04;

z_U = 20;
q_U0 = [10; -2];
commMargin = 0.30;
eps_t = 1e-3;
Vmax = 8;
x_min = -5;
x_max = 35;
y_min = -15;
y_max = 15;
gridStepCoarse = 1.0;
gridStepFine = 0.5;
outerIter = 22;
innerIter = 18;

eps_v = 1e-6;
epsFD = [1e-4, 1e-4, 1e-5, 1e-5];
nSub = 5;
k_rep = 2;
n_demo = 30;

% 初始估计相对真实状态的固定误差，三个浮标故意不同。
s_err0 = [ 0.45, -0.70,  1.10; ...
           0.35,  0.60, -0.90; ...
           0.05, -0.07,  0.09; ...
          -0.04,  0.06, -0.08];
sig_x0 = [0.70, 1.10, 1.60];
sig_z0 = [0.70, 1.10, 1.50];
sig_vx0 = [0.08, 0.12, 0.16];
sig_vz0 = [0.08, 0.12, 0.16];

I4 = eye(4);
I3 = eye(3);

%% 2. 海浪、浮标、阵列与通信参数
ww = 2 * pi / T_wave;
kw = ww^2 / g;
lambda_w = 2 * pi / kw;
kd_deep = kw * water_depth;
tanh_kd = tanh(kd_deep);

Ac = pi * D_e^2 / 4;
Me = m_b + rho_water * (C_M - 1) * Ac * L_w;
Kh = rho_water * g * Ac;
Ch = 2 * zeta * sqrt(Kh * Me);
CDp = 0.5 * rho_water * C_D * D_e * L_w;
wn_heave = sqrt(Kh / Me);

lambda_c = c0 / fc;
da = lambda_c / 2;
kappa_a = 2 * pi * da / lambda_c;
Nant = NantSide^2;
n_ant_idx = (0:NantSide-1).';
chi2 = chi * chi;

alpha_tilde = lambda_c / (4 * pi);
ds = sqrt(P_b * alpha_tilde^2 / (sigma_c2 * (2^R_s - 1)));
ds_safe = ds - commMargin;
rate_at_ds = log2(1 + P_b * alpha_tilde^2 / (ds^2 * sigma_c2));

Ts2 = Ts * Ts;
halfTs2 = 0.5 * Ts2;
dt_sub = Ts / nSub;
t_grid = (0:N-1) * Ts;
step_max = Vmax * Ts;

sx2 = sigma_ax^2;
sz2 = sigma_az^2;
Qs = zeros(4, 4);
Qs(1, 1) = (Ts^4) / 4 * sx2;
Qs(1, 3) = (Ts^3) / 2 * sx2;
Qs(3, 1) = Qs(1, 3);
Qs(3, 3) = Ts^2 * sx2;
Qs(2, 2) = (Ts^4) / 4 * sz2;
Qs(2, 4) = (Ts^3) / 2 * sz2;
Qs(4, 2) = Qs(2, 4);
Qs(4, 4) = Ts^2 * sz2;
Qs = (Qs + Qs') / 2;

E = [1, 0; 0, 1; 0, 0; 0, 0];
Dm_diag = [0.02^2, 0.02^2, (1e-9)^2];
Dm = diag(Dm_diag);
Dm_is = diag(1 ./ sqrt(Dm_diag));

Fcv = [1, 0, Ts, 0; ...
       0, 1, 0, Ts; ...
       0, 0, 1, 0; ...
       0, 0, 0, 1];

s_true0 = zeros(4, K);
s_true0(1, :) = x_buoy0;
s_true0(2, :) = z_buoy0;
s_hat0 = s_true0 + s_err0;
M0 = zeros(4, 4, K);
for k = 1:K
    M0(:, :, k) = diag([sig_x0(k)^2, sig_z0(k)^2, sig_vx0(k)^2, sig_vz0(k)^2]);
end

fprintf('\n========== 参数与深水条件 ==========\n');
fprintf('波浪角频率 omega_w = %.6f rad/s\n', ww);
fprintf('深水波数 k_w = %.6f rad/m，波长 lambda_w = %.4f m\n', kw, lambda_w);
fprintf('k_w * d = %.4f，tanh(k_w*d) = %.8f\n', kd_deep, tanh_kd);
if kd_deep > 3
    fprintf('深水条件 k_w*d >> 1：成立。\n');
else
    fprintf('深水条件 k_w*d >> 1：不成立。\n');
end
fprintf('等效面积 Ac = %.4f m^2，等效质量 Me = %.4f kg\n', Ac, Me);
fprintf('恢复刚度 Kh = %.4f N/m，阻尼 Ch = %.4f N/(m/s)，CDp = %.4f\n', Kh, Ch, CDp);
fprintf('垂荡固有角频率 = %.4f rad/s（周期 %.4f s）\n', wn_heave, 2 * pi / wn_heave);
fprintf('载波波长 lambda_c = %.6e m，阵元间距 da = %.6e m，kappa_a = %.12f\n', ...
    lambda_c, da, kappa_a);
fprintf('有效通信距离 d_s = %.6f m（理想波束对准）\n', ds);
fprintf('通信安全阈值 d_s - Delta_d = %.6f m，Delta_d = %.2f m\n', ds_safe, commMargin);
fprintf('无人机高度 z_U = %.2f m，初始水平位置 = (%.2f, %.2f) m\n', z_U, q_U0(1), q_U0(2));
fprintf('d = d_s 且 |omega|^2 = 1 时的频谱效率 = %.8f bit/(s*Hz)，阈值 R_s = %.4f\n', ...
    rate_at_ds, R_s);

init_dist_ok = true;
fprintf('\n========== 初始位置通信安全余量 ==========\n');
for k = 1:K
    dx0 = q_U0(1) - s_hat0(1, k);
    dy0 = q_U0(2) - y_buoy(k);
    dz0 = z_U - s_hat0(2, k);
    d0_pred = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
    fprintf('浮标 %d 初始预测距离 = %.4f m，安全阈值 = %.4f m\n', k, d0_pred, ds_safe);
    if d0_pred > ds_safe
        init_dist_ok = false;
        fprintf('错误: 初始位置不满足 d_pred <= d_s-Delta_d，浮标 %d 超出 %.4f m。第 1 帧不会移动无人机来掩盖。\n', ...
            k, d0_pred - ds_safe);
    end
end

%% 启动自检：阵列范数、半波长、雅可比、通信信噪比只衰减一次
ux_test = 0.25;
uy_test = -0.15;
a_x_test = exp(-1i * kappa_a * ux_test * n_ant_idx);
a_y_test = exp(-1i * kappa_a * uy_test * n_ant_idx);
a_test = kron(a_x_test, a_y_test) / NantSide;
a_dim_ok = (numel(a_test) == 16) && (Nant == 16);
a_norm_ok = abs(norm(a_test) - 1) < 1e-10;
da_ok = abs(da - lambda_c / 2) < 1e-12;
kappa_ok = abs(kappa_a - pi) < 1e-10;

d_chk = 30;
beam_chk = 0.8;
alpha_chk = (alpha_tilde / d_chk) * exp(-1i * 2 * pi * d_chk / lambda_c);
snr_once = P_b * (abs(alpha_chk)^2) * beam_chk / sigma_c2;
snr_ref = P_b * (alpha_tilde^2) * beam_chk / (d_chk^2 * sigma_c2);
snr_double = P_b * (alpha_tilde^2) * beam_chk / (d_chk^4 * sigma_c2);
pathloss_once_ok = abs(snr_once - snr_ref) / snr_ref < 1e-10;
pathloss_not_double = abs(snr_once - snr_double) / snr_double > 10;
ds_rate_ok = abs(rate_at_ds - R_s) < 1e-8;

% 用浮标 2（与无人机初始 y 相同）暴露印刷雅可比的矛盾，并用中心差分核对解析式。
k_jac = 2;
xU_jac = q_U0(1);
yU_jac = q_U0(2);
x_jac = x_buoy0(k_jac);
z_jac = z_buoy0;
y_jac = y_buoy(k_jac);
dx_jac = xU_jac - x_jac;
dy_jac = yU_jac - y_jac;
dz_jac = z_U - z_jac;
d_jac = sqrt(dx_jac^2 + dy_jac^2 + dz_jac^2);
d3_jac = d_jac^3;
H_analytic = zeros(3, 4);
H_analytic(1, 1) =  kappa_a * (dy_jac^2 + dz_jac^2) / d3_jac;
H_analytic(1, 2) = -kappa_a * dx_jac * dz_jac / d3_jac;
H_analytic(2, 1) = -kappa_a * dx_jac * dy_jac / d3_jac;
H_analytic(2, 2) = -kappa_a * dy_jac * dz_jac / d3_jac;
H_analytic(3, 1) = -2 * dx_jac / (c0 * d_jac);
H_analytic(3, 2) = -2 * dz_jac / (c0 * d_jac);
H22_paper = -kappa_a * (dx_jac^2 + dy_jac^2) / d3_jac;

eps_h = 1e-5;
s_jac = [x_jac; z_jac; 0; 0];
H_fd = zeros(3, 4);
for jj = 1:4
    s_plus = s_jac;
    s_minus = s_jac;
    s_plus(jj) = s_plus(jj) + eps_h;
    s_minus(jj) = s_minus(jj) - eps_h;
    dx_p = xU_jac - s_plus(1);
    dz_p = z_U - s_plus(2);
    d_p = sqrt(dx_p^2 + dy_jac^2 + dz_p^2);
    h_plus = [-kappa_a * dx_p / d_p; -kappa_a * dy_jac / d_p; 2 * d_p / c0];
    dx_m = xU_jac - s_minus(1);
    dz_m = z_U - s_minus(2);
    d_m = sqrt(dx_m^2 + dy_jac^2 + dz_m^2);
    h_minus = [-kappa_a * dx_m / d_m; -kappa_a * dy_jac / d_m; 2 * d_m / c0];
    H_fd(:, jj) = (h_plus - h_minus) / (2 * eps_h);
end
H_fd_err = max(abs(H_analytic(:) - H_fd(:)));
H_fd_ok = H_fd_err < 1e-8;

fprintf('\n========== 测量雅可比核对（浮标 2，dy = %.4f m）==========\n', dy_jac);
fprintf('解析 H(2,2) = %.6e\n', H_analytic(2, 2));
fprintf('中心差分 H(2,2) = %.6e\n', H_fd(2, 2));
fprintf('文中印刷 H(2,2) = %.6e\n', H22_paper);
fprintf('解析矩阵与中心差分的最大绝对误差 = %.3e\n', H_fd_err);
fprintf('印刷式与中心差分之差 = %.6e\n', H22_paper - H_fd(2, 2));
fprintf('后续 EKF 与信息矩阵使用解析导数，观测方程保持不变。\n');

%% 3. 浮标真实轨迹：子步长二阶 Heun，仅真实轨迹含确定性扰动
x_true = zeros(K, N);
z_true = zeros(K, N);
vx_true = zeros(K, N);
vz_true = zeros(K, N);
eta_true = zeros(K, N);
x_true(:, 1) = x_buoy0(:);
z_true(:, 1) = z_buoy0;
for k = 1:K
    eta_true(k, 1) = a * cos(kw * x_true(k, 1) - ww * t_grid(1));
end

for n = 2:N
    for k = 1:K
        x_w = x_true(k, n - 1);
        z_w = z_true(k, n - 1);
        vx_w = vx_true(k, n - 1);
        vz_w = vz_true(k, n - 1);
        k_buoy = k;
        for sub = 1:nSub
            x1 = x_w;
            z1 = z_w;
            vx1 = vx_w;
            vz1 = vz_w;
            t1 = t_grid(n - 1) + (sub - 1) * dt_sub;
            Ax_stage = zeros(1, 2);
            Az_stage = zeros(1, 2);
            vx_stage = zeros(1, 2);
            vz_stage = zeros(1, 2);
            for stage = 1:2
                if stage == 1
                    t_w = t1;
                    x_w = x1;
                    z_w = z1;
                    vx_w = vx1;
                    vz_w = vz1;
                else
                    t_w = t1 + dt_sub;
                    x_w = x1 + dt_sub * vx1;
                    z_w = z1 + dt_sub * vz1;
                    vx_w = vx1 + dt_sub * Ax_stage(1);
                    vz_w = vz1 + dt_sub * Az_stage(1);
                end
                disturb_flag = 1;
                phase_w = kw * x_w - ww * t_w;
                eta_w = a * cos(phase_w);
                etad_w = a * ww * sin(phase_w);
                Ez_w = (2 / kw) * exp(kw * z_w) * sinh(kw * L_w / 2);
                ubar_w = (a * ww / L_w) * Ez_w * cos(phase_w);
                wbar_w = (a * ww / L_w) * Ez_w * sin(phase_w);
                axbar_w = (a * ww * ww / L_w) * Ez_w * sin(phase_w);
                azbar_w = -(a * ww * ww / L_w) * Ez_w * cos(phase_w);
                uxrel_w = ubar_w + U_c - vx_w;
                uzrel_w = wbar_w - vz_w;
                Vrel_w = sqrt(uxrel_w * uxrel_w + uzrel_w * uzrel_w + eps_v * eps_v);
                FxI_w = rho_water * C_M * Ac * L_w * axbar_w;
                FzI_w = rho_water * C_M * Ac * L_w * azbar_w;
                FxD_w = CDp * Vrel_w * uxrel_w;
                FzD_w = CDp * Vrel_w * uzrel_w;
                FzR_w = -Kh * (z_w - z0 - eta_w) - Ch * (vz_w - etad_w);
                Ax_w = (FxI_w + FxD_w) / Me;
                Az_w = (FzI_w + FzD_w + FzR_w) / Me;
                Ax_stage(stage) = Ax_w;
                Az_stage(stage) = Az_w;
                vx_stage(stage) = vx_w;
                vz_stage(stage) = vz_w;
            end
            x_w = x1 + 0.5 * dt_sub * (vx_stage(1) + vx_stage(2));
            z_w = z1 + 0.5 * dt_sub * (vz_stage(1) + vz_stage(2));
            vx_w = vx1 + 0.5 * dt_sub * (Ax_stage(1) + Ax_stage(2));
            vz_w = vz1 + 0.5 * dt_sub * (Az_stage(1) + Az_stage(2));
        end
        x_true(k, n) = x_w;
        z_true(k, n) = z_w;
        vx_true(k, n) = vx_w;
        vz_true(k, n) = vz_w;
        eta_true(k, n) = a * cos(kw * x_true(k, n) - ww * t_grid(n));
    end
end

truth_ok = all(isfinite(x_true(:))) && all(isfinite(z_true(:))) && ...
    max(z_true(:)) < 0 && min(z_true(:)) > -5 && max(abs(x_true(:))) < 80;
fprintf('\n真实轨迹末状态 x = [%.4f, %.4f, %.4f] m\n', x_true(:, end));
fprintf('真实轨迹末状态 z = [%.4f, %.4f, %.4f] m\n', z_true(:, end));
fprintf('水平位置范围 [%.3f, %.3f] m，竖直位置范围 [%.3f, %.3f] m\n', ...
    min(x_true(:)), max(x_true(:)), min(z_true(:)), max(z_true(:)));

%% 4. EKF 初始化与名义闭环记录
s_post = zeros(4, K, 3);
M_post = zeros(4, 4, K, 3);
for sch = 1:3
    for k = 1:K
        s_post(:, k, sch) = s_hat0(:, k);
        M_post(:, :, k, sch) = M0(:, :, k);
    end
end
q_prev_all = [q_U0, q_U0, q_U0];

pred_hist = zeros(4, K, N, 3);
post_hist = zeros(4, K, N, 3);
qU_hist = zeros(2, N, 3);
p_hist = zeros(K, N, 3);
t_obj = zeros(N, 3);
rate_hist = zeros(K, N, 3);
beam_hist = zeros(K, N, 3);
rho_hist = zeros(K, N, 3);
step_hist = zeros(1, N);
n_feas_coarse = zeros(1, N);
grid_demo = zeros(2, 0);

n_cov_fix = 0;
max_jitter = 0;
n_nonfinite = 0;
n_skip_update = 0;
n_power_violate = 0;
n_speed_violate = 0;
n_infeasible = 0;
n_true_violate = zeros(1, 3);
n_rate_violate = zeros(1, 3);
n_equiv_violate = zeros(1, 3);
n_pred_violate_C = 0;
n_nonmono = 0;
n_Syy_fix = 0;
snr_gap_max = 0;
beam_overshoot = 0;
G_check = eye(4);
G_checked = false;

fprintf('\n========== 名义闭环：三方案 EKF + 位置/功率优化 ==========\n');
tic;

for n = 1:N
    s_pred = zeros(4, K, 3);
    M_pred = zeros(4, 4, K, 3);

    % ---------- 8. 对三个方案做一步预测（第 1 帧使用初始先验） ----------
    for sch = 1:3
        if n == 1
            s_pred(:, :, sch) = s_post(:, :, sch);
            M_pred(:, :, :, sch) = M_post(:, :, :, sch);
        else
            t_pred = t_grid(n - 1);
            for k = 1:K
                s_in = s_post(:, k, sch);
                M_in = M_post(:, :, k, sch);
                k_buoy = k;
                Ax_samples = zeros(1, 9);
                Az_samples = zeros(1, 9);
                for sample_id = 1:9
                    x_w = s_in(1);
                    z_w = s_in(2);
                    vx_w = s_in(3);
                    vz_w = s_in(4);
                    if sample_id > 1
                        pair_id = sample_id - 1;
                        jj = floor((pair_id - 1) / 2) + 1;
                        if mod(pair_id, 2) == 1
                            sgn = 1.0;
                        else
                            sgn = -1.0;
                        end
                        if jj == 1
                            x_w = s_in(1) + sgn * epsFD(1);
                        elseif jj == 2
                            z_w = s_in(2) + sgn * epsFD(2);
                        elseif jj == 3
                            vx_w = s_in(3) + sgn * epsFD(3);
                        else
                            vz_w = s_in(4) + sgn * epsFD(4);
                        end
                    end
                    t_w = t_pred;
                    disturb_flag = 0;
                    phase_w = kw * x_w - ww * t_w;
                    eta_w = a * cos(phase_w);
                    etad_w = a * ww * sin(phase_w);
                    Ez_w = (2 / kw) * exp(kw * z_w) * sinh(kw * L_w / 2);
                    ubar_w = (a * ww / L_w) * Ez_w * cos(phase_w);
                    wbar_w = (a * ww / L_w) * Ez_w * sin(phase_w);
                    axbar_w = (a * ww * ww / L_w) * Ez_w * sin(phase_w);
                    azbar_w = -(a * ww * ww / L_w) * Ez_w * cos(phase_w);
                    uxrel_w = ubar_w + U_c - vx_w;
                    uzrel_w = wbar_w - vz_w;
                    Vrel_w = sqrt(uxrel_w * uxrel_w + uzrel_w * uzrel_w + eps_v * eps_v);
                    FxI_w = rho_water * C_M * Ac * L_w * axbar_w;
                    FzI_w = rho_water * C_M * Ac * L_w * azbar_w;
                    FxD_w = CDp * Vrel_w * uxrel_w;
                    FzD_w = CDp * Vrel_w * uzrel_w;
                    FzR_w = -Kh * (z_w - z0 - eta_w) - Ch * (vz_w - etad_w);
                    Ax_w = (FxI_w + FxD_w) / Me;
                    Az_w = (FzI_w + FzD_w + FzR_w) / Me;
                    % 名义预测与 Monte Carlo 滤波器都不加入额外确定性扰动。
                    Ax_samples(sample_id) = Ax_w;
                    Az_samples(sample_id) = Az_w;
                end
                Ax_nom = Ax_samples(1);
                Az_nom = Az_samples(1);
                dAx_ds = zeros(1, 4);
                dAz_ds = zeros(1, 4);
                for jj = 1:4
                    i_plus = 2 * jj;
                    i_minus = 2 * jj + 1;
                    dAx_ds(jj) = (Ax_samples(i_plus) - Ax_samples(i_minus)) / (2 * epsFD(jj));
                    dAz_ds(jj) = (Az_samples(i_plus) - Az_samples(i_minus)) / (2 * epsFD(jj));
                end
                s_out = zeros(4, 1);
                s_out(1) = s_in(1) + Ts * s_in(3) + halfTs2 * Ax_nom;
                s_out(2) = s_in(2) + Ts * s_in(4) + halfTs2 * Az_nom;
                s_out(3) = s_in(3) + Ts * Ax_nom;
                s_out(4) = s_in(4) + Ts * Az_nom;
                Gmat = zeros(4, 4);
                Gmat(1, 1) = 1 + halfTs2 * dAx_ds(1);
                Gmat(1, 2) =     halfTs2 * dAx_ds(2);
                Gmat(1, 3) = Ts + halfTs2 * dAx_ds(3);
                Gmat(1, 4) =      halfTs2 * dAx_ds(4);
                Gmat(2, 1) =      halfTs2 * dAz_ds(1);
                Gmat(2, 2) = 1 + halfTs2 * dAz_ds(2);
                Gmat(2, 3) =      halfTs2 * dAz_ds(3);
                Gmat(2, 4) = Ts + halfTs2 * dAz_ds(4);
                Gmat(3, 1) = Ts * dAx_ds(1);
                Gmat(3, 2) = Ts * dAx_ds(2);
                Gmat(3, 3) = 1 + Ts * dAx_ds(3);
                Gmat(3, 4) = Ts * dAx_ds(4);
                Gmat(4, 1) = Ts * dAz_ds(1);
                Gmat(4, 2) = Ts * dAz_ds(2);
                Gmat(4, 3) = Ts * dAz_ds(3);
                Gmat(4, 4) = 1 + Ts * dAz_ds(4);
                M_out = Gmat * M_in * Gmat' + Qs;
                M_out = (M_out + M_out') / 2;
                evM = eig(M_out);
                min_ev = min(real(evM));
                if min_ev < 0
                    jitter_now = 1e-12 - min_ev;
                    M_out = M_out + jitter_now * I4;
                    M_out = (M_out + M_out') / 2;
                    n_cov_fix = n_cov_fix + 1;
                    if jitter_now > max_jitter
                        max_jitter = jitter_now;
                    end
                end
                s_pred(:, k, sch) = s_out;
                M_pred(:, :, k, sch) = M_out;
                if ~G_checked && sch == 3 && k == 1
                    G_check = Gmat;
                    G_checked = true;
                end
            end
        end
    end

    q_slot = zeros(2, 3);
    p_slot = zeros(K, 3);
    t_slot = zeros(1, 3);
    geom_dx = zeros(K, 3);
    geom_dy = zeros(K, 3);
    geom_dz = zeros(K, 3);
    geom_d = zeros(K, 3);
    geom_rho = zeros(K, 3);
    geom_H = zeros(3, 4, K, 3);

    % ---------- 6-7. 候选位置搜索 + Schur 补等价的功率二分 ----------
    for sch = 1:3
        J0_sch = zeros(4, 4, K);
        for k = 1:K
            Mloc = (M_pred(:, :, k, sch) + M_pred(:, :, k, sch)') / 2;
            if rcond(Mloc) < 1e-12
                Mloc = Mloc + 1e-10 * I4;
            end
            J0_loc = Mloc \ I4;
            J0_sch(:, :, k) = (J0_loc + J0_loc') / 2;
        end

        found = false;
        best_t = inf;
        best_move = inf;
        best_q = q_prev_all(:, sch);
        best_p = (P_U / K) * ones(K, 1);
        best_d = zeros(K, 1);
        best_rho = zeros(K, 1);
        best_dx = zeros(K, 1);
        best_dy = zeros(K, 1);
        best_dz = zeros(K, 1);
        best_H = zeros(3, 4, K);
        hold_saved = false;
        hold_d = zeros(K, 1);
        hold_rho = zeros(K, 1);
        hold_dx = zeros(K, 1);
        hold_dy = zeros(K, 1);
        hold_dz = zeros(K, 1);
        hold_H = zeros(3, 4, K);
        hold_A = zeros(4, 4, K);
        best_infeas_d = inf;

        if sch == 3 && n > 1
            n_pass = 2;
        else
            n_pass = 1;
        end

        for pass = 1:n_pass
            if pass == 1
                if sch < 3 || (sch == 3 && n == 1)
                    cand_q = q_U0;
                else
                    cand_q = zeros(2, 0);
                    nHalfC = floor(step_max / gridStepCoarse + 1e-9);
                    for ix = -nHalfC:nHalfC
                        for iy = -nHalfC:nHalfC
                            dxg = ix * gridStepCoarse;
                            dyg = iy * gridStepCoarse;
                            if dxg * dxg + dyg * dyg <= step_max * step_max + 1e-8
                                qx = q_prev_all(1, sch) + dxg;
                                qy = q_prev_all(2, sch) + dyg;
                                if qx >= x_min && qx <= x_max && qy >= y_min && qy <= y_max
                                    cand_q(:, end + 1) = [qx; qy]; %#ok<AGROW>
                                end
                            end
                        end
                    end
                end
            else
                cand_q = zeros(2, 0);
                fine_off = [-gridStepFine, 0, gridStepFine];
                for ix = 1:3
                    for iy = 1:3
                        qx = best_q(1) + fine_off(ix);
                        qy = best_q(2) + fine_off(iy);
                        dxg = qx - q_prev_all(1, sch);
                        dyg = qy - q_prev_all(2, sch);
                        if dxg * dxg + dyg * dyg <= step_max * step_max + 1e-8
                            if qx >= x_min && qx <= x_max && qy >= y_min && qy <= y_max
                                cand_q(:, end + 1) = [qx; qy]; %#ok<AGROW>
                            end
                        end
                    end
                end
            end

            nC = size(cand_q, 2);
            for ic = 1:nC
                qx_g = cand_q(1, ic);
                qy_g = cand_q(2, ic);
                H_cand = zeros(3, 4, K);
                A_cand = zeros(4, 4, K);
                d_cand = zeros(K, 1);
                rho_cand = zeros(K, 1);
                dx_cand = zeros(K, 1);
                dy_cand = zeros(K, 1);
                dz_cand = zeros(K, 1);
                for k = 1:K
                    dx_g = qx_g - s_pred(1, k, sch);
                    dy_g = qy_g - y_buoy(k);
                    dz_g = z_U - s_pred(2, k, sch);
                    d_g = sqrt(dx_g * dx_g + dy_g * dy_g + dz_g * dz_g);
                    d3_g = d_g * d_g * d_g;
                    H_g = zeros(3, 4);
                    H_g(1, 1) =  kappa_a * (dy_g * dy_g + dz_g * dz_g) / d3_g;
                    H_g(1, 2) = -kappa_a * dx_g * dz_g / d3_g;
                    H_g(2, 1) = -kappa_a * dx_g * dy_g / d3_g;
                    H_g(2, 2) = -kappa_a * dy_g * dz_g / d3_g;
                    H_g(3, 1) = -2 * dx_g / (c0 * d_g);
                    H_g(3, 2) = -2 * dz_g / (c0 * d_g);
                    Hs_g = Dm_is * H_g;
                    A_g = Hs_g' * Hs_g;
                    A_g = (A_g + A_g') / 2;
                    rho_g = chi2 / (4 * d_g * d_g * sigma_r2);
                    H_cand(:, :, k) = H_g;
                    A_cand(:, :, k) = A_g;
                    d_cand(k) = d_g;
                    rho_cand(k) = rho_g;
                    dx_cand(k) = dx_g;
                    dy_cand(k) = dy_g;
                    dz_cand(k) = dz_g;
                end

                if norm([qx_g - q_prev_all(1, sch); qy_g - q_prev_all(2, sch)]) <= 1e-8
                    hold_saved = true;
                    hold_d = d_cand;
                    hold_rho = rho_cand;
                    hold_dx = dx_cand;
                    hold_dy = dy_cand;
                    hold_dz = dz_cand;
                    hold_H = H_cand;
                    hold_A = A_cand;
                end
                max_d_cand = max(d_cand);
                if max_d_cand < best_infeas_d
                    best_infeas_d = max_d_cand;
                end

                comm_ok = all(d_cand <= ds_safe);
                if sch == 3 && pass == 1 && n == n_demo && comm_ok
                    grid_demo(:, end + 1) = [qx_g; qy_g]; %#ok<AGROW>
                    n_feas_coarse(n) = n_feas_coarse(n) + 1;
                elseif sch == 3 && pass == 1 && comm_ok
                    n_feas_coarse(n) = n_feas_coarse(n) + 1;
                end

                accept = true;
                if sch == 3 && n > 1 && ~comm_ok
                    accept = false;
                end
                if ~accept
                    continue;
                end

                if sch == 1
                    p_use = (P_U / K) * ones(K, 1);
                else
                    % 原问题是 min t, s.t. trace(E' J(p_k)^{-1} E) <= t。
                    % 由 Schur 补，该约束等价于存在 W 使 trace(W)<=t 且
                    % [J, E; E', W] 半正定，J = J0 + p*rho*A。
                    % 固定无人机位置后，各浮标目标只依赖自身功率，耦合仅来自
                    % sum(p)<=P_U。外层二分 t，内层求每个浮标的最小可行功率，
                    % 是该凸问题在给定容差下的精确数值解，不是启发式。
                    f0 = zeros(K, 1);
                    fP = zeros(K, 1);
                    for k = 1:K
                        Jb = (J0_sch(:, :, k) + J0_sch(:, :, k)') / 2;
                        f0(k) = real(trace(E' * (Jb \ E)));
                        JP = Jb + P_U * rho_cand(k) * A_cand(:, :, k);
                        JP = (JP + JP') / 2;
                        fP(k) = real(trace(E' * (JP \ E)));
                        if fP(k) > f0(k) + 1e-8 * max(1, abs(f0(k)))
                            n_nonmono = n_nonmono + 1;
                        end
                    end
                    t_lo = max(fP);
                    t_hi = max(f0);
                    if t_lo > t_hi
                        t_lo = t_hi;
                    end
                    p_use = zeros(K, 1);
                    for io = 1:outerIter
                        t_mid = 0.5 * (t_lo + t_hi);
                        p_try = zeros(K, 1);
                        feas = true;
                        for k = 1:K
                            if f0(k) <= t_mid
                                p_try(k) = 0;
                            elseif fP(k) > t_mid
                                feas = false;
                                break;
                            else
                                pL = 0;
                                pR = P_U;
                                for ii = 1:innerIter
                                    pM = 0.5 * (pL + pR);
                                    JM = J0_sch(:, :, k) + pM * rho_cand(k) * A_cand(:, :, k);
                                    JM = (JM + JM') / 2;
                                    fM = real(trace(E' * (JM \ E)));
                                    if fM <= t_mid
                                        pR = pM;
                                    else
                                        pL = pM;
                                    end
                                end
                                p_try(k) = pR;
                            end
                        end
                        if feas && sum(p_try) <= P_U + 1e-8
                            t_hi = t_mid;
                            p_use = p_try;
                        else
                            t_lo = t_mid;
                        end
                    end
                end

                t_achieved = 0;
                for k = 1:K
                    Jk = J0_sch(:, :, k) + p_use(k) * rho_cand(k) * A_cand(:, :, k);
                    Jk = (Jk + Jk') / 2;
                    fval = real(trace(E' * (Jk \ E)));
                    if fval > t_achieved
                        t_achieved = fval;
                    end
                end
                move_now = norm([qx_g - q_prev_all(1, sch); qy_g - q_prev_all(2, sch)]);
                update_pick = false;
                if ~found || t_achieved < best_t * (1 - eps_t)
                    update_pick = true;
                elseif t_achieved <= best_t * (1 + eps_t) && move_now < best_move
                    update_pick = true;
                end
                if update_pick
                    best_t = t_achieved;
                    best_move = move_now;
                    best_p = p_use;
                    best_q = [qx_g; qy_g];
                    best_d = d_cand;
                    best_rho = rho_cand;
                    best_dx = dx_cand;
                    best_dy = dy_cand;
                    best_dz = dz_cand;
                    best_H = H_cand;
                    found = true;
                end
            end
        end

        if ~found
            n_infeasible = n_infeasible + 1;
            fprintf('错误: 帧 %d 方案 %d 没有满足 d_pred <= d_s-Delta_d 的候选位置。最近的最大距离为 %.4f m，安全阈值 %.4f m。不把违约点当作可行解。\n', ...
                n, sch, best_infeas_d, ds_safe);
            if hold_saved
                f0 = zeros(K, 1);
                fP = zeros(K, 1);
                for k = 1:K
                    Jb = (J0_sch(:, :, k) + J0_sch(:, :, k)') / 2;
                    f0(k) = real(trace(E' * (Jb \ E)));
                    JP = Jb + P_U * hold_rho(k) * hold_A(:, :, k);
                    JP = (JP + JP') / 2;
                    fP(k) = real(trace(E' * (JP \ E)));
                end
                t_lo = max(fP);
                t_hi = max(f0);
                if t_lo > t_hi
                    t_lo = t_hi;
                end
                p_use = zeros(K, 1);
                for io = 1:outerIter
                    t_mid = 0.5 * (t_lo + t_hi);
                    p_try = zeros(K, 1);
                    feas = true;
                    for k = 1:K
                        if f0(k) <= t_mid
                            p_try(k) = 0;
                        elseif fP(k) > t_mid
                            feas = false;
                            break;
                        else
                            pL = 0;
                            pR = P_U;
                            for ii = 1:innerIter
                                pM = 0.5 * (pL + pR);
                                JM = J0_sch(:, :, k) + pM * hold_rho(k) * hold_A(:, :, k);
                                JM = (JM + JM') / 2;
                                fM = real(trace(E' * (JM \ E)));
                                if fM <= t_mid
                                    pR = pM;
                                else
                                    pL = pM;
                                end
                            end
                            p_try(k) = pR;
                        end
                    end
                    if feas && sum(p_try) <= P_U + 1e-8
                        t_hi = t_mid;
                        p_use = p_try;
                    else
                        t_lo = t_mid;
                    end
                end
                t_achieved = 0;
                for k = 1:K
                    Jk = J0_sch(:, :, k) + p_use(k) * hold_rho(k) * hold_A(:, :, k);
                    Jk = (Jk + Jk') / 2;
                    fval = real(trace(E' * (Jk \ E)));
                    if fval > t_achieved
                        t_achieved = fval;
                    end
                end
                best_t = t_achieved;
                best_p = p_use;
                best_q = q_prev_all(:, sch);
                best_d = hold_d;
                best_rho = hold_rho;
                best_dx = hold_dx;
                best_dy = hold_dy;
                best_dz = hold_dz;
                best_H = hold_H;
            else
                fprintf('错误: 帧 %d 方案 %d 连上一位置的几何都未生成，无法继续该帧的功率分配。\n', n, sch);
            end
        end

        q_slot(:, sch) = best_q;
        p_slot(:, sch) = best_p;
        t_slot(sch) = best_t;
        geom_d(:, sch) = best_d;
        geom_rho(:, sch) = best_rho;
        geom_dx(:, sch) = best_dx;
        geom_dy(:, sch) = best_dy;
        geom_dz(:, sch) = best_dz;
        geom_H(:, :, :, sch) = best_H;
        rho_hist(:, n, sch) = geom_rho(:, sch);

        psum = sum(best_p);
        if psum > P_U + 1e-6 || any(best_p < -1e-8)
            n_power_violate = n_power_violate + 1;
        end
        step_now = norm(best_q - q_prev_all(:, sch));
        if sch == 3
            if n == 1
                step_hist(n) = 0;
                q_slot(:, sch) = q_U0;
                best_q = q_U0;
            else
                step_hist(n) = step_now;
                if step_now > step_max + 1e-6
                    n_speed_violate = n_speed_violate + 1;
                end
            end
            for k = 1:K
                if best_d(k) > ds_safe + 1e-8
                    n_pred_violate_C = n_pred_violate_C + 1;
                end
            end
        end
        q_prev_all(:, sch) = best_q;
    end

    % ---------- 实际波束匹配与上行频谱效率（使用真实 |omega|^2） ----------
    for sch = 1:3
        for k = 1:K
            dx_t = q_slot(1, sch) - x_true(k, n);
            dy_t = q_slot(2, sch) - y_buoy(k);
            dz_t = z_U - z_true(k, n);
            d_t = sqrt(dx_t * dx_t + dy_t * dy_t + dz_t * dz_t);
            dx_p = q_slot(1, sch) - s_pred(1, k, sch);
            dy_p = q_slot(2, sch) - y_buoy(k);
            dz_p = z_U - s_pred(2, k, sch);
            d_p = sqrt(dx_p * dx_p + dy_p * dy_p + dz_p * dz_p);

            ux_t = dx_t / d_t;
            uy_t = dy_t / d_t;
            a_x_t = exp(-1i * kappa_a * ux_t * n_ant_idx);
            a_y_t = exp(-1i * kappa_a * uy_t * n_ant_idx);
            a_true = kron(a_x_t, a_y_t) / NantSide;

            ux_p = dx_p / d_p;
            uy_p = dy_p / d_p;
            a_x_p = exp(-1i * kappa_a * ux_p * n_ant_idx);
            a_y_p = exp(-1i * kappa_a * uy_p * n_ant_idx);
            a_pred = kron(a_x_p, a_y_p) / NantSide;

            omega = a_true' * a_pred;
            beam = abs(omega)^2;
            if beam > 1
                if beam - 1 > beam_overshoot
                    beam_overshoot = beam - 1;
                end
                if beam < 1 + 1e-8
                    beam = 1;
                end
            end
            alpha_c = (alpha_tilde / d_t) * exp(-1i * 2 * pi * d_t / lambda_c);
            snr_c = P_b * (abs(alpha_c)^2) * beam / sigma_c2;
            snr_ref_n = P_b * (alpha_tilde^2) * beam / (d_t * d_t * sigma_c2);
            gap_n = abs(snr_c - snr_ref_n);
            if gap_n > snr_gap_max
                snr_gap_max = gap_n;
            end
            rate_hist(k, n, sch) = log2(1 + snr_c);
            beam_hist(k, n, sch) = beam;
            if d_t > ds + 1e-6
                n_true_violate(sch) = n_true_violate(sch) + 1;
            end
            if rate_hist(k, n, sch) < R_s - 1e-6
                n_rate_violate(sch) = n_rate_violate(sch) + 1;
            end
            if d_t > ds * sqrt(max(beam, 0)) + 1e-6
                n_equiv_violate(sch) = n_equiv_violate(sch) + 1;
            end
        end
    end

    % ---------- 8. 测量更新。功率为零时不做更新，避免除零 ----------
    % 三个方案共用同一组标准正态向量，再按各自 R 着色，便于配对比较。
    % 缩放更新与原始 R、h 代数等价：S = Dm^{-1/2}，Rs = I/(p*rho)，
    % Hs = S*H。这样避免时延量纲 1e-18 与方向余弦量纲直接相加。
    for k = 1:K
        zdraw = randn(3, 1);
        for sch = 1:3
            p_m = p_slot(k, sch);
            rho_m = geom_rho(k, sch);
            if p_m <= 1e-12 || ~(rho_m > 0)
                s_post(:, k, sch) = s_pred(:, k, sch);
                M_post(:, :, k, sch) = M_pred(:, :, k, sch);
                n_skip_update = n_skip_update + 1;
            else
                H_m = geom_H(:, :, k, sch);
                dx_m = geom_dx(k, sch);
                dy_m = geom_dy(k, sch);
                dz_m = geom_dz(k, sch);
                d_m = geom_d(k, sch);
                h_pred_m = [-kappa_a * dx_m / d_m; -kappa_a * dy_m / d_m; 2 * d_m / c0];
                dx_t = q_slot(1, sch) - x_true(k, n);
                dy_t = q_slot(2, sch) - y_buoy(k);
                dz_t = z_U - z_true(k, n);
                d_t = sqrt(dx_t * dx_t + dy_t * dy_t + dz_t * dz_t);
                h_true_m = [-kappa_a * dx_t / d_t; -kappa_a * dy_t / d_t; 2 * d_t / c0];
                R_orig = diag(Dm_diag ./ (p_m * rho_m));
                R_orig = (R_orig + R_orig') / 2;
                noise_m = chol(R_orig, 'lower') * zdraw;
                y_meas = h_true_m + noise_m;

                Hs = Dm_is * H_m;
                hs = Dm_is * h_pred_m;
                ys = Dm_is * y_meas;
                Rs = I3 / (p_m * rho_m);
                M_in = M_pred(:, :, k, sch);
                Syy = Hs * M_in * Hs' + Rs;
                Syy = (Syy + Syy') / 2;
                if rcond(Syy) < 1e-12
                    Syy = Syy + 1e-10 * I3;
                    n_Syy_fix = n_Syy_fix + 1;
                end
                Kgain = (Syy \ (M_in * Hs')')';
                s_new = s_pred(:, k, sch) + Kgain * (ys - hs);
                Ikh = I4 - Kgain * Hs;
                M_new = Ikh * M_in * Ikh' + Kgain * Rs * Kgain';
                M_new = (M_new + M_new') / 2;
                evM = eig(M_new);
                min_ev = min(real(evM));
                if min_ev < 0
                    jitter_now = 1e-12 - min_ev;
                    M_new = M_new + jitter_now * I4;
                    M_new = (M_new + M_new') / 2;
                    n_cov_fix = n_cov_fix + 1;
                    if jitter_now > max_jitter
                        max_jitter = jitter_now;
                    end
                end
                if any(~isfinite(s_new)) || any(~isfinite(M_new(:)))
                    n_nonfinite = n_nonfinite + 1;
                end
                s_post(:, k, sch) = s_new;
                M_post(:, :, k, sch) = M_new;
            end
        end
    end

    for sch = 1:3
        pred_hist(:, :, n, sch) = s_pred(:, :, sch);
        post_hist(:, :, n, sch) = s_post(:, :, sch);
        qU_hist(:, n, sch) = q_slot(:, sch);
        p_hist(:, n, sch) = p_slot(:, sch);
    end
    t_obj(n, :) = t_slot;

    if mod(n, 10) == 0 || n == 1
        fprintf('帧 %02d/%02d  sqrt(t)=[%.3f %.3f %.3f]  qC=(%.2f, %.2f)  pC=[%.2f %.2f %.2f]\n', ...
            n, N, sqrt(t_slot(1)), sqrt(t_slot(2)), sqrt(t_slot(3)), ...
            q_slot(1, 3), q_slot(2, 3), p_slot(1, 3), p_slot(2, 3), p_slot(3, 3));
        if exist('OCTAVE_VERSION', 'builtin')
            fflush(stdout);
        end
    end
end
time_nominal = toc;
fprintf('名义闭环用时 %.2f s\n', time_nominal);

%% 9. Monte Carlo：冻结方案 C 的位置、功率和测量噪声强度
q_plan = qU_hist(:, :, 3);
p_plan = p_hist(:, :, 3);
rho_plan = rho_hist(:, :, 3);
err_p_sum_airy = zeros(1, N);
err_v_sum_airy = zeros(1, N);
err_p_sum_cv = zeros(1, N);
err_v_sum_cv = zeros(1, N);
theta_sum_airy = zeros(1, N);
theta_sum_cv = zeros(1, N);
err_p_buoy_airy = zeros(K, 1);
err_v_buoy_airy = zeros(K, 1);
err_p_buoy_cv = zeros(K, 1);
err_v_buoy_cv = zeros(K, 1);
sum_ex_airy = zeros(K, N);
sum_ez_airy = zeros(K, N);
init_err_sum = zeros(4, K);
init_err_outer = zeros(4, 4, K);
n_mc_init = 0;
w_outer = zeros(4, 4);
n_w = 0;

fprintf('\n========== Monte Carlo：Nmc = %d ==========\n', Nmc);
tic;
for imc = 1:Nmc
    s_true_mc = s_true0;
    s_airy = zeros(4, K);
    s_cv = zeros(4, K);
    M_airy = M0;
    M_cv = M0;
    for k = 1:K
        L0 = chol((M0(:, :, k) + M0(:, :, k)') / 2, 'lower');
        initial_error = L0 * randn(4, 1);
        s_airy(:, k) = s_true_mc(:, k) + initial_error;
        s_cv(:, k) = s_airy(:, k);
        init_err_sum(:, k) = init_err_sum(:, k) + initial_error;
        init_err_outer(:, :, k) = init_err_outer(:, :, k) + initial_error * initial_error';
        n_mc_init = n_mc_init + 1;
    end
    for n = 1:N
        if n > 1
            t_pred = t_grid(n - 1);
            for k = 1:K
                x_w = s_true_mc(1, k);
                z_w = s_true_mc(2, k);
                vx_w = s_true_mc(3, k);
                vz_w = s_true_mc(4, k);
                k_buoy = k;
                for sub = 1:nSub
                    x1 = x_w;
                    z1 = z_w;
                    vx1 = vx_w;
                    vz1 = vz_w;
                    t1 = t_grid(n - 1) + (sub - 1) * dt_sub;
                    Ax_stage = zeros(1, 2);
                    Az_stage = zeros(1, 2);
                    vx_stage = zeros(1, 2);
                    vz_stage = zeros(1, 2);
                    for stage = 1:2
                        if stage == 1
                            t_w = t1;
                            x_w = x1;
                            z_w = z1;
                            vx_w = vx1;
                            vz_w = vz1;
                        else
                            t_w = t1 + dt_sub;
                            x_w = x1 + dt_sub * vx1;
                            z_w = z1 + dt_sub * vz1;
                            vx_w = vx1 + dt_sub * Ax_stage(1);
                            vz_w = vz1 + dt_sub * Az_stage(1);
                        end
                        phase_w = kw * x_w - ww * t_w;
                        eta_w = a * cos(phase_w);
                        etad_w = a * ww * sin(phase_w);
                        Ez_w = (2 / kw) * exp(kw * z_w) * sinh(kw * L_w / 2);
                        ubar_w = (a * ww / L_w) * Ez_w * cos(phase_w);
                        wbar_w = (a * ww / L_w) * Ez_w * sin(phase_w);
                        axbar_w = (a * ww * ww / L_w) * Ez_w * sin(phase_w);
                        azbar_w = -(a * ww * ww / L_w) * Ez_w * cos(phase_w);
                        uxrel_w = ubar_w + U_c - vx_w;
                        uzrel_w = wbar_w - vz_w;
                        Vrel_w = sqrt(uxrel_w * uxrel_w + uzrel_w * uzrel_w + eps_v * eps_v);
                        FxI_w = rho_water * C_M * Ac * L_w * axbar_w;
                        FzI_w = rho_water * C_M * Ac * L_w * azbar_w;
                        FxD_w = CDp * Vrel_w * uxrel_w;
                        FzD_w = CDp * Vrel_w * uzrel_w;
                        FzR_w = -Kh * (z_w - z0 - eta_w) - Ch * (vz_w - etad_w);
                        Ax_w = (FxI_w + FxD_w) / Me;
                        Az_w = (FzI_w + FzD_w + FzR_w) / Me;
                        Ax_stage(stage) = Ax_w;
                        Az_stage(stage) = Az_w;
                        vx_stage(stage) = vx_w;
                        vz_stage(stage) = vz_w;
                    end
                    x_w = x1 + 0.5 * dt_sub * (vx_stage(1) + vx_stage(2));
                    z_w = z1 + 0.5 * dt_sub * (vz_stage(1) + vz_stage(2));
                    vx_w = vx1 + 0.5 * dt_sub * (Ax_stage(1) + Ax_stage(2));
                    vz_w = vz1 + 0.5 * dt_sub * (Az_stage(1) + Az_stage(2));
                end
                xi_x = randn;
                xi_z = randn;
                w_mc = [0.5 * Ts2 * sigma_ax * xi_x; ...
                        0.5 * Ts2 * sigma_az * xi_z; ...
                        Ts * sigma_ax * xi_x; ...
                        Ts * sigma_az * xi_z];
                s_true_mc(1, k) = x_w + w_mc(1);
                s_true_mc(2, k) = z_w + w_mc(2);
                s_true_mc(3, k) = vx_w + w_mc(3);
                s_true_mc(4, k) = vz_w + w_mc(4);
                w_outer = w_outer + w_mc * w_mc';
                n_w = n_w + 1;

                s_in = s_airy(:, k);
                M_in = M_airy(:, :, k);
                k_buoy = k;
                Ax_samples = zeros(1, 9);
                Az_samples = zeros(1, 9);
                for sample_id = 1:9
                    x_w = s_in(1);
                    z_w = s_in(2);
                    vx_w = s_in(3);
                    vz_w = s_in(4);
                    if sample_id > 1
                        pair_id = sample_id - 1;
                        jj = floor((pair_id - 1) / 2) + 1;
                        if mod(pair_id, 2) == 1
                            sgn = 1.0;
                        else
                            sgn = -1.0;
                        end
                        if jj == 1
                            x_w = s_in(1) + sgn * epsFD(1);
                        elseif jj == 2
                            z_w = s_in(2) + sgn * epsFD(2);
                        elseif jj == 3
                            vx_w = s_in(3) + sgn * epsFD(3);
                        else
                            vz_w = s_in(4) + sgn * epsFD(4);
                        end
                    end
                    t_w = t_pred;
                    disturb_flag = 0;
                    phase_w = kw * x_w - ww * t_w;
                    eta_w = a * cos(phase_w);
                    etad_w = a * ww * sin(phase_w);
                    Ez_w = (2 / kw) * exp(kw * z_w) * sinh(kw * L_w / 2);
                    ubar_w = (a * ww / L_w) * Ez_w * cos(phase_w);
                    wbar_w = (a * ww / L_w) * Ez_w * sin(phase_w);
                    axbar_w = (a * ww * ww / L_w) * Ez_w * sin(phase_w);
                    azbar_w = -(a * ww * ww / L_w) * Ez_w * cos(phase_w);
                    uxrel_w = ubar_w + U_c - vx_w;
                    uzrel_w = wbar_w - vz_w;
                    Vrel_w = sqrt(uxrel_w * uxrel_w + uzrel_w * uzrel_w + eps_v * eps_v);
                    FxI_w = rho_water * C_M * Ac * L_w * axbar_w;
                    FzI_w = rho_water * C_M * Ac * L_w * azbar_w;
                    FxD_w = CDp * Vrel_w * uxrel_w;
                    FzD_w = CDp * Vrel_w * uzrel_w;
                    FzR_w = -Kh * (z_w - z0 - eta_w) - Ch * (vz_w - etad_w);
                    Ax_w = (FxI_w + FxD_w) / Me;
                    Az_w = (FzI_w + FzD_w + FzR_w) / Me;
                    % 名义预测与 Monte Carlo 滤波器都不加入额外确定性扰动。
                    Ax_samples(sample_id) = Ax_w;
                    Az_samples(sample_id) = Az_w;
                end
                Ax_nom = Ax_samples(1);
                Az_nom = Az_samples(1);
                dAx_ds = zeros(1, 4);
                dAz_ds = zeros(1, 4);
                for jj = 1:4
                    i_plus = 2 * jj;
                    i_minus = 2 * jj + 1;
                    dAx_ds(jj) = (Ax_samples(i_plus) - Ax_samples(i_minus)) / (2 * epsFD(jj));
                    dAz_ds(jj) = (Az_samples(i_plus) - Az_samples(i_minus)) / (2 * epsFD(jj));
                end
                s_out = zeros(4, 1);
                s_out(1) = s_in(1) + Ts * s_in(3) + halfTs2 * Ax_nom;
                s_out(2) = s_in(2) + Ts * s_in(4) + halfTs2 * Az_nom;
                s_out(3) = s_in(3) + Ts * Ax_nom;
                s_out(4) = s_in(4) + Ts * Az_nom;
                Gmat = zeros(4, 4);
                Gmat(1, 1) = 1 + halfTs2 * dAx_ds(1);
                Gmat(1, 2) =     halfTs2 * dAx_ds(2);
                Gmat(1, 3) = Ts + halfTs2 * dAx_ds(3);
                Gmat(1, 4) =      halfTs2 * dAx_ds(4);
                Gmat(2, 1) =      halfTs2 * dAz_ds(1);
                Gmat(2, 2) = 1 + halfTs2 * dAz_ds(2);
                Gmat(2, 3) =      halfTs2 * dAz_ds(3);
                Gmat(2, 4) = Ts + halfTs2 * dAz_ds(4);
                Gmat(3, 1) = Ts * dAx_ds(1);
                Gmat(3, 2) = Ts * dAx_ds(2);
                Gmat(3, 3) = 1 + Ts * dAx_ds(3);
                Gmat(3, 4) = Ts * dAx_ds(4);
                Gmat(4, 1) = Ts * dAz_ds(1);
                Gmat(4, 2) = Ts * dAz_ds(2);
                Gmat(4, 3) = Ts * dAz_ds(3);
                Gmat(4, 4) = 1 + Ts * dAz_ds(4);
                M_out = Gmat * M_in * Gmat' + Qs;
                M_out = (M_out + M_out') / 2;
                evM = eig(M_out);
                min_ev = min(real(evM));
                if min_ev < 0
                    jitter_now = 1e-12 - min_ev;
                    M_out = M_out + jitter_now * I4;
                    M_out = (M_out + M_out') / 2;
                    n_cov_fix = n_cov_fix + 1;
                    if jitter_now > max_jitter
                        max_jitter = jitter_now;
                    end
                end
                s_airy(:, k) = s_out;
                M_airy(:, :, k) = M_out;

                s_cv(:, k) = Fcv * s_cv(:, k);
                M_cv(:, :, k) = Fcv * M_cv(:, :, k) * Fcv' + Qs;
                M_cv(:, :, k) = (M_cv(:, :, k) + M_cv(:, :, k)') / 2;
                evM = eig(M_cv(:, :, k));
                min_ev = min(real(evM));
                if min_ev < 0
                    jitter_now = 1e-12 - min_ev;
                    M_cv(:, :, k) = M_cv(:, :, k) + jitter_now * I4;
                    M_cv(:, :, k) = (M_cv(:, :, k) + M_cv(:, :, k)') / 2;
                    n_cov_fix = n_cov_fix + 1;
                    if jitter_now > max_jitter
                        max_jitter = jitter_now;
                    end
                end
            end
        end

        qn = q_plan(:, n);
        for k = 1:K
            p_m = p_plan(k, n);
            dx_a = qn(1) - s_airy(1, k);
            dy_a = qn(2) - y_buoy(k);
            dz_a = z_U - s_airy(2, k);
            d_a = sqrt(dx_a * dx_a + dy_a * dy_a + dz_a * dz_a);
            d3_a = d_a * d_a * d_a;
            H_a = zeros(3, 4);
            H_a(1, 1) =  kappa_a * (dy_a * dy_a + dz_a * dz_a) / d3_a;
            H_a(1, 2) = -kappa_a * dx_a * dz_a / d3_a;
            H_a(2, 1) = -kappa_a * dx_a * dy_a / d3_a;
            H_a(2, 2) = -kappa_a * dy_a * dz_a / d3_a;
            H_a(3, 1) = -2 * dx_a / (c0 * d_a);
            H_a(3, 2) = -2 * dz_a / (c0 * d_a);
            rho_a = chi2 / (4 * d_a * d_a * sigma_r2);

            dx_c = qn(1) - s_cv(1, k);
            dy_c = qn(2) - y_buoy(k);
            dz_c = z_U - s_cv(2, k);
            d_c = sqrt(dx_c * dx_c + dy_c * dy_c + dz_c * dz_c);
            d3_c = d_c * d_c * d_c;
            H_c = zeros(3, 4);
            H_c(1, 1) =  kappa_a * (dy_c * dy_c + dz_c * dz_c) / d3_c;
            H_c(1, 2) = -kappa_a * dx_c * dz_c / d3_c;
            H_c(2, 1) = -kappa_a * dx_c * dy_c / d3_c;
            H_c(2, 2) = -kappa_a * dy_c * dz_c / d3_c;
            H_c(3, 1) = -2 * dx_c / (c0 * d_c);
            H_c(3, 2) = -2 * dz_c / (c0 * d_c);

            rho_m = rho_plan(k, n);
            if p_m <= 1e-12 || ~(rho_m > 0)
                n_skip_update = n_skip_update + 1;
            else
                dx_t = qn(1) - s_true_mc(1, k);
                dy_t = qn(2) - y_buoy(k);
                dz_t = z_U - s_true_mc(2, k);
                d_t = sqrt(dx_t * dx_t + dy_t * dy_t + dz_t * dz_t);
                h_true_m = [-kappa_a * dx_t / d_t; -kappa_a * dy_t / d_t; 2 * d_t / c0];
                R_orig = diag(Dm_diag ./ (p_m * rho_m));
                R_orig = (R_orig + R_orig') / 2;
                zdraw = randn(3, 1);
                y_meas = h_true_m + chol(R_orig, 'lower') * zdraw;

                h_airy = [-kappa_a * dx_a / d_a; -kappa_a * dy_a / d_a; 2 * d_a / c0];
                Hs = Dm_is * H_a;
                Rs = I3 / (p_m * rho_m);
                M_in = M_airy(:, :, k);
                Syy = Hs * M_in * Hs' + Rs;
                Syy = (Syy + Syy') / 2;
                if rcond(Syy) < 1e-12
                    Syy = Syy + 1e-10 * I3;
                    n_Syy_fix = n_Syy_fix + 1;
                end
                Kgain = (Syy \ (M_in * Hs')')';
                s_airy(:, k) = s_airy(:, k) + Kgain * (Dm_is * (y_meas - h_airy));
                Ikh = I4 - Kgain * Hs;
                M_new = Ikh * M_in * Ikh' + Kgain * Rs * Kgain';
                M_new = (M_new + M_new') / 2;
                evM = eig(M_new);
                min_ev = min(real(evM));
                if min_ev < 0
                    jitter_now = 1e-12 - min_ev;
                    M_new = M_new + jitter_now * I4;
                    M_new = (M_new + M_new') / 2;
                    n_cov_fix = n_cov_fix + 1;
                    if jitter_now > max_jitter
                        max_jitter = jitter_now;
                    end
                end
                M_airy(:, :, k) = M_new;

                % CV-EKF 使用同一测量和同一 R，线性化点用其自身预测。
                h_cv = [-kappa_a * dx_c / d_c; -kappa_a * dy_c / d_c; 2 * d_c / c0];
                Hs = Dm_is * H_c;
                M_in = M_cv(:, :, k);
                Syy = Hs * M_in * Hs' + Rs;
                Syy = (Syy + Syy') / 2;
                if rcond(Syy) < 1e-12
                    Syy = Syy + 1e-10 * I3;
                    n_Syy_fix = n_Syy_fix + 1;
                end
                Kgain = (Syy \ (M_in * Hs')')';
                s_cv(:, k) = s_cv(:, k) + Kgain * (Dm_is * (y_meas - h_cv));
                Ikh = I4 - Kgain * Hs;
                M_new = Ikh * M_in * Ikh' + Kgain * Rs * Kgain';
                M_new = (M_new + M_new') / 2;
                evM = eig(M_new);
                min_ev = min(real(evM));
                if min_ev < 0
                    jitter_now = 1e-12 - min_ev;
                    M_new = M_new + jitter_now * I4;
                    M_new = (M_new + M_new') / 2;
                    n_cov_fix = n_cov_fix + 1;
                    if jitter_now > max_jitter
                        max_jitter = jitter_now;
                    end
                end
                M_cv(:, :, k) = M_new;
            end

            if any(~isfinite(s_airy(:, k))) || any(~isfinite(s_cv(:, k))) || ...
                    any(~isfinite(M_airy(:, :, k))) || any(~isfinite(M_cv(:, :, k)))
                n_nonfinite = n_nonfinite + 1;
            end

            ex_a = s_true_mc(1, k) - s_airy(1, k);
            ez_a = s_true_mc(2, k) - s_airy(2, k);
            evx_a = s_true_mc(3, k) - s_airy(3, k);
            evz_a = s_true_mc(4, k) - s_airy(4, k);
            ex_c = s_true_mc(1, k) - s_cv(1, k);
            ez_c = s_true_mc(2, k) - s_cv(2, k);
            evx_c = s_true_mc(3, k) - s_cv(3, k);
            evz_c = s_true_mc(4, k) - s_cv(4, k);
            sum_ex_airy(k, n) = sum_ex_airy(k, n) + ex_a;
            sum_ez_airy(k, n) = sum_ez_airy(k, n) + ez_a;
            ep_a = ex_a * ex_a + ez_a * ez_a;
            ev_a = evx_a * evx_a + evz_a * evz_a;
            ep_c = ex_c * ex_c + ez_c * ez_c;
            ev_c = evx_c * evx_c + evz_c * evz_c;
            err_p_sum_airy(n) = err_p_sum_airy(n) + ep_a;
            err_v_sum_airy(n) = err_v_sum_airy(n) + ev_a;
            err_p_sum_cv(n) = err_p_sum_cv(n) + ep_c;
            err_v_sum_cv(n) = err_v_sum_cv(n) + ev_c;
            err_p_buoy_airy(k) = err_p_buoy_airy(k) + ep_a;
            err_v_buoy_airy(k) = err_v_buoy_airy(k) + ev_a;
            err_p_buoy_cv(k) = err_p_buoy_cv(k) + ep_c;
            err_v_buoy_cv(k) = err_v_buoy_cv(k) + ev_c;
            theta_sum_airy(n) = theta_sum_airy(n) + real(trace(E' * M_airy(:, :, k) * E));
            theta_sum_cv(n) = theta_sum_cv(n) + real(trace(E' * M_cv(:, :, k) * E));
        end
    end
    if mod(imc, 20) == 0
        fprintf('Monte Carlo %d / %d 完成\n', imc, Nmc);
        if exist('OCTAVE_VERSION', 'builtin')
            fflush(stdout);
        end
    end
end
time_mc = toc;
fprintf('Monte Carlo 用时 %.2f s\n', time_mc);

%% 10. 性能指标
rmse_p_airy = sqrt(err_p_sum_airy / (Nmc * K));
rmse_v_airy = sqrt(err_v_sum_airy / (Nmc * K));
rmse_p_cv = sqrt(err_p_sum_cv / (Nmc * K));
rmse_v_cv = sqrt(err_v_sum_cv / (Nmc * K));
theta_bar_airy = theta_sum_airy / (Nmc * K);
theta_bar_cv = theta_sum_cv / (Nmc * K);
sqrt_theta_airy = sqrt(max(theta_bar_airy, 0));
sqrt_theta_cv = sqrt(max(theta_bar_cv, 0));

rmse_pos_buoy_airy = sqrt(err_p_buoy_airy / (Nmc * N));
rmse_vel_buoy_airy = sqrt(err_v_buoy_airy / (Nmc * N));
rmse_pos_buoy_cv = sqrt(err_p_buoy_cv / (Nmc * N));
rmse_vel_buoy_cv = sqrt(err_v_buoy_cv / (Nmc * N));
rmse_pos_all_airy = sqrt(sum(err_p_sum_airy) / (Nmc * K * N));
rmse_vel_all_airy = sqrt(sum(err_v_sum_airy) / (Nmc * K * N));
rmse_pos_all_cv = sqrt(sum(err_p_sum_cv) / (Nmc * K * N));
rmse_vel_all_cv = sqrt(sum(err_v_sum_cv) / (Nmc * K * N));

mean_sqrt_t = zeros(1, 3);
for sch = 1:3
    mean_sqrt_t(sch) = mean(sqrt(t_obj(:, sch)));
end
p_sum_hist = zeros(3, N);
for sch = 1:3
    p_sum_hist(sch, :) = sum(p_hist(:, :, sch), 1);
end
max_power_sum = max(p_sum_hist(:));
step_hist(1) = 0;
max_speed = max(step_hist(2:end)) / Ts;

ex_bar = sum_ex_airy / Nmc;
ez_bar = sum_ez_airy / Nmc;
B2_airy = mean(ex_bar.^2 + ez_bar.^2, 1);
V_airy = max(rmse_p_airy.^2 - B2_airy, 0);
sqrt_V_airy = sqrt(V_airy);
mean_bias = mean(sqrt(ex_bar(:).^2 + ez_bar(:).^2));

Qw_emp = w_outer / max(n_w, 1);
proc_rel = norm(Qw_emp - Qs, 'fro') / max(norm(Qs, 'fro'), 1e-30);
init_ratio = zeros(4, K);
for k = 1:K
    mom_k = init_err_outer(:, :, k) / Nmc;
    mdiag = diag(M0(:, :, k));
    init_ratio(:, k) = diag(mom_k) ./ mdiag;
end
min_rate_B = zeros(1, N);
min_rate_C = zeros(1, N);
min_beam_B = zeros(1, N);
min_beam_C = zeros(1, N);
min_rate_A = zeros(1, N);
for n = 1:N
    min_rate_A(n) = min(rate_hist(:, n, 1));
    min_rate_B(n) = min(rate_hist(:, n, 2));
    min_rate_C(n) = min(rate_hist(:, n, 3));
    min_beam_B(n) = min(beam_hist(:, n, 2));
    min_beam_C(n) = min(beam_hist(:, n, 3));
end

half_idx = floor(N / 2):N;
ratio_late = mean(rmse_p_airy(half_idx).^2) / max(mean(theta_bar_airy(half_idx)), 1e-12);
V_late = mean(V_airy(half_idx));
Theta_late = mean(theta_bar_airy(half_idx));
ratio_debiased = V_late / max(Theta_late, 1e-12);

G_struct_ok = G_checked && abs(G_check(1, 3) - Ts) < 0.05 && ...
    G_check(2, 2) < 0.95 && G_check(3, 3) < 1.05 && G_check(4, 4) < 1.05;
beam_ok = beam_overshoot <= 1e-8;
power_ok = (n_power_violate == 0) && (max_power_sum <= P_U + 1e-6) && ...
    min(p_hist(:)) >= -1e-8;
speed_ok = (n_speed_violate == 0) && (max_speed <= Vmax + 1e-6);
cov_ok = max_jitter < 1e-6;
finite_ok = (n_nonfinite == 0) && all(isfinite(rmse_p_airy)) && ...
    all(isfinite(rmse_p_cv)) && all(isfinite(theta_bar_airy)) && ...
    all(isfinite(t_obj(:)));
q0_ok = norm(qU_hist(:, 1, 3) - q_U0) <= 1e-8;
step1_ok = abs(step_hist(1)) <= 1e-10;
pred_comm_ok = init_dist_ok && (n_pred_violate_C == 0) && (n_infeasible == 0);
rate_ok = (n_rate_violate(3) == 0);
init_cov_ok = all(init_ratio(:) >= 0.35) && all(init_ratio(:) <= 2.2);
proc_ok = proc_rel < 0.15;
bias_ok = all(isfinite(ex_bar(:))) && all(isfinite(ez_bar(:))) && ...
    all(isfinite(V_airy)) && all(isfinite(B2_airy));
ratio_stat_ok = isfinite(ratio_debiased) && ratio_debiased >= 0;
mono_ok = (n_nonmono == 0);
array_ok = a_dim_ok && a_norm_ok && da_ok && kappa_ok;

fprintf('\n========== 仿真结果摘要 ==========\n');
fprintf('三个浮标 Airy-Morison-EKF 全程位置 RMSE (m)：');
fprintf(' %.4f', rmse_pos_buoy_airy);
fprintf('\n');
fprintf('三个浮标 CV-EKF 全程位置 RMSE (m)：');
fprintf(' %.4f', rmse_pos_buoy_cv);
fprintf('\n');
fprintf('Airy-Morison-EKF 全程位置/速度 RMSE = %.4f / %.4f m 或 m/s\n', ...
    rmse_pos_all_airy, rmse_vel_all_airy);
fprintf('CV-EKF           全程位置/速度 RMSE = %.4f / %.4f m 或 m/s\n', ...
    rmse_pos_all_cv, rmse_vel_all_cv);
fprintf('后半程 mean(RMSE_p^2) / mean(Theta_bar) = %.4f\n', ratio_late);
fprintf('平均位置偏差 = %.6f m\n', mean_bias);
fprintf('后半程经验误差方差 V = %.6e m^2\n', V_late);
fprintf('后半程理论协方差 Theta = %.6e m^2\n', Theta_late);
fprintf('去偏后经验方差 / 理论协方差 = %.4f\n', ratio_debiased);
fprintf('说明: 原始 RMSE^2/Theta = %.4f，主要来自平均位置偏差 %.4f m；去偏后比值 %.4f。剩余差别对应一步 Taylor 预测与五子步 Heun 的离散化差异，未改 sigma_ax、sigma_az 或测量噪声。\n', ...
    ratio_late, mean_bias, ratio_debiased);
fprintf('Monte Carlo 平均协方差指标 mean(Theta_bar) = %.6e m^2\n', mean(theta_bar_airy));
fprintf('min(min_rate_A) = %.6f\n', min(min_rate_A));
fprintf('min(min_rate_B) = %.6f\n', min(min_rate_B));
fprintf('min(min_rate_C) = %.6f\n', min(min_rate_C));
fprintf('最小上行频谱效率：方案A %.4f，方案B %.4f，方案C %.4f bit/(s*Hz)\n', ...
    min(min_rate_A), min(min_rate_B), min(min_rate_C));
fprintf('最小波束匹配增益：方案B %.4f，方案C %.4f\n', min(min_beam_B), min(min_beam_C));
fprintf('无人机最大实际速度 = %.4f m/s（限制 %.4f m/s）\n', max_speed, Vmax);
fprintf('感知功率最大使用量 = %.6f W（预算 %.4f W）\n', max_power_sum, P_U);
fprintf('时间平均最坏位置不确定度 sqrt(t)：方案A %.4f，方案B %.4f，方案C %.4f m\n', ...
    mean_sqrt_t(1), mean_sqrt_t(2), mean_sqrt_t(3));
fprintf('联合优化相对固定位置+平均功率，sqrt(t) 平均下降 %.2f%%\n', ...
    100 * (mean_sqrt_t(1) - mean_sqrt_t(3)) / mean_sqrt_t(1));
fprintf('联合优化相对固定位置+优化功率，sqrt(t) 平均下降 %.2f%%\n', ...
    100 * (mean_sqrt_t(2) - mean_sqrt_t(3)) / mean_sqrt_t(2));
fprintf('真实距离超过 d_s 的浮标-时隙次数：方案A %d，方案B %d，方案C %d\n', ...
    n_true_violate(1), n_true_violate(2), n_true_violate(3));
fprintf('实际频谱效率低于 R_s 的次数：方案A %d，方案B %d，方案C %d\n', ...
    n_rate_violate(1), n_rate_violate(2), n_rate_violate(3));
fprintf('真实距离超过 d_s*|omega| 的次数：方案A %d，方案B %d，方案C %d\n', ...
    n_equiv_violate(1), n_equiv_violate(2), n_equiv_violate(3));
fprintf('方案C 预测距离超过安全阈值的次数 %d，无可行候选次数 %d\n', ...
    n_pred_violate_C, n_infeasible);
fprintf('初始误差二阶矩 / diag(M0) 的范围 [%.3f, %.3f]\n', min(init_ratio(:)), max(init_ratio(:)));
fprintf('过程噪声样本协方差与 Qs 的相对差 %.4f（样本数 %d）\n', proc_rel, n_w);
if min(min_rate_C) < R_s - 1e-6
    fprintf('说明: 方案 C 的最小频谱效率 %.6f 仍低于 R_s = %.4f，未改绘图数据。\n', ...
        min(min_rate_C), R_s);
end
fprintf('协方差负特征值修正次数 %d，最大修正量 %.3e\n', n_cov_fix, max_jitter);
fprintf('功率为零而跳过的更新次数 %d（含三方案名义闭环与 Monte Carlo）\n', n_skip_update);
fprintf('信息矩阵非单调次数 %d，新息协方差修正次数 %d\n', n_nonmono, n_Syy_fix);
fprintf('状态转移雅可比抽查 G(1,3)=%.4f（Ts=%.4f），G(2,2)=%.4f，G(3,3)=%.4f\n', ...
    G_check(1, 3), Ts, G_check(2, 2), G_check(3, 3));

fprintf('\n========== 静态检查 ==========\n');
check_name = { ...
    '深水条件'; ...
    '阵元间距为半波长'; ...
    'kappa_a 等于 pi'; ...
    '阵列响应为 16 维且范数为 1'; ...
    '测量雅可比与中心差分一致'; ...
    '通信信噪比没有第二次距离衰减'; ...
    'd_s 处理想频谱效率等于 R_s'; ...
    '真实轨迹有限且浮标保持在水下'; ...
    '波束匹配系数模平方不超过 1'; ...
    '无人机步长不超过 Vmax*Ts'; ...
    '感知功率非负且不超过总预算'; ...
    '方案C预测位置满足通信距离'; ...
    '协方差半正定修正量可忽略'; ...
    'Monte Carlo 与优化结果无 NaN/Inf'; ...
    '功率目标关于 p 单调'; ...
    '状态转移雅可比结构合理'; ...
    '第 1 帧位置等于 q_U0'; ...
    '第 1 帧移动距离为零'; ...
    '初始预测距离满足安全余量'; ...
    '第 2 帧起速度不超过 Vmax'; ...
    '优化位置满足通信安全余量'; ...
    '方案C实际频谱效率不低于 R_s'; ...
    'Monte Carlo 初始误差符合 M0'; ...
    '过程噪声协方差与 Qs 一致'; ...
    '偏差统计无 NaN 或 Inf'; ...
    '去偏方差与后验协方差比值有限'};
check_flag = [ ...
    kd_deep > 3; ...
    da_ok; ...
    kappa_ok; ...
    array_ok; ...
    H_fd_ok; ...
    pathloss_once_ok && pathloss_not_double && snr_gap_max < 1e-8; ...
    ds_rate_ok; ...
    truth_ok; ...
    beam_ok; ...
    speed_ok; ...
    power_ok; ...
    pred_comm_ok; ...
    cov_ok; ...
    finite_ok; ...
    mono_ok; ...
    G_struct_ok; ...
    q0_ok; ...
    step1_ok; ...
    init_dist_ok; ...
    speed_ok; ...
    pred_comm_ok; ...
    rate_ok; ...
    init_cov_ok; ...
    proc_ok; ...
    bias_ok; ...
    ratio_stat_ok];
n_pass_check = 0;
for ci = 1:numel(check_flag)
    if check_flag(ci)
        fprintf('  [通过] %s\n', check_name{ci});
        n_pass_check = n_pass_check + 1;
    else
        fprintf('  [失败] %s\n', check_name{ci});
    end
end
fprintf('检查合计 %d / %d 通过。\n', n_pass_check, numel(check_flag));
fprintf('RMSE 的平方与 Theta_bar 同为 m^2；图中比较的是 RMSE 与 sqrt(Theta_bar)，单位均为 m。\n');
fprintf('优化和 EKF 测量噪声使用论文中的 rho（理想波束 |omega|^2=1）。\n');
fprintf('通信性能评估使用真实阵列响应内积，不把 |omega|^2 置为 1。\n');

%% 11. 七张 IEEE 风格图
colors = [0.00, 0.45, 0.74; ...
          0.85, 0.33, 0.10; ...
          0.47, 0.67, 0.19];
line_styles = {'-', '--', '-.'};
scheme_colors = [0.35, 0.35, 0.35; ...
                 0.85, 0.33, 0.10; ...
                 0.00, 0.45, 0.74];

% 图1 海浪激励及浮标运动响应
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 7.6, 3.7], 'Visible', 'off');
subplot(1, 2, 1);
hold on;
for k = 1:K
    plot(t_grid, eta_true(k, :), 'LineStyle', line_styles{k}, 'Color', colors(k, :), 'LineWidth', 1.4);
end
grid on; box on;
xlim([t_grid(1), t_grid(end)]);
xlabel('时间 (s)', 'Interpreter', 'none', 'FontName', fig_font);
ylabel('海浪表面位移 (m)', 'Interpreter', 'none', 'FontName', fig_font);
title('(a) 浮标位置处的海浪表面位移', 'Interpreter', 'none', 'FontName', fig_font);
legend({'浮标 1', '浮标 2', '浮标 3'}, 'Location', 'northeast', 'Interpreter', 'none', 'FontName', fig_font);
set(gca, 'FontName', fig_font, 'FontSize', 11);
hold off;

subplot(1, 2, 2);
hold on;
for k = 1:K
    plot(t_grid, z_true(k, :), 'LineStyle', line_styles{k}, 'Color', colors(k, :), 'LineWidth', 1.4);
end
grid on; box on;
xlim([t_grid(1), t_grid(end)]);
xlabel('时间 (s)', 'Interpreter', 'none', 'FontName', fig_font);
ylabel('浮标竖直位置 z (m)', 'Interpreter', 'none', 'FontName', fig_font);
title('(b) 三个浮标的竖直运动', 'Interpreter', 'none', 'FontName', fig_font);
legend({'浮标 1', '浮标 2', '浮标 3'}, 'Location', 'northeast', 'Interpreter', 'none', 'FontName', fig_font);
set(gca, 'FontName', fig_font, 'FontSize', 11);
hold off;
set(fig, 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
print(fig, fullfile(out_dir, 'fig1_wave_buoy.png'), '-dpng', '-r300');
close(fig);

% 图2 代表性浮标真实位置与 EKF 估计
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 7.6, 3.7], 'Visible', 'off');
subplot(1, 2, 1);
hold on;
plot(t_grid, x_true(k_rep, :), '-', 'Color', [0.15, 0.15, 0.15], 'LineWidth', 1.5);
plot(t_grid, squeeze(post_hist(1, k_rep, :, 3)), '--', 'Color', colors(1, :), 'LineWidth', 1.4);
grid on; box on;
xlim([t_grid(1), t_grid(end)]);
xlabel('时间 (s)', 'Interpreter', 'none', 'FontName', fig_font);
ylabel('水平位置 x (m)', 'Interpreter', 'none', 'FontName', fig_font);
title('(a) 代表性浮标水平位置', 'Interpreter', 'none', 'FontName', fig_font);
legend({'真实值', 'Airy-Morison-EKF'}, 'Location', 'northeast', 'Interpreter', 'none', 'FontName', fig_font);
set(gca, 'FontName', fig_font, 'FontSize', 11);
hold off;

subplot(1, 2, 2);
hold on;
plot(t_grid, z_true(k_rep, :), '-', 'Color', [0.15, 0.15, 0.15], 'LineWidth', 1.5);
plot(t_grid, squeeze(post_hist(2, k_rep, :, 3)), '--', 'Color', colors(2, :), 'LineWidth', 1.4);
grid on; box on;
xlim([t_grid(1), t_grid(end)]);
xlabel('时间 (s)', 'Interpreter', 'none', 'FontName', fig_font);
ylabel('竖直位置 z (m)', 'Interpreter', 'none', 'FontName', fig_font);
title('(b) 代表性浮标竖直位置', 'Interpreter', 'none', 'FontName', fig_font);
legend({'真实值', 'Airy-Morison-EKF'}, 'Location', 'northeast', 'Interpreter', 'none', 'FontName', fig_font);
set(gca, 'FontName', fig_font, 'FontSize', 11);
hold off;
set(fig, 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
print(fig, fullfile(out_dir, 'fig2_EKF_tracking.png'), '-dpng', '-r300');
close(fig);

% 图3 RMSE 对比
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 7.6, 3.7], 'Visible', 'off');
subplot(1, 2, 1);
hold on;
plot(t_grid, rmse_p_airy, '-', 'Color', colors(1, :), 'LineWidth', 1.5);
plot(t_grid, rmse_p_cv, '--', 'Color', colors(2, :), 'LineWidth', 1.5);
grid on; box on;
xlim([t_grid(1), t_grid(end)]);
xlabel('时间 (s)', 'Interpreter', 'none', 'FontName', fig_font);
ylabel('位置 RMSE (m)', 'Interpreter', 'none', 'FontName', fig_font);
title('(a) 位置 RMSE', 'Interpreter', 'none', 'FontName', fig_font);
legend({'Airy-Morison-EKF', 'CV-EKF'}, 'Location', 'northeast', 'Interpreter', 'none', 'FontName', fig_font);
set(gca, 'FontName', fig_font, 'FontSize', 11);
hold off;

subplot(1, 2, 2);
hold on;
plot(t_grid, rmse_v_airy, '-', 'Color', colors(1, :), 'LineWidth', 1.5);
plot(t_grid, rmse_v_cv, '--', 'Color', colors(2, :), 'LineWidth', 1.5);
grid on; box on;
xlim([t_grid(1), t_grid(end)]);
xlabel('时间 (s)', 'Interpreter', 'none', 'FontName', fig_font);
ylabel('速度 RMSE (m/s)', 'Interpreter', 'none', 'FontName', fig_font);
title('(b) 速度 RMSE', 'Interpreter', 'none', 'FontName', fig_font);
legend({'Airy-Morison-EKF', 'CV-EKF'}, 'Location', 'northeast', 'Interpreter', 'none', 'FontName', fig_font);
set(gca, 'FontName', fig_font, 'FontSize', 11);
hold off;
set(fig, 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
print(fig, fullfile(out_dir, 'fig3_RMSE_comparison.png'), '-dpng', '-r300');
close(fig);

% 图4 无人机水平轨迹
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 8.4, 5.2], 'Visible', 'off');
hold on;
plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], ...
    '--', 'Color', [0.65, 0.65, 0.65], 'LineWidth', 1.0);
for k = 1:K
    plot(x_true(k, :), y_buoy(k) * ones(1, N), 'LineStyle', line_styles{k}, ...
        'Color', colors(k, :), 'LineWidth', 1.5);
    plot(x_true(k, 1), y_buoy(k), 'o', 'Color', colors(k, :), ...
        'MarkerFaceColor', colors(k, :), 'MarkerSize', 6, 'HandleVisibility', 'off');
    plot(x_true(k, n_demo), y_buoy(k), 's', 'Color', colors(k, :), ...
        'MarkerFaceColor', 'w', 'MarkerSize', 7, 'LineWidth', 1.1, 'HandleVisibility', 'off');
end
cmap_u = zeros(N, 3);
for n = 1:N
    alpha_u = (n - 1) / max(N - 1, 1);
    cmap_u(n, :) = (1 - alpha_u) * [0.10, 0.30, 0.70] + alpha_u * [0.75, 0.10, 0.10];
end
for n = 1:N - 1
    plot(qU_hist(1, n:n + 1, 3), qU_hist(2, n:n + 1, 3), '-', ...
        'Color', cmap_u(n, :), 'LineWidth', 1.8, 'HandleVisibility', 'off');
end
plot(nan, nan, '-', 'Color', [0.15, 0.25, 0.55], 'LineWidth', 1.8);
plot(q_U0(1), q_U0(2), 'p', 'MarkerSize', 12, 'MarkerFaceColor', [0.95, 0.75, 0.10], ...
    'MarkerEdgeColor', 'k', 'LineWidth', 0.6);
if ~isempty(grid_demo)
    plot(grid_demo(1, :), grid_demo(2, :), 'x', 'Color', [0.35, 0.35, 0.35], ...
        'MarkerSize', 7, 'LineWidth', 1.1);
end
plot(qU_hist(1, end, 3), qU_hist(2, end, 3), '*', 'Color', [0.70, 0.05, 0.05], ...
    'MarkerSize', 11, 'LineWidth', 1.3);
grid on; box on;
axis equal;
xlim([x_min - 1, x_max + 1]);
ylim([y_min - 1, y_max + 1]);
xlabel('x (m)', 'Interpreter', 'none', 'FontName', fig_font);
ylabel('y (m)', 'Interpreter', 'none', 'FontName', fig_font);
title('无人机水平位置优化轨迹', 'Interpreter', 'none', 'FontName', fig_font);
if ~isempty(grid_demo)
    legend({'飞行区域', '浮标 1', '浮标 2', '浮标 3', ...
        '联合优化轨迹（蓝到红表示时间）', '初始位置', ...
        sprintf('t = %.2f s 可行候选网格', t_grid(n_demo)), '最终位置'}, ...
        'Location', 'northwest', 'Interpreter', 'none', 'FontName', fig_font, 'FontSize', 8);
else
    legend({'飞行区域', '浮标 1', '浮标 2', '浮标 3', ...
        '联合优化轨迹（蓝到红表示时间）', '初始位置', '最终位置'}, ...
        'Location', 'northwest', 'Interpreter', 'none', 'FontName', fig_font, 'FontSize', 8);
end
set(gca, 'FontName', fig_font, 'FontSize', 11, 'Position', [0.07, 0.12, 0.48, 0.78]);
hold off;

% 飞行区域远大于实际机动范围，右侧放大图用于看清轨迹随时间的变化。
axes('Position', [0.64, 0.16, 0.32, 0.68]);
hold on;
for n = 1:N - 1
    plot(qU_hist(1, n:n + 1, 3), qU_hist(2, n:n + 1, 3), '-', ...
        'Color', cmap_u(n, :), 'LineWidth', 1.6);
end
if ~isempty(grid_demo)
    plot(grid_demo(1, :), grid_demo(2, :), 'x', 'Color', [0.35, 0.35, 0.35], ...
        'MarkerSize', 6, 'LineWidth', 1.0);
end
plot(q_U0(1), q_U0(2), 'p', 'MarkerSize', 9, 'MarkerFaceColor', [0.95, 0.75, 0.10], ...
    'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
plot(qU_hist(1, end, 3), qU_hist(2, end, 3), '*', 'Color', [0.70, 0.05, 0.05], ...
    'MarkerSize', 9, 'LineWidth', 1.1);
plot(x_true(2, :), y_buoy(2) * ones(1, N), '--', 'Color', colors(2, :), 'LineWidth', 1.0);
grid on; box on;
x_lo = min(qU_hist(1, :, 3)) - 1;
x_hi = max(qU_hist(1, :, 3)) + 1;
y_lo = min(qU_hist(2, :, 3)) - 1;
y_hi = max(qU_hist(2, :, 3)) + 1;
if x_hi - x_lo < 4
    x_mid = 0.5 * (x_lo + x_hi);
    x_lo = x_mid - 2;
    x_hi = x_mid + 2;
end
if y_hi - y_lo < 4
    y_mid = 0.5 * (y_lo + y_hi);
    y_lo = y_mid - 2;
    y_hi = y_mid + 2;
end
xlim([x_lo, x_hi]);
ylim([y_lo, y_hi]);
xlabel('x (m)', 'Interpreter', 'none', 'FontName', fig_font, 'FontSize', 8);
ylabel('y (m)', 'Interpreter', 'none', 'FontName', fig_font, 'FontSize', 8);
title('轨迹局部放大', 'Interpreter', 'none', 'FontName', fig_font, 'FontSize', 9);
set(gca, 'FontName', fig_font, 'FontSize', 8);
hold off;
set(fig, 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
print(fig, fullfile(out_dir, 'fig4_UAV_trajectory.png'), '-dpng', '-r300');
close(fig);

% 图5 感知功率分配（方案 C）
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 6.6, 4.0], 'Visible', 'off');
hold on;
for k = 1:K
    plot(t_grid, p_hist(k, :, 3), 'LineStyle', line_styles{k}, 'Color', colors(k, :), 'LineWidth', 1.5);
end
plot([t_grid(1), t_grid(end)], [P_U, P_U], ':', 'Color', [0.15, 0.15, 0.15], 'LineWidth', 1.6);
mk_p = 1:4:N;
plot(t_grid(mk_p), p_sum_hist(3, mk_p), 'o', 'Color', [0.15, 0.15, 0.15], ...
    'MarkerFaceColor', [0.15, 0.15, 0.15], 'MarkerSize', 3.5, 'LineStyle', 'none');
grid on; box on;
xlim([t_grid(1), t_grid(end)]);
ylim([0, P_U * 1.12]);
xlabel('时间 (s)', 'Interpreter', 'none', 'FontName', fig_font);
ylabel('感知功率 (W)', 'Interpreter', 'none', 'FontName', fig_font);
title('联合优化的浮标感知功率分配', 'Interpreter', 'none', 'FontName', fig_font);
legend({'浮标 1', '浮标 2', '浮标 3', '总功率预算 P_U', '功率总和'}, ...
    'Location', 'northeast', 'Interpreter', 'none', 'FontName', fig_font);
set(gca, 'FontName', fig_font, 'FontSize', 11);
hold off;
set(fig, 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
print(fig, fullfile(out_dir, 'fig5_power_allocation.png'), '-dpng', '-r300');
close(fig);

% 图6 上行通信性能：固定位置（方案 B）与联合优化（方案 C）
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 7.6, 3.7], 'Visible', 'off');
subplot(1, 2, 1);
hold on;
plot(t_grid, min_rate_B, '--', 'Color', scheme_colors(2, :), 'LineWidth', 1.5);
plot(t_grid, min_rate_C, '-', 'Color', scheme_colors(3, :), 'LineWidth', 1.5);
plot([t_grid(1), t_grid(end)], [R_s, R_s], ':', 'Color', [0.15, 0.15, 0.15], 'LineWidth', 1.3);
grid on; box on;
xlim([t_grid(1), t_grid(end)]);
rate_lo = min([min(min_rate_B), min(min_rate_C), R_s]);
rate_hi = max([max(min_rate_B), max(min_rate_C), R_s]);
rate_pad = 0.08 * max(rate_hi - rate_lo, 0.05);
ylim([rate_lo - rate_pad, rate_hi + rate_pad]);
xlabel('时间 (s)', 'Interpreter', 'none', 'FontName', fig_font);
ylabel('最小频谱效率 (bit/(s*Hz))', 'Interpreter', 'none', 'FontName', fig_font);
title('(a) 最小上行通信频谱效率', 'Interpreter', 'none', 'FontName', fig_font);
legend({'固定位置（方案 B）', '联合优化（方案 C）', '最低频谱效率 R_s'}, ...
    'Location', 'northeast', 'Interpreter', 'none', 'FontName', fig_font);
set(gca, 'FontName', fig_font, 'FontSize', 11);
hold off;

subplot(1, 2, 2);
hold on;
plot(t_grid, min_beam_B, '--', 'Color', scheme_colors(2, :), 'LineWidth', 1.5);
plot(t_grid, min_beam_C, '-', 'Color', scheme_colors(3, :), 'LineWidth', 1.5);
grid on; box on;
xlim([t_grid(1), t_grid(end)]);
beam_lo = min([min(min_beam_B), min(min_beam_C)]);
ylim([beam_lo - 0.004, 1.001]);
xlabel('时间 (s)', 'Interpreter', 'none', 'FontName', fig_font);
ylabel('最小波束匹配增益', 'Interpreter', 'none', 'FontName', fig_font);
title('(b) 最小波束匹配增益', 'Interpreter', 'none', 'FontName', fig_font);
legend({'固定位置（方案 B）', '联合优化（方案 C）'}, ...
    'Location', 'southeast', 'Interpreter', 'none', 'FontName', fig_font);
set(gca, 'FontName', fig_font, 'FontSize', 11);
hold off;
set(fig, 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
print(fig, fullfile(out_dir, 'fig6_communication_performance.png'), '-dpng', '-r300');
close(fig);

% 图7 联合优化效果及协方差一致性
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 7.6, 3.7], 'Visible', 'off');
subplot(1, 2, 1);
hold on;
plot(t_grid, sqrt(t_obj(:, 1)), 'LineStyle', '-', 'Color', scheme_colors(1, :), 'LineWidth', 1.5);
plot(t_grid, sqrt(t_obj(:, 2)), 'LineStyle', '--', 'Color', scheme_colors(2, :), 'LineWidth', 1.5);
plot(t_grid, sqrt(t_obj(:, 3)), 'LineStyle', '-', 'Color', scheme_colors(3, :), 'LineWidth', 1.5);
grid on; box on;
xlim([t_grid(1), t_grid(end)]);
xlabel('时间 (s)', 'Interpreter', 'none', 'FontName', fig_font);
ylabel('最坏位置不确定度 (m)', 'Interpreter', 'none', 'FontName', fig_font);
title('(a) 三方案最坏位置不确定度', 'Interpreter', 'none', 'FontName', fig_font);
legend({'方案 A：固定位置+平均功率', '方案 B：固定位置+优化功率', '方案 C：位置与功率联合优化'}, ...
    'Location', 'northeast', 'Interpreter', 'none', 'FontName', fig_font, 'FontSize', 8);
set(gca, 'FontName', fig_font, 'FontSize', 11);
hold off;

subplot(1, 2, 2);
hold on;
plot(t_grid, rmse_p_airy, '-', 'Color', colors(1, :), 'LineWidth', 1.5);
plot(t_grid, sqrt_V_airy, '-.', 'Color', colors(3, :), 'LineWidth', 1.5);
plot(t_grid, sqrt_theta_airy, '--', 'Color', colors(2, :), 'LineWidth', 1.5);
grid on; box on;
xlim([t_grid(1), t_grid(end)]);
xlabel('时间 (s)', 'Interpreter', 'none', 'FontName', fig_font);
ylabel('位置误差 (m)', 'Interpreter', 'none', 'FontName', fig_font);
title('(b) RMSE、去偏标准差与协方差指标', 'Interpreter', 'none', 'FontName', fig_font);
legend({'位置 RMSE', '去偏后经验标准差', '协方差指标平方根'}, 'Location', 'northeast', ...
    'Interpreter', 'none', 'FontName', fig_font, 'FontSize', 8);
set(gca, 'FontName', fig_font, 'FontSize', 11);
hold off;
set(fig, 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
print(fig, fullfile(out_dir, 'fig7_joint_optimization.png'), '-dpng', '-r300');
close(fig);

%% 12. 保存数据
summary = struct();
summary.ds = ds;
summary.kw = kw;
summary.lambda_w = lambda_w;
summary.kd_deep = kd_deep;
summary.lambda_c = lambda_c;
summary.da = da;
summary.kappa_a = kappa_a;
summary.Me = Me;
summary.Kh = Kh;
summary.Ch = Ch;
summary.CDp = CDp;
summary.rmse_pos_buoy_airy = rmse_pos_buoy_airy;
summary.rmse_pos_buoy_cv = rmse_pos_buoy_cv;
summary.rmse_pos_all_airy = rmse_pos_all_airy;
summary.rmse_vel_all_airy = rmse_vel_all_airy;
summary.rmse_pos_all_cv = rmse_pos_all_cv;
summary.rmse_vel_all_cv = rmse_vel_all_cv;
summary.mean_theta_bar_airy = mean(theta_bar_airy);
summary.min_rate = [min(min_rate_A), min(min_rate_B), min(min_rate_C)];
summary.max_speed = max_speed;
summary.max_power_sum = max_power_sum;
summary.mean_sqrt_t = mean_sqrt_t;
summary.H22_analytic = H_analytic(2, 2);
summary.H22_fd = H_fd(2, 2);
summary.H22_paper = H22_paper;
summary.n_pass_check = n_pass_check;
summary.n_check = numel(check_flag);
summary.ratio_late = ratio_late;
summary.ratio_debiased = ratio_debiased;
summary.V_late = V_late;
summary.Theta_late = Theta_late;
summary.mean_bias = mean_bias;
summary.n_true_violate = n_true_violate;
summary.n_rate_violate = n_rate_violate;
summary.n_equiv_violate = n_equiv_violate;
summary.n_infeasible = n_infeasible;
summary.n_pred_violate_C = n_pred_violate_C;
summary.proc_rel = proc_rel;
summary.ds_safe = ds_safe;
summary.z_U = z_U;
summary.q_U0 = q_U0;
summary.beam_overshoot = beam_overshoot;
summary.time_nominal = time_nominal;
summary.time_mc = time_mc;

save(fullfile(out_dir, 'simulation_results.mat'), ...
    't_grid', 'x_true', 'z_true', 'vx_true', 'vz_true', 'eta_true', 'y_buoy', ...
    'qU_hist', 'p_hist', 't_obj', 'rate_hist', 'beam_hist', ...
    'pred_hist', 'post_hist', 'step_hist', 'grid_demo', 'n_demo', ...
    'rmse_p_airy', 'rmse_v_airy', 'rmse_p_cv', 'rmse_v_cv', ...
    'theta_bar_airy', 'theta_bar_cv', 'sqrt_theta_airy', 'sqrt_V_airy', ...
    'B2_airy', 'rho_hist', 'ds_safe', ...
    'p_sum_hist', 'min_rate_A', 'min_rate_B', 'min_rate_C', ...
    'min_beam_B', 'min_beam_C', 'summary', 'check_flag', 'check_name', ...
    'ds', 'kw', 'lambda_w', 'lambda_c', 'da', 'kappa_a', 'P_U', 'R_s', ...
    'H_analytic', 'H_fd', 'H22_paper', '-v7');

fprintf('\n数据和图片已写入：\n%s\n', out_dir);
fprintf('simulation_results.mat\n');
fprintf('fig1_wave_buoy.png\n');
fprintf('fig2_EKF_tracking.png\n');
fprintf('fig3_RMSE_comparison.png\n');
fprintf('fig4_UAV_trajectory.png\n');
fprintf('fig5_power_allocation.png\n');
fprintf('fig6_communication_performance.png\n');
fprintf('fig7_joint_optimization.png\n');
fprintf('全部计算结束。\n');
