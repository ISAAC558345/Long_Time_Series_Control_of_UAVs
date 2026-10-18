# 基于深水 Airy–Morison 动力学的海上无人机通感预测波束成形

> 本稿从已完成的浮标运动模型之后开始，补齐雷达测量、上行通信、扩展卡尔曼滤波、后验克拉美罗界、优化问题和性能评估。海洋动力学沿用前文；通信侧的帧结构、回波测量、PCRB 和资源优化分别对应 Zhou 等（2025）的时域辅助波束、Li 等（2026）的海上通感预测波束、Li 等（2025）的 PCRB 波束设计，以及 Tang 等（2024）的平台抖动测量误差。仿真曲线需要按第 8 节的方案实际计算，本文不预填数值结果。

---

## 0 与前文的衔接，以及需要回改的三处

前文已经给出深水 Airy 波作用下的二维相对运动 Morison 模型，并把浮标在短时窗口内的状态取为

$$
\boldsymbol{s}(t)=\big[x(t),\,z(t),\,v_x(t),\,v_z(t)\big]^{\mathsf T},
$$

其中 $z$ 是浸没段中心的竖向坐标，$y$ 在该窗口内保持常数，$y$ 向海流取零。后文全部沿用这一状态。把浸没段半长记为 $L_w/2$，静水平衡中心记为 $z_0=-L_w/2$，则雷达和通信使用的散射点坐标是

$$
\boldsymbol{p}_k(t)=\big[x_k(t),\; y_k,\; \zeta_k(t)\big]^{\mathsf T},\qquad
\zeta_k(t)=z_k(t)-z_0.
$$

这样静止海面上 $\zeta_k=0$，波浪起伏直接进入斜距和俯仰。无人机坐标记为 $\boldsymbol{c}_{\mathrm U,n}=[x_{\mathrm U,n},\,y_{\mathrm U,n},\,H]^{\mathsf T}$，$H$ 在一个任务内固定。

正式往下写之前，建议把前文信号模型里的三处改掉，否则后面的滤波器和优化问题无法闭合。

1. 发射波束 $\boldsymbol{f}_{k,n}$ 是 $\mathbb C^{\bar R\times 1}$ 的预编码向量，不是矩阵。浮标个数统一记为 $K$，帧序号记为 $n=0,1,\ldots,N-1$，阵列为 $R\times R$，阵元总数 $\bar R=R^2$。
2. 前文把回波里的多普勒相位删掉了。短脉冲内部的多普勒相位旋转确实可以放进匹配滤波增益里处理，但帧间多普勒频移是径向速度的测量，Li 等（2026）和 Zhou 等（2025）都把它当作 EKF 的观测。海上浮标的轨道速度只有每秒一米量级，载波若取 18 GHz，对应的双程多普勒仍有几十赫兹，足够用来修正速度。后文恢复这一项。
3. 前文的加速度公式里，水平方向没有写外部扰动力，垂直方向却保留了 $F_{e,z}$。风、二阶漂移和局部湍流在两个方向上都存在，而且它们不是已知输入。后文把它们全部放进过程噪声，确定性加速度只保留惯性力、阻力、静水恢复力和垂向阻尼。

回波表达式改为

$$
\begin{aligned}
\boldsymbol{y}_{k,n}(t)
&=
\sqrt{p_{k,n}}\,e^{j2\pi f^{\mathrm D}_{k,n}t}
\Big(
\beta_{k,n}\,\boldsymbol{a}(\boldsymbol{p}_{k,n},\boldsymbol{c}_{\mathrm U,n})
\boldsymbol{a}^{\mathsf H}(\boldsymbol{p}_{k,n},\boldsymbol{c}_{\mathrm U,n}) \\
&\quad
+\rho_{k,n}\beta^{\mathrm{clu}}_{k,n}\,
\boldsymbol{a}(\hat{\boldsymbol{p}}_{k,n|n-1},\boldsymbol{c}_{\mathrm U,n})
\boldsymbol{a}^{\mathsf H}(\hat{\boldsymbol{p}}_{k,n|n-1},\boldsymbol{c}_{\mathrm U,n})
\Big)
\boldsymbol{f}_{k,n}\bar x_{k,n}(t-\tau_{k,n})
+\boldsymbol{z}_{k,n}(t).
\end{aligned}
\tag{1}
$$

其中 $\boldsymbol{z}_{k,n}(t)\sim\mathcal{CN}(\boldsymbol 0,N_0\boldsymbol{I}_{\bar R})$。$\rho_{k,n}\in(0,1)$ 是多普勒域杂波抑制之后仍漏进目标多普勒单元的残余海杂波系数，取法与 Li 等（2026）相同：定向波束把照射斑限制在预测方向附近，所以杂波的角度和时延用预测值，杂波多普勒用 $\rho_{k,n}e^{j2\pi f^{\mathrm D}_{k,n}t}$ 表示。大规模阵列下不同浮标的导向矢量近似正交，再考虑到浮标在海面上的角间隔通常大于陆地车辆，后文对每个浮标单独写回波，不再保留多目标回波求和。

预测导向的发射波束保持前文已经写出的形式

$$
\boldsymbol{f}_{k,n}
=
\boldsymbol{a}\!\left(\hat{\boldsymbol{p}}_{k,n|n-1},\,\boldsymbol{c}_{\mathrm U,n}\right).
\tag{2}
$$

这就是 Zhou 等所说的时域辅助：第 $n$ 帧的波束不依赖当前帧的信道估计，而依赖上一帧滤波结果向前积分得到的坐标。

---

## 1 任务帧结构

任务时长 $T$ 分成 $N$ 个无线帧，帧长 $\Delta_t=T/N$。相邻两帧组成一个操作块。奇数帧做通感照射和状态更新，偶数帧做浮标上行数据回传。无人机在一个操作块内保持悬停，只在操作块之间改变水平位置。这个安排来自 Li 等（2026）：浮标在两帧内的位移远小于无人机为了改善几何而需要飞行的距离，频繁机动没有必要。

帧 0 使用宽波束或正交波形做一次初始探测，对应 Zhou 等的全向阶段，用来得到每个浮标的粗略方向余弦和时延，并把位置反解出来。速度不必置零。前文的深度平均水质点速度直接给出与波浪相位一致的初值

$$
\begin{aligned}
\hat v_{x,k,0}
&=
\bar u^{\mathrm{water}}(x_{k,0},z_{k,0},t_0)+U_c,\\
\hat v_{z,k,0}
&=
\bar w^{\mathrm{water}}(x_{k,0},z_{k,0},t_0).
\end{aligned}
\tag{3}
$$

从帧 1 开始进入定向跟踪。每个操作块内的顺序是：

1. 用第 $n-1$ 帧的后验状态，沿 Airy–Morison 方程积分一个帧长，得到 $\hat{\boldsymbol{s}}_{k,n|n-1}$ 和预测协方差 $\boldsymbol{P}_{k,n|n-1}$。
2. 由预测坐标生成式 (2) 的发射波束，并在功率约束下为各浮标分配感知功率。需要的话同时更新无人机水平位置，供下一个操作块使用。
3. 发射通感信号，对回波做匹配滤波，得到方向余弦、时延和多普勒。
4. 用这些测量更新 EKF，得到 $\hat{\boldsymbol{s}}_{k,n}$ 和 $\boldsymbol{P}_{k,n}$。
5. 再向前积分一个帧长，得到 $\hat{\boldsymbol{p}}_{k,n+1|n}$。上行接收波束取

$$
\boldsymbol{w}_{k,n+1}
=
\boldsymbol{a}\!\left(\hat{\boldsymbol{p}}_{k,n+1|n},\,\boldsymbol{c}_{\mathrm U,n}\right).
\tag{4}
$$

6. 浮标在第 $n+1$ 帧上行发送观测数据。无人机用估计出的多普勒做补偿。若上行里带有已知导频，可以用这一帧的信道估计再修正一次角度，这一步是可选的，不影响下面的感知理论。

状态积分用子步长 $\Delta_s$，一个无线帧含 $N_s=\Delta_t/\Delta_s$ 个子步。EKF 的量测更新按无线帧做，不按子步做。波浪周期一般是数秒到十余秒，子步取 $0.02\,\mathrm s$ 量级就足以积分 Morison 方程；无线帧可以取 $0.5\,\mathrm s$，与 Li 等（2026）的调度间隔一致。

---

## 2 海杂波反射系数

浮标反射系数沿用前文

$$
\beta_{k,n}=\frac{\chi_{k,n}}{2d_{k,n}},
\tag{5}
$$

并保持 $\chi_{k,n}\approx\chi_{k,0}$。这个 $\chi_{k,0}$ 在帧 0 估计一次，后面当作已知复常数。它已经吸收了参考距离上的散射强度，所以这里不再把 $\beta$ 改写成 $1/d^2$。

海杂波不能只写成一个常数。照射斑的后向散射系数用 Morchin 模型

$$
\begin{aligned}
\bar\chi_{k,n}^{\mathrm{clu}}
&=
\frac{4\times 10^{0.6(\kappa_s+1)-7}}{\lambda}
\,\chi_{k,n}^{0}\sin\varphi_{k,n} \\
&\quad
+\Gamma_s\exp\!\Big(-\tan^2\big(\pi/2-\varphi_{k,n}\big)\,\Gamma_s\Big).
\end{aligned}
\tag{6}
$$

$\kappa_s\in\{0,1,\ldots,9\}$ 是海况等级，$\varphi_{k,n}$ 是掠射角，$\Gamma_s=\cot^2\big(2.44(\kappa_s+1)^{1.08}/57.29\big)$，

$$
\chi_{k,n}^{0}=
\begin{cases}
(\varphi_{k,n}/\varphi_0)^{1.9}, & \varphi_{k,n}\le\varphi_0,\\
1, & \varphi_{k,n}>\varphi_0,
\end{cases}
\qquad
\varphi_0=\arcsin\!\left(\frac{\lambda}{0.1\pi+0.184\pi\kappa_s}\right).
\tag{7}
$$

无人机高度几十米、水平距离几百米时，$\varphi_{k,n}>\varphi_0$ 通常成立，于是 $\chi_{k,n}^{0}=1$，并且 $\sin\varphi_{k,n}=H/d_{k,n}$。杂波斑面积由距离分辨率和方位波束宽度决定。半功率方位波束宽度取 $\phi_{\mathrm{3dB}}=0.886\lambda/(R d_a)$。在 $d_a=\lambda/2$ 时，

$$
A_{k,n}^{\mathrm{sc}}
=
\frac{0.886\,c_0\,d_{k,n}^{2}}{B R\,\bar d_{k,n}},
\tag{8}
$$

其中 $c_0$ 是光速，$B$ 是带宽，$\bar d_{k,n}=\sqrt{(x_{\mathrm U,n}-x_{k,n})^2+(y_{\mathrm U,n}-y_k)^2}$ 是水平距离。把 $\chi_{k,n}^{\mathrm{clu}}=\bar\chi_{k,n}^{\mathrm{clu}}A_{k,n}^{\mathrm{sc}}$ 按与浮标相同的传播因子写成反射系数，得到

$$
\beta_{k,n}^{\mathrm{clu}}
=
\Xi_1\frac{H}{\bar d_{k,n}}
+
\Xi_2\frac{d_{k,n}}{\bar d_{k,n}}
\exp\!\left(-\Gamma_s\Big(\frac{\bar d_{k,n}}{H}\Big)^2\right),
\tag{9}
$$

$$
\begin{aligned}
\Xi_1
&=
\frac{1.772\,c_0\times 10^{0.6(\kappa_s+1)-7}}{B R\lambda},
&
\Xi_2
&=
\frac{0.443\,c_0\,\Gamma_s}{B R}.
\end{aligned}
\tag{10}
$$

式 (9) 只依赖于几何和海况，不依赖于浮标的运动状态方程。因此它只进入测量噪声，不进入第 4 节的状态转移。

---

## 3 雷达测量模型

### 3.1 匹配滤波

对第 $k$ 个浮标的回波做二维匹配滤波

$$
\big\{\hat f^{\mathrm D}_{k,n},\,\hat\tau_{k,n}\big\}
=
\arg\max_{f^{\mathrm D},\,\tau}
\left|
\int_{0}^{\Delta_t}
\boldsymbol{y}_{k,n}(t)\,
\bar x_{k,n}^{*}(t-\tau)\,
e^{-j2\pi f^{\mathrm D}t}\,\mathrm dt
\right|^2.
\tag{11}
$$

匹配滤波增益记为 $M_f$。在预测波束已经落在主瓣内时，

$$
\varpi_{k,n}^{\mathrm{tx}}
=
\boldsymbol{a}^{\mathsf H}(\boldsymbol{p}_{k,n},\boldsymbol{c}_{\mathrm U,n})\,
\boldsymbol{a}(\hat{\boldsymbol{p}}_{k,n|n-1},\boldsymbol{c}_{\mathrm U,n})
\tag{12}
$$

的模接近 1。归一化之后，时延和多普勒估计的信杂噪比为

$$
\mathrm{SCNR}_{k,n}
=
\frac{
p_{k,n}M_f^2\,|\beta_{k,n}|^2\,|\varpi_{k,n}^{\mathrm{tx}}|^2
}{
N_0
+
p_{k,n}M_f^2\,\rho_{k,n}^2\,|\beta_{k,n}^{\mathrm{clu}}|^2
}.
\tag{13}
$$

分子里的 $|\beta_{k,n}|^2|\varpi_{k,n}^{\mathrm{tx}}|^2$ 就是 Li 等（2025）强调的感知信道信息：反射系数、路径距离和波束对准误差一起决定这一帧的测量信息量。分母里的第二项是残余海杂波。它和噪声不同，增大 $p_{k,n}$ 不能把它去掉。海况变高时 $\beta^{\mathrm{clu}}$ 变大，SCNR 会出现由杂波决定的上限。

### 3.2 提供给 EKF 的四个实测量

阵列快拍本身可以当作复测量，Li 等（2026）的费舍尔信息就是这样推导的。对本文的状态维数和后续实现，四个实测量已经包含同一部分几何信息，而且在无人机正下方不会出现方位角的坐标奇异。定义从浮标指向无人机的相对矢量

$$
\boldsymbol{\Delta}_{k,n}
=
\boldsymbol{c}_{\mathrm U,n}-\boldsymbol{p}_{k,n},
\qquad
d_{k,n}=\|\boldsymbol{\Delta}_{k,n}\|,
\qquad
\boldsymbol{e}_{k,n}=\boldsymbol{\Delta}_{k,n}/d_{k,n}.
\tag{14}
$$

方向余弦、双程时延和双程多普勒为

$$
\begin{aligned}
u_{x,k,n}&=\Delta_{k,n}^{x}/d_{k,n},&
u_{y,k,n}&=\Delta_{k,n}^{y}/d_{k,n},\\
\tau_{k,n}&=2d_{k,n}/c_0,\\
f^{\mathrm D}_{k,n}
&=
\frac{2}{\lambda}
\boldsymbol{e}_{k,n}^{\mathsf T}
\big(\boldsymbol{v}_{k,n}-\boldsymbol{v}_{\mathrm U,n}\big).
\end{aligned}
\tag{15}
$$

这里 $\boldsymbol{v}_{k,n}=[v_{x,k,n},\,0,\,v_{z,k,n}]^{\mathsf T}$。垂直速度必须留在多普勒里。$\boldsymbol{e}_{k,n}$ 的垂直分量大约是 $H/d_{k,n}$，升沉速度在斜距上的投影和水平轨道速度是同一量级，删掉 $v_z$ 会把波浪周期里的一部分径向速度当成测量偏差。

无人机在感知帧内的速度 $\boldsymbol{v}_{\mathrm U,n}$ 已知。一个操作块内悬停时它为零；若希望在飞越过程中继续跟踪，保留这一项即可，这与 Zhou 等把平台速度写进多普勒测量是同一做法。

测量方程写成

$$
\boldsymbol{m}_{k,n}
=
\boldsymbol{h}(\boldsymbol{s}_{k,n};\boldsymbol{c}_{\mathrm U,n})
+
\boldsymbol{\nu}_{k,n},
\tag{16}
$$

$$
\boldsymbol{h}
=
\big[u_{x,k,n},\; u_{y,k,n},\; \tau_{k,n},\; f^{\mathrm D}_{k,n}\big]^{\mathsf T},
\qquad
\boldsymbol{\nu}_{k,n}\sim\mathcal N(\boldsymbol 0,\boldsymbol{R}_{k,n}).
$$

$\boldsymbol{R}_{k,n}$ 取对角阵。时延和多普勒的方差用 SCNR 的倒数，系数 $c_\tau$、$c_f$ 由波形的等效带宽和等效时宽决定，仿真中作为波形常数给定：

$$
\sigma_{\tau,k,n}^{2}
=
\frac{c_\tau^{2}}{\mathrm{SCNR}_{k,n}},
\qquad
\sigma_{f,k,n}^{2}
=
\frac{c_f^{2}}{\mathrm{SCNR}_{k,n}}.
\tag{17}
$$

两个方向余弦来自 $R$ 元线阵孔径。在主瓣附近，

$$
\sigma_{u,q,k,n}^{2}
=
\frac{6}{\mathrm{SCNR}_{k,n}\,R(R^2-1)}
\left(\frac{\lambda}{2\pi d_a}\right)^{2}
+
\sigma_{j,q}^{2},
\qquad q\in\{x,y\}.
\tag{18}
$$

第一项随感知功率和波束增益改善而下降。第二项是平台抖动，处理方式对应 Tang 等（2024）：阵面滚转和俯仰把法线方向抖开，等效成方向余弦上的零均值误差。它不随发射功率下降，所以跟踪误差存在由抖动决定的下限。抖动方差可以写成小角度近似 $\sigma_{j,q}^{2}\approx\sigma_{\mathrm{att}}^{2}$，其中 $\sigma_{\mathrm{att}}^{2}$ 是法线角抖动的方差。本文不把波束方向再交给粒子群去搜索；抖动只进入测量协方差，由 EKF 和后面的 PCRB 自动吸收。

式 (18) 有一个使用条件：无人机不能一直处在浮标正上方且 $y_{\mathrm U}=y_k$。那时 $u_y\equiv 0$，方位孔径只提供增益，不提供位置信息。仿真航迹应让无人机相对浮标保持水平横向偏移。

### 3.3 测量雅可比

记 $\boldsymbol{h}$ 对 $\boldsymbol{s}=[x,z,v_x,v_z]^{\mathsf T}$ 的雅可比为 $\boldsymbol{H}\in\mathbb R^{4\times 4}$，在预测状态 $\hat{\boldsymbol{s}}_{k,n|n-1}$ 上取值。位置只通过 $\boldsymbol{p}$ 进入几何，速度只通过多普勒进入测量。

$$
\frac{\partial d}{\partial x}=-e_x,
\qquad
\frac{\partial d}{\partial z}=-e_z,
\qquad
\frac{\partial d}{\partial v_x}=\frac{\partial d}{\partial v_z}=0.
\tag{19}
$$

方向余弦对位置的导数是

$$
\begin{aligned}
\frac{\partial u_x}{\partial x}
&=
-\frac{1-e_x^{2}}{d},
&
\frac{\partial u_x}{\partial z}
&=
\frac{e_x e_z}{d},\\[4pt]
\frac{\partial u_y}{\partial x}
&=
\frac{e_x e_y}{d},
&
\frac{\partial u_y}{\partial z}
&=
\frac{e_y e_z}{d}.
\end{aligned}
\tag{20}
$$

它们对两个速度分量的导数为零。时延一行是

$$
\frac{\partial\tau}{\partial x}=-\frac{2e_x}{c_0},
\qquad
\frac{\partial\tau}{\partial z}=-\frac{2e_z}{c_0},
\qquad
\frac{\partial\tau}{\partial v_x}=\frac{\partial\tau}{\partial v_z}=0.
\tag{21}
$$

多普勒一行同时依赖位置和速度。令 $\boldsymbol{v}_{\mathrm{rel}}=\boldsymbol{v}-\boldsymbol{v}_{\mathrm U}$，则

$$
\frac{\partial f^{\mathrm D}}{\partial s_i}
=
\frac{2}{\lambda}
\left(
\frac{\partial\boldsymbol{e}}{\partial s_i}\cdot\boldsymbol{v}_{\mathrm{rel}}
+
\boldsymbol{e}\cdot\frac{\partial\boldsymbol{v}_{\mathrm{rel}}}{\partial s_i}
\right),
\tag{22}
$$

$$
\frac{\partial\boldsymbol{e}}{\partial s_i}
=
\frac{1}{d}
\left(
\frac{\partial\boldsymbol{\Delta}}{\partial s_i}
-
\boldsymbol{e}\,\frac{\partial d}{\partial s_i}
\right).
$$

其中 $\partial\boldsymbol{\Delta}/\partial x=[-1,0,0]^{\mathsf T}$，$\partial\boldsymbol{\Delta}/\partial z=[0,0,-1]^{\mathsf T}$，$\partial\boldsymbol{v}_{\mathrm{rel}}/\partial v_x=[1,0,0]^{\mathsf T}$，$\partial\boldsymbol{v}_{\mathrm{rel}}/\partial v_z=[0,0,1]^{\mathsf T}$，其余为零。

这个雅可比说明一次单站观测的信息结构：$(u_x,u_y,\tau)$ 约束的是位置，$f^{\mathrm D}$ 只约束径向速度。$v_x$ 和 $v_z$ 不能由一次多普勒完全分开。能够把二者分开的，是下一节里波浪相位已知的确定性加速度。这是本文用 Airy–Morison 模型、而不是只在量测里增加一个角度的原因。

---

## 4 由海洋动力学得到的状态转移

### 4.1 连续时间方程

把前文的确定性加速度记为 $A_x(\boldsymbol{s},t)$ 和 $A_z(\boldsymbol{s},t)$。为了后面写雅可比，把它们重新收成一组封闭表达式。相位 $\psi=kx-\omega t$，深度积分 $\mathcal E(z)=(2/k)e^{kz}\sinh(k L_w/2)$，深度平均水质点速度

$$
\bar u=\alpha_u\mathcal E(z)\cos\psi,
\qquad
\bar w=\alpha_u\mathcal E(z)\sin\psi,
\qquad
\alpha_u=\frac{a\omega}{L_w}.
\tag{23}
$$

相对速度和它的平滑模长为

$$
u_x=\bar u+U_c-v_x,
\quad
u_z=\bar w-v_z,
\quad
R_v=\sqrt{u_x^{2}+u_z^{2}+\varepsilon_v^{2}}.
\tag{24}
$$

$$
\begin{aligned}
F_x^{\mathrm I}&=C_I\mathcal E(z)\sin\psi,\\
F_z^{\mathrm I}&=-C_I\mathcal E(z)\cos\psi,\\
F_x^{\mathrm D}&=C_D' R_v u_x,\\
F_z^{\mathrm D}&=C_D' R_v u_z,\\
F_z^{\mathrm R}
&=
-K_s\big(z-z_0-a\cos\psi\big)
-C_h\big(v_z-a\omega\sin\psi\big).
\end{aligned}
\tag{25}
$$

系数与前文一致：$C_I=\rho C_M A_e a\omega^{2}$，$C_D'=\frac12\rho C_D D_e L_w$，$M_e=m_b+\rho(C_M-1)A_e L_w$。竖直圆柱的静水恢复刚度取水线面提供的静水力刚度

$$
K_s=\rho g A_e,
\qquad
A_e=\frac{\pi D_e^{2}}{4}.
\tag{26}
$$

$C_h$ 取临界阻尼的一个比例，例如 $C_h=2\zeta_h\sqrt{K_s M_e}$，$\zeta_h\in[0.05,0.2]$。这样恢复力不是一个自由曲线拟合参数。

确定性动力学是

$$
\dot{\boldsymbol{s}}
=
\boldsymbol{f}(\boldsymbol{s},t)
=
\begin{bmatrix}
v_x\\
v_z\\
A_x(\boldsymbol{s},t)\\
A_z(\boldsymbol{s},t)
\end{bmatrix},
\quad
\begin{aligned}
A_x&=(F_x^{\mathrm I}+F_x^{\mathrm D})/M_e,\\
A_z&=(F_z^{\mathrm I}+F_z^{\mathrm D}+F_z^{\mathrm R})/M_e.
\end{aligned}
\tag{27}
$$

未建模力只出现在过程噪声里

$$
\dot{\boldsymbol{s}}
=
\boldsymbol{f}(\boldsymbol{s},t)
+
\boldsymbol{G}\boldsymbol{w}(t),
\qquad
\boldsymbol{G}
=
\begin{bmatrix}
0&0\\
0&0\\
1&0\\
0&1
\end{bmatrix},
\tag{28}
$$

$\boldsymbol{w}(t)$ 是功率谱密度为 $\boldsymbol{Q}_c=\mathrm{diag}(\sigma_{a,x}^{2},\sigma_{a,z}^{2})$ 的白噪声。这里的 $\sigma_{a,x}$、$\sigma_{a,z}$ 只代表风、二阶慢漂移和 Morison 经验系数误差，不能再把一阶波浪轨道加速度放进去。一阶轨道加速度已经在 $A_x$、$A_z$ 中。若仍采用 Li 等（2026）里 Singer 模型那种 $0.3\,\mathrm{m/s^2}$ 量级的过程噪声，滤波器会把确定性波浪也当成随机激励，前文推导的动力学就不起作用。

环境参数 $(a,\omega,k,U_c)$ 在一个任务内当作已知慢变量，可由任务开始前的海况观测或帧 0 的一段回波估计。本文的 EKF 状态只含浮标运动学量。把波浪参数也放进状态属于后续工作。

### 4.2 连续雅可比

EKF 的协方差传播需要 $\boldsymbol{F}_c(\boldsymbol{s},t)=\partial\boldsymbol{f}/\partial\boldsymbol{s}$。它的前两行是

$$
\boldsymbol{F}_c
=
\begin{bmatrix}
0&0&1&0\\
0&0&0&1\\
M_e^{-1}\partial_x(F_x^{\mathrm I}+F_x^{\mathrm D})
&
M_e^{-1}\partial_z(F_x^{\mathrm I}+F_x^{\mathrm D})
&
M_e^{-1}\partial_{v_x}F_x^{\mathrm D}
&
M_e^{-1}\partial_{v_z}F_x^{\mathrm D}
\\
M_e^{-1}\partial_x(F_z^{\mathrm I}+F_z^{\mathrm D}+F_z^{\mathrm R})
&
\cdots
&
\cdots
&
\cdots
\end{bmatrix}.
\tag{29}
$$

各分项可以收成矩阵，不必逐个展开成十几条标量公式。记 $\boldsymbol{u}=[u_x,u_z]^{\mathsf T}$，相对速度对状态的雅可比 $\boldsymbol{J}_u\in\mathbb R^{2\times 4}$ 为

$$
\boldsymbol{J}_u
=
\begin{bmatrix}
-k\bar w & k\bar u & -1 & 0\\
k\bar u & k\bar w & 0 & -1
\end{bmatrix}.
\tag{30}
$$

惯性力对水平位置和竖向位置的导数满足旋转关系

$$
\frac{\partial}{\partial x}
\begin{bmatrix}F_x^{\mathrm I}\\ F_z^{\mathrm I}\end{bmatrix}
=
k
\begin{bmatrix}0&-1\\ 1&0\end{bmatrix}
\begin{bmatrix}F_x^{\mathrm I}\\ F_z^{\mathrm I}\end{bmatrix},
\qquad
\frac{\partial}{\partial z}
\begin{bmatrix}F_x^{\mathrm I}\\ F_z^{\mathrm I}\end{bmatrix}
=
k
\begin{bmatrix}F_x^{\mathrm I}\\ F_z^{\mathrm I}\end{bmatrix},
\tag{31}
$$

对两个速度分量为零。阻力对状态的导数是

$$
\frac{\partial\boldsymbol{F}^{\mathrm D}}{\partial\boldsymbol{s}}
=
C_D'
\left(
R_v\boldsymbol{I}_2
+
\frac{\boldsymbol{u}\boldsymbol{u}^{\mathsf T}}{R_v}
\right)
\boldsymbol{J}_u.
\tag{32}
$$

$\varepsilon_v>0$ 保证 $\boldsymbol{u}=\boldsymbol 0$ 时这一式等于 $C_D'\varepsilon_v\boldsymbol{J}_u$，雅可比有界。恢复力只作用在 $z$ 方向，

$$
\begin{aligned}
\frac{\partial F_z^{\mathrm R}}{\partial x}
&=
-K_s a k\sin\psi
+
C_h a\omega k\cos\psi,\\
\frac{\partial F_z^{\mathrm R}}{\partial z}
&=
-K_s,\\
\frac{\partial F_z^{\mathrm R}}{\partial v_z}
&=
-C_h,
\end{aligned}
\tag{33}
$$

其余分量为零。式 (29) 到式 (33) 一起构成连续线性化。它们显含波浪相位 $\psi(x,t)$，所以状态转移是时变的。Singer 模型的转移矩阵是常矩阵，这是二者在滤波器层面的本质差别。

### 4.3 一个无线帧上的离散预测

从 $t_{n-1}$ 到 $t_n=t_{n-1}+\Delta_t$，预测均值沿微分方程本身积分，不能写成常矩阵乘状态。用四阶龙格–库塔，步长 $\Delta_s$：

$$
\hat{\boldsymbol{s}}_{k,n|n-1}
=
\boldsymbol{\Phi}_{t_{n-1}\to t_n}\!\big(\hat{\boldsymbol{s}}_{k,n-1}\big).
\tag{34}
$$

协方差的状态转移矩阵 $\boldsymbol{\Psi}_{k,n-1}$ 是变分方程的解

$$
\frac{\mathrm d}{\mathrm dt}\boldsymbol{\Psi}(t,t_{n-1})
=
\boldsymbol{F}_c\big(\boldsymbol{s}(t),t\big)\,
\boldsymbol{\Psi}(t,t_{n-1}),
\qquad
\boldsymbol{\Psi}(t_{n-1},t_{n-1})=\boldsymbol{I}_4.
\tag{35}
$$

它和状态积分用同一组子步同时计算。过程噪声的离散协方差是

$$
\boldsymbol{Q}_{k,n-1}
=
\int_{t_{n-1}}^{t_n}
\boldsymbol{\Psi}(t_n,\tau)\,
\boldsymbol{G}\boldsymbol{Q}_c\boldsymbol{G}^{\mathsf T}\,
\boldsymbol{\Psi}(t_n,\tau)^{\mathsf T}
\,\mathrm d\tau.
\tag{36}
$$

子步上可用左矩形或 Van Loan 方法累加。短子步上的局部近似是：位置–速度块

$$
\boldsymbol{Q}^{(\Delta_s)}
\approx
\begin{bmatrix}
\tfrac13\Delta_s^{3} & \tfrac12\Delta_s^{2}\\
\tfrac12\Delta_s^{2} & \Delta_s
\end{bmatrix}
\otimes
\boldsymbol{Q}_c
$$

再按 $[x,z,v_x,v_z]$ 的顺序重排。这个近似只用于一个子步，整个无线帧的 $\boldsymbol{Q}_{k,n-1}$ 必须把式 (35) 的 $\boldsymbol{\Psi}$ 乘进去，否则波浪周期内的误差增长方向是错的。

因此，提供给 EKF 的离散模型是

$$
\begin{aligned}
\boldsymbol{s}_{k,n}
&=
\boldsymbol{\Phi}_{t_{n-1}\to t_n}(\boldsymbol{s}_{k,n-1})
+
\boldsymbol{\eta}_{k,n-1},\\
\boldsymbol{m}_{k,n}
&=
\boldsymbol{h}(\boldsymbol{s}_{k,n};\boldsymbol{c}_{\mathrm U,n})
+
\boldsymbol{\nu}_{k,n},
\end{aligned}
\tag{37}
$$

$\boldsymbol{\eta}_{k,n-1}\sim\mathcal N(\boldsymbol 0,\boldsymbol{Q}_{k,n-1})$，$\boldsymbol{\nu}_{k,n}\sim\mathcal N(\boldsymbol 0,\boldsymbol{R}_{k,n})$。均值传播是非线性流 $\boldsymbol{\Phi}$，协方差传播才使用 $\boldsymbol{\Psi}$。Li 等（2026）的 Singer 模型在离散化之后均值也是线性的，所以那里可以用 $\hat{\boldsymbol{s}}_{n|n-1}=\boldsymbol{G}\hat{\boldsymbol{s}}_{n-1}+\bar T\boldsymbol{b}_c$。本文不能把这一步搬过来。

---

## 5 扩展卡尔曼滤波

第 $k$ 个浮标、第 $n$ 个感知帧的滤波按下面三步进行。

**预测。** 用式 (34)、式 (35)、式 (36) 计算

$$
\begin{aligned}
\hat{\boldsymbol{s}}_{k,n|n-1}
&=
\boldsymbol{\Phi}_{t_{n-1}\to t_n}(\hat{\boldsymbol{s}}_{k,n-1}),\\
\boldsymbol{P}_{k,n|n-1}
&=
\boldsymbol{\Psi}_{k,n-1}
\boldsymbol{P}_{k,n-1}
\boldsymbol{\Psi}_{k,n-1}^{\mathsf T}
+
\boldsymbol{Q}_{k,n-1}.
\end{aligned}
\tag{38}
$$

**增益。** 在预测状态上计算 $\boldsymbol{H}_{k,n}$ 和 $\boldsymbol{R}_{k,n}$。计算 $\boldsymbol{R}_{k,n}$ 时，式 (13) 里的真实距离和真实波束误差还不知道，用预测坐标代替，并且在跟踪已经锁住主瓣时取 $|\varpi_{k,n}^{\mathrm{tx}}|=1$。这一近似与 Li 等（2026）相同，它使协方差可以在收到回波之前算好。

$$
\boldsymbol{K}_{k,n}
=
\boldsymbol{P}_{k,n|n-1}
\boldsymbol{H}_{k,n}^{\mathsf T}
\big(
\boldsymbol{H}_{k,n}
\boldsymbol{P}_{k,n|n-1}
\boldsymbol{H}_{k,n}^{\mathsf T}
+
\boldsymbol{R}_{k,n}
\big)^{-1}.
\tag{39}
$$

**更新。**

$$
\begin{aligned}
\hat{\boldsymbol{s}}_{k,n}
&=
\hat{\boldsymbol{s}}_{k,n|n-1}
+
\boldsymbol{K}_{k,n}
\big(
\boldsymbol{m}_{k,n}
-
\boldsymbol{h}(\hat{\boldsymbol{s}}_{k,n|n-1})
\big),\\[4pt]
\boldsymbol{P}_{k,n}
&=
\big(\boldsymbol{I}-\boldsymbol{K}_{k,n}\boldsymbol{H}_{k,n}\big)
\boldsymbol{P}_{k,n|n-1}.
\end{aligned}
\tag{40}
$$

更新后的状态再经 $\boldsymbol{\Phi}_{t_n\to t_{n+1}}$ 得到上行帧使用的 $\hat{\boldsymbol{p}}_{k,n+1|n}$。波束式 (2) 和式 (4) 只用这个点预测，不把协方差展成多个粒子。Tang 等用粒子群是为了在抖动存在时搜索接收波束的折中指向；本文的波束指向已经由点预测决定，抖动的影响放在式 (18) 的方差里。

新息 $\boldsymbol{m}_{k,n}-\boldsymbol{h}(\hat{\boldsymbol{s}}_{k,n|n-1})$ 的样本协方差应与 $\boldsymbol{H}\boldsymbol{P}\boldsymbol{H}^{\mathsf T}+\boldsymbol{R}$ 同量级。若仿真中新息持续远大于这个理论值，说明 $(a,\omega,k)$ 或 $C_D,C_M$ 与真实海况不一致，应先检查模型，而不是加大滤波器增益。

---

## 6 预测位置的后验克拉美罗界

资源优化需要一个不依赖某一次噪声实现的感知指标。用后验费舍尔信息矩阵。先验由式 (38) 的预测协方差给出，

$$
\boldsymbol{s}_{k,n}
\sim
\mathcal N\!\big(\hat{\boldsymbol{s}}_{k,n|n-1},\,\boldsymbol{P}_{k,n|n-1}\big).
\tag{41}
$$

这是实高斯。测量在线性化之后也是实高斯，于是后验费舍尔信息矩阵是

$$
\boldsymbol{J}_{k,n}
=
\boldsymbol{P}_{k,n|n-1}^{-1}
+
\boldsymbol{H}_{k,n}^{\mathsf T}
\boldsymbol{R}_{k,n}^{-1}
\boldsymbol{H}_{k,n}.
\tag{42}
$$

任何无偏估计的误差协方差都不小于 $\boldsymbol{J}_{k,n}^{-1}$。式 (42) 的第二项按四条测量拆开，

$$
\boldsymbol{J}_{k,n}
=
\boldsymbol{P}_{k,n|n-1}^{-1}
+
\sum_{i=1}^{4}
\omega_{i,k,n}\,\boldsymbol{A}_{i,k,n},
\tag{43}
$$

$\omega_{i,k,n}=1/\sigma_{i,k,n}^{2}$，$\boldsymbol{A}_{i,k,n}=\boldsymbol{h}_i\boldsymbol{h}_i^{\mathsf T}$，$\boldsymbol{h}_i^{\mathsf T}$ 是 $\boldsymbol{H}_{k,n}$ 的第 $i$ 行。方向余弦那两项的 $\omega$ 里含有抖动方差，所以 $\omega$ 不是 SCNR 的正比函数，优化时不要把式 (18) 误写成与功率成反比的单一项。

上行波束使用的是再向前一帧的位置。令 $\boldsymbol{E}_{xz}=[\boldsymbol{I}_2\;\boldsymbol{0}_{2\times 2}]^{\mathsf T}$ 从状态中取出 $[x,z]$，下一帧位置误差的线性化传播是

$$
\boldsymbol{S}_{k,n}
=
\boldsymbol{E}_{xz}^{\mathsf T}
\boldsymbol{\Psi}_{k,n},
\tag{44}
$$

其中 $\boldsymbol{\Psi}_{k,n}$ 是从 $t_n$ 积分到 $t_{n+1}$ 的变分矩阵，在预测状态上计算，与功率和无人机位置无关。下一帧位置的 PCRB 为

$$
\Theta_{k,n}
=
\mathrm{tr}\!\big(
\boldsymbol{S}_{k,n}
\boldsymbol{J}_{k,n}^{-1}
\boldsymbol{S}_{k,n}^{\mathsf T}
\big)
+
\mathrm{tr}\!\big(
\boldsymbol{E}_{xz}^{\mathsf T}
\boldsymbol{Q}_{k,n}
\boldsymbol{E}_{xz}
\big).
\tag{45}
$$

第二项不含优化变量。最小化 $\Theta_{k,n}$ 等价于最小化第一项。

这里和 Li 等（2026）有一处故意做了不同处理。他们的传感指标是当前帧水平位置的后验 PCRB。那个指标对线性常矩阵转移是合理的。本文的 $\boldsymbol{\Psi}_{k,n}$ 会把速度误差积分成下一帧的位置误差，而速度误差正好由多普勒和 Morison 耦合决定。若只惩罚当前的 $x,z$ 方差，优化器会低估多普勒测量的价值。式 (45) 把海洋模型的 $\boldsymbol{\Psi}_{k,n}$ 放进传感指标，使“轨迹预测误差”而不是“当前定位误差”成为优化目标。

$\boldsymbol{J}_{k,n}$ 依赖 $p_{k,n}$ 和 $\boldsymbol{c}_{\mathrm U,n}$，因为 SCNR、杂波系数和 $\boldsymbol{H}_{k,n}$ 都依赖几何。$\boldsymbol{P}_{k,n|n-1}$ 和 $\boldsymbol{\Psi}_{k,n}$ 在求解当前帧的资源问题之前已经由滤波器算完，优化时当作常数。海洋模型和资源分配在这一帧上是分开的：前者产生预测和先验信息，后者决定测量信息。

---

## 7 上行通信模型

浮标是单天线，发射功率 $P_b$ 固定。第 $n+1$ 帧无人机收到的信号是

$$
\begin{aligned}
\tilde y_{k,n+1}(t)
&=
\sqrt{P_b}\,
\alpha_{k,n+1}\,
e^{j2\pi f^{\mathrm D}_{k,n+1}t}\,
\boldsymbol{w}_{k,n+1}^{\mathsf H}
\boldsymbol{a}(\boldsymbol{p}_{k,n+1},\boldsymbol{c}_{\mathrm U,n})\,
\tilde x_{k,n+1}(t) \\
&\quad
+z_c(t).
\end{aligned}
\tag{46}
$$

信道系数按视距写成

$$
\alpha_{k,n+1}
=
\tilde\alpha\,
d_{k,n+1}^{-1}
\exp\!\big(j 2\pi d_{k,n+1}/\lambda\big),
\tag{47}
$$

$\tilde\alpha$ 是 1 m 处的复增益。多普勒已由感知帧估计并补偿。大规模阵列的接收波束压住了浮标间干扰，因此速率式里不再加多用户干扰项。这是海上稀疏浮标场景的建模选择；若两个浮标的角间隔小于波束宽度，应回到 Zhou 等的干扰速率公式。

记接收失配

$$
\varpi_{k,n+1}^{\mathrm{rx}}
=
\boldsymbol{a}^{\mathsf H}(\hat{\boldsymbol{p}}_{k,n+1|n},\boldsymbol{c}_{\mathrm U,n})\,
\boldsymbol{a}(\boldsymbol{p}_{k,n+1},\boldsymbol{c}_{\mathrm U,n}).
\tag{48}
$$

则

$$
\mathrm{SNR}_{k,n+1}
=
\frac{P_b\,|\tilde\alpha|^{2}\,|\varpi_{k,n+1}^{\mathrm{rx}}|^{2}}{d_{k,n+1}^{2} N_0},
\qquad
R_{k,n+1}
=
\log_2(1+\mathrm{SNR}_{k,n+1}).
\tag{49}
$$

$|\varpi^{\mathrm{rx}}|^{2}$ 是波束增益损失，由预测位置误差决定；$d_{k,n+1}$ 由无人机位置决定。二者都进入速率，但在优化时刻的已知程度不同。第 $n$ 帧的功率和位置必须在收到第 $n$ 帧回波之前定下来，那时 $\hat{\boldsymbol{p}}_{k,n+1|n}$ 还不存在。因果的做法是用两步预测 $\hat{\boldsymbol{p}}_{k,n+1|n-1}$ 写路径损耗，并把 $|\varpi^{\mathrm{rx}}|$ 先取 1。由此得到速率门槛 $R_s$ 的充分近似

$$
\big\|\boldsymbol{c}_{\mathrm U,n}-\hat{\boldsymbol{p}}_{k,n+1|n-1}\big\|^{2}
\le
\frac{P_b\,|\tilde\alpha|^{2}}{N_0(2^{R_s}-1)}.
\tag{50}
$$

这是关于 $\boldsymbol{c}_{\mathrm U,n}$ 的二阶锥约束。真实速率仍用式 (49) 评估。海况变差、预测脱离主瓣时，式 (50) 可能被满足而式 (49) 不满足。性能评估必须同时报告这两个量，不能只报告约束余量。

本文不把 $\boldsymbol{w}_{k,n}$ 或 $\boldsymbol{f}_{k,n}$ 放开成自由复矩阵。Zhou 等最大化的是带多用户干扰的下行和速率，波束矩阵本身是优化变量，所以需要半定松弛和秩一迭代。本文的链路是视距上行采集，感知和通信波束都是预测方向上的匹配导向矢量，剩下真正还能设计的是感知功率和无人机水平位置。Li 等（2025）把感知信道放进 PCRB、再用 PCRB 设计波束；在导向矢量结构已经由式 (2) 固定之后，感知信道通过式 (13) 进入 $\boldsymbol{J}_{k,n}$，设计变量退化为 $\{p_{k,n}\}$ 和 $\boldsymbol{c}_{\mathrm U,n}$。

---

## 8 优化问题

每个感知帧求解

$$
\begin{aligned}
(\mathrm{P1})\quad
\min_{\{p_{k,n}\},\,x_{\mathrm U,n},\,y_{\mathrm U,n}}
\quad
&
\max_{k\in\{1,\ldots,K\}}
\Theta_{k,n} \\
\mathrm{s.t.}\quad
&
R_{k,n+1}\ge R_s,
&& \forall k, \\
&
\sum_{k=1}^{K}p_{k,n}\le P_{\mathrm U},
\quad p_{k,n}>0,
&& \forall k, \\
&
\big\|\boldsymbol{c}_{\mathrm U,n}-\boldsymbol{c}_{\mathrm U,n-1}\big\|
\le
V_{\max}\Delta_t.
\end{aligned}
\tag{51}
$$

目标是最差浮标的下一帧位置 PCRB。第一条约束用式 (50) 实现。第二条是感知功率预算，$P_{\mathrm U}$ 只计射频功率，不计飞行功耗。第三条是相邻操作块之间的速度约束。高度 $H$ 不优化。

引入 $\vartheta_n$ 和 $\boldsymbol{U}_{k,n}\in\mathbb S_{+}^{2}$，把最小最大写成

$$
\begin{aligned}
(\mathrm{P2})\quad
\min\quad
&
\vartheta_n \\
\mathrm{s.t.}\quad
&
\mathrm{tr}(\boldsymbol{U}_{k,n})\le\vartheta_n,
&&\forall k,\\
&
\boldsymbol{U}_{k,n}
\succeq
\boldsymbol{S}_{k,n}
\boldsymbol{J}_{k,n}^{-1}
\boldsymbol{S}_{k,n}^{\mathsf T},
&&\forall k,\\
&
\text{式 (51) 的三个约束}.
\end{aligned}
\tag{52}
$$

$\boldsymbol{J}_{k,n}\succ 0$ 时，第二行等价于线性矩阵不等式

$$
\begin{bmatrix}
\boldsymbol{J}_{k,n} & \boldsymbol{S}_{k,n}^{\mathsf T}\\
\boldsymbol{S}_{k,n} & \boldsymbol{U}_{k,n}
\end{bmatrix}
\succeq 0.
\tag{53}
$$

非凸性只来自 $\boldsymbol{J}_{k,n}$ 中的 $\omega_{i,k,n}$ 和 $\boldsymbol{A}_{i,k,n}$。$\boldsymbol{A}_{i,k,n}$ 依赖导向几何和杂波系数，随无人机位置变化，但在主瓣跟踪区内变化慢。采用 Li 等（2026）的滞后逐次凸近似：第 $\ell$ 次迭代把 $\boldsymbol{A}_{i,k,n}$ 和 $\beta_{k,n}^{\mathrm{clu}}$ 固定在上一轮的无人机位置上，只对标量权重做凸下界。

对时延和多普勒，方差上限引入辅助变量 $\nu_{i,k,n}$，

$$
\nu_{i,k,n}
\ge
c_i^{2}
\left(
\frac{4N_0\,\hat d_{k,n|n-1}^{2}}{M_f^{2}|\chi_{k}|^{2}\,p_{k,n}}
+
\frac{4\rho_{k,n}^{2}\,|\beta_{k,n}^{\mathrm{clu},(\ell-1)}|^{2}\,\hat d_{k,n|n-1}^{2}}{|\chi_{k}|^{2}}
\right),
\quad i\in\{\tau,f\}.
\tag{54}
$$

右边第一项是 $1/p_{k,n}$ 的正常数倍，第二项在本轮是常数。方向余弦的方差多一个与功率无关的抖动项，写成

$$
\nu_{q,k,n}
\ge
\frac{6}{R(R^2-1)}
\left(\frac{\lambda}{2\pi d_a}\right)^{2}
\sigma_{\mathrm{SCNR},k,n}^{2}
+
\sigma_{j,q}^{2},
\tag{55}
$$

其中 $\sigma_{\mathrm{SCNR},k,n}^{2}$ 表示式 (54) 括号内、去掉 $c_i^{2}$ 之后的同一信杂噪比倒数，用 $\beta^{\mathrm{clu},(\ell-1)}$ 和预测距离计算。于是 $\omega_{i,k,n}\ge 1/\nu_{i,k,n}$。$1/\nu$ 在 $\nu>0$ 上凸，在 $\nu_{i,k,n}^{(\ell-1)}$ 处的切线是全局下界

$$
\frac{1}{\nu_{i,k,n}}
\ge
\underline\omega_{i,k,n}^{(\ell)}(\nu_{i,k,n})
=
\frac{1}{\nu_{i,k,n}^{(\ell-1)}}
-
\frac{\nu_{i,k,n}-\nu_{i,k,n}^{(\ell-1)}}{\big(\nu_{i,k,n}^{(\ell-1)}\big)^{2}}.
\tag{56}
$$

本轮使用的保守信息矩阵是仿射的，

$$
\widetilde{\boldsymbol{J}}_{k,n}^{(\ell)}
=
\boldsymbol{P}_{k,n|n-1}^{-1}
+
\sum_{i=1}^{4}
\underline\omega_{i,k,n}^{(\ell)}(\nu_{i,k,n})\,
\boldsymbol{A}_{i,k,n}^{(\ell-1)}.
\tag{57}
$$

因为 $\underline\omega\le 1/\nu$ 且 $\boldsymbol{A}_{i}\succeq 0$，有 $\widetilde{\boldsymbol{J}}^{(\ell)}\preceq\boldsymbol{J}$ 在滞后几何下的对应值。用它替换式 (53) 中的 $\boldsymbol{J}$，得到的是内近似，可行点对原问题保守。

第 $\ell$ 轮求解凸问题

$$
\begin{aligned}
(\mathrm{P3})\quad
\min\quad
&
\vartheta_n \\
\mathrm{s.t.}\quad
&
\mathrm{tr}(\boldsymbol{U}_{k,n})\le\vartheta_n,\\
&
\begin{bmatrix}
\widetilde{\boldsymbol{J}}_{k,n}^{(\ell)} & \boldsymbol{S}_{k,n}^{\mathsf T}\\
\boldsymbol{S}_{k,n} & \boldsymbol{U}_{k,n}
\end{bmatrix}\succeq 0,\\
&
\text{式 (50)、功率预算、速度约束、式 (54)、式 (55)},\\
&
\underline\omega_{i,k,n}^{(\ell)}(\nu_{i,k,n})\ge 0,\\
&
\big\|\boldsymbol{c}_{\mathrm U,n}^{(\ell)}-\boldsymbol{c}_{\mathrm U,n}^{(\ell-1)}\big\|\le\delta^{(\ell)}.
\end{aligned}
\tag{58}
$$

最后一行是信赖域，用来限制滞后近似的步长。$\boldsymbol{S}_{k,n}$ 和 $\boldsymbol{P}_{k,n|n-1}$ 在整个 SCA 循环中不变。

算法如下。

1. 输入预测状态、$\boldsymbol{P}_{k,n|n-1}$、$\boldsymbol{S}_{k,n}$、上一操作块的无人机位置、功率和速率门槛。
2. 用当前无人机位置初始化 $p_{k,n}^{(0)}=P_{\mathrm U}/K$，计算 $\boldsymbol{A}_{i,k,n}^{(0)}$、$\beta^{\mathrm{clu},(0)}$ 和可行的 $\nu^{(0)}$。
3. 对 $\ell=1,2,\ldots,I_{\max}$：用 $\boldsymbol{c}_{\mathrm U,n}^{(\ell-1)}$ 更新 $\boldsymbol{A}_{i}$ 和 $\beta^{\mathrm{clu}}$；按式 (56) 写 $\underline\omega$；按式 (57) 组装 $\widetilde{\boldsymbol{J}}$；求解 (P3)。
4. 若 $|\vartheta_n^{(\ell)}-\vartheta_n^{(\ell-1)}|/\max\{1,\vartheta_n^{(\ell-1)}\}\le\epsilon$，停止。
5. 输出 $\{p_{k,n}^{\star}\}$ 和 $\boldsymbol{c}_{\mathrm U,n}^{\star}$。用这组功率发射当前感知帧；$\boldsymbol{c}_{\mathrm U,n}^{\star}$ 在本操作块结束后飞到，并作为下一操作块的悬停位置。

每个 (P3) 含 $K$ 个 $6\times 6$ 的线性矩阵不等式，标量变量数目是 $\mathcal O(K)$。内点法每步代价是 $\mathcal O(K^3)$，外层次数记为 $I_{\mathrm{SCA}}$，总复杂度 $\mathcal O(I_{\mathrm{SCA}}I_{\mathrm{IP}}K^3)$。状态维数是 4 而不是 Singer 模型的 6，所以矩阵块比 Li 等（2026）更小，量级不变。

全量一阶泰勒展开可以把 $\boldsymbol{A}_{i,k,n}$ 对无人机位置的导数也线性化，近似更紧，但每一轮都要对全部浮标重算信息矩阵的导数。滞后更新适合本文的跟踪区：EKF 把指向误差留在主瓣内，$\boldsymbol{A}_{i,k,n}$ 在相邻迭代之间变化平缓。仿真里把全量线性化作为复杂度对照，不作为主算法。

---

## 9 性能评估

优化用 $\Theta_{k,n}$，评估用实际误差和实际速率。二者不要混用。建议的指标如下。

**轨迹预测。** 在上行帧比较 $\hat{\boldsymbol{p}}_{k,n+1|n}$ 与真实散射点。分别统计 $x$、$\zeta$ 和水平面位置的 RMSE，以及 $v_x$、$v_z$ 的 RMSE。同时画出 $\Theta_{k,n}$ 和位置 RMSE。模型匹配且噪声接近高斯时，RMSE 应贴着 PCRB；明显低于 PCRB 通常是估算 RMSE 的样本太少，而不是滤波器优于界。

**波束。** 记录 $|\varpi_{k,n}^{\mathrm{tx}}|^{2}$ 和 $|\varpi_{k,n+1}^{\mathrm{rx}}|^{2}$，以及波束脱离半功率宽度的帧比例。只报告平均增益会掩盖偶尔的失锁。

**通信。** 最差浮标的上行速率 $R_{k,n+1}$，速率低于 $R_s$ 的帧比例，以及全部浮标的速率和。式 (50) 的几何约束是否成立单独统计。

**资源。** 各浮标功率占比，无人机水平航迹，以及 (P3) 的迭代次数和求解时间。

对照算法保持同一测量模型和同一优化器，只替换被检验的那一块。

| 对照 | 替换内容 | 用来说明的问题 |
| --- | --- | --- |
| Singer-EKF | 用 Li 等（2026）的相关加速度模型产生预测，过程噪声取他们的 $\sigma_a$ | 确定性波浪模型相对于统计加速度模型的预测收益 |
| 匀速 EKF | $\dot{\boldsymbol{v}}=\boldsymbol 0$，过程噪声只加在速度上 | 连加速度相关性都不使用时的性能 |
| 开环预测 | 保留 Airy–Morison 积分，但令卡尔曼增益为零 | 回波更新是否必要 |
| 等功率 | 航迹仍优化，$p_{k,n}=P_{\mathrm U}/K$ | 功率分配相对于航迹的边际收益 |
| 固定无人机 | 功率仍优化，位置锁在初始点 | 几何调整的收益 |
| 无多普勒 | 测量只留 $u_x,u_y,\tau$ | 前文若删除多普勒，速度是否仍可观测 |
| 无垂向多普勒 | $f^{\mathrm D}$ 里去掉 $v_z$ | 升沉速度对径向速度的贡献 |

Singer 对照的真实轨迹必须仍由 Airy–Morison 方程生成。若真实轨迹也改成 Singer 过程，实验就不再检验本文的海洋模型。

建议的默认参数分成三组。通信和平台参数对齐 Li 等（2026），以便对照；浮标和水动力参数按前文的线性波假设闭合，不另设一套互相矛盾的海况。

通信与平台：

| 参数 | 取值 |
| --- | --- |
| 海域 | $400\,\mathrm m\times 400\,\mathrm m$，中心在原点 |
| 帧数与帧长 | $N=100$，$\Delta_t=0.5\,\mathrm s$，子步 $\Delta_s=0.02\,\mathrm s$ |
| 浮标数 | $K=9$，另做 $K=15$ |
| 无人机 | 初始 $[120,120,50]^{\mathsf T}\,\mathrm m$，$V_{\max}=30\,\mathrm{m/s}$，$H=50\,\mathrm m$ |
| 阵列与载波 | $8\times 8$，必要时加一组 $32\times 32$；$f_c=18\,\mathrm{GHz}$，$B=100\,\mathrm{MHz}$，$d_a=\lambda/2$ |
| 功率与噪声 | $P_{\mathrm U}=36\,\mathrm{dBm}$，$P_b=20\,\mathrm{dBm}$，$N_0=-174\,\mathrm{dBm/Hz}$ |
| 速率门槛 | $R_s=5\,\mathrm{bit/s/Hz}$ |
| 海况与杂波 | $\kappa_s=5$，$\rho=0.2$，$(c_\tau,c_f)$ 先取 $(0.1,0.1)$ 再做放大 |
| 姿态抖动 | $\sigma_{\mathrm{att}}=0$，以及 $0.2^\circ$、$0.5^\circ$ 两组 |

浮标与波浪，一组互相闭合的默认值：

| 参数 | 取值 | 来源 |
| --- | --- | --- |
| $D_e$ | $0.6\,\mathrm m$ | 竖直圆柱等效直径 |
| $L_w$ | $0.8\,\mathrm m$ | 短时窗口内浸没长度 |
| $A_e$ | $\pi D_e^{2}/4$ | 前文截面积，也是水线面积 |
| $\rho$ | $1025\,\mathrm{kg/m^3}$ | 海水密度 |
| $C_D,\,C_M$ | $1.2,\,2.0$ | 前文按线性波取值 |
| $m_b$ | $\rho A_e L_w$ | 与浸没体积对应的排水质量 |
| $M_e$ | $m_b+\rho(C_M-1)A_e L_w$ | 含附加质量 |
| $K_s$ | $\rho g A_e$ | 式 (26) |
| $C_h$ | $2\zeta_h\sqrt{K_s M_e}$，$\zeta_h=0.1$ | 轻阻尼升沉 |
| $a,\,T_p$ | $0.8\,\mathrm m,\,8\,\mathrm s$ | 波陡 $a/\lambda\ll 1$ |
| $\omega,\,k$ | $2\pi/T_p,\;\omega^{2}/g$ | 深水色散 |
| $U_c$ | $0.4\,\mathrm{m/s}$，沿 $x$ | 短时窗口内的定常海流 |
| $\sigma_{a,x},\sigma_{a,z}$ | $0.05\,\mathrm{m/s^2}$ | 只代表模型误差 |
| $\varepsilon_v$ | $10^{-3}\,\mathrm{m/s}$ | 阻力雅可比的平滑项 |
| 水深 | 大于 $\lambda/2$ | 保证深水假设 |

这组波浪的波长约 $100\,\mathrm m$，波陡约 $0.008$，与前文线性微幅、深水两个假设一致。升沉固有周期大约 $2.5\,\mathrm s$，短于波浪周期，浮标会跟随波面并带有相位滞后，Morison 恢复力会表现出这个滞后。不要把 $\sigma_{a}$ 改成 $0.3\,\mathrm{m/s^2}$ 来“对齐”Singer 论文；那个数是用来代替波浪的，不是模型误差。

图按下面的顺序做。

1. 一条任务内的真实浮标轨迹、EKF 预测轨迹和无人机水平航迹。用来检查滤波器是否跟得上波浪振荡和海流漂移，以及无人机是飞向浮标群的折中位置，而不是尾随某一个浮标。
2. 最差浮标的位置 RMSE、$\Theta_{k,n}$ 和上行速率随帧号的变化，蒙特卡洛不少于 100 次。初始阶段 RMSE 和 PCRB 可以不一致，因为初值来自帧 0 的粗估计；进入跟踪区后二者应一起平稳。
3. 只放大 $\sigma_a$、只放大 $(c_\tau,c_f)$、只提高 $\kappa_s$ 的三条曲线。过程噪声放大考察模型误差，测量常数放大考察回波质量，海况升高同时改变杂波和真实波浪，应单独增加波幅和波周期，而不是只改 $\kappa_s$。
4. $K=9$ 和 $K=15$ 的最差浮标速率与 RMSE 的经验分布。浮标变多时最差用户变差，是最小最大准则的直接结果。
5. 第 9 节表格中的七个对照，在 $K=15$、海况高于默认值的条件下比较 RMSE 和速率。
6. 滞后 SCA 与全量一阶线性化的 $\vartheta_n$、速率和单次任务求解时间。
7. $8\times 8$ 与 $32\times 32$ 两组阵列上的波束失配和失锁比例。默认的 $8\times 8$ 在 50 m 高度、百米级斜距上，半功率波束宽度约 $13^\circ$。亚米级波浪位移对应的角度误差远小于这个宽度，平均波束增益不会出现数量级变化。大阵列或更长的预测间隔才会让指向误差进入主瓣边缘。两组都要做，否则容易把“速率几乎不变”误读成模型无效，或把小的增益差写成大幅提升。

由模型可以直接推出、并应在上述图里核对的结论是这些。

- 开环预测的位置误差随时间积累。回波更新把误差拉回到由 SCNR 和先验共同决定的水平。通信只靠运动学外推时，失锁帧比例会明显高于有感知更新的方案。
- 在真实运动由 Airy–Morison 方程生成时，用同一方程做预测，所需的过程噪声只覆盖模型误差。Singer 模型必须用较大的随机加速度去包络整个轨道运动，因此同样的预测时长下，它的 $\boldsymbol{P}_{n|n-1}$ 更大，下一帧位置 RMSE 更高。差距在测量变差、预测间隔变长时更明显，因为这时滤波器更依赖状态方程。
- 去掉多普勒之后，$v_x$ 与 $v_z$ 主要靠动力学耦合来区分，速度 RMSE 上升，并在下一帧转化为位置误差。式 (45) 把 $\boldsymbol{\Psi}$ 放进目标之后，优化器会给多普勒信息足够的权重；若改回只惩罚当前帧位置方差，功率分配对速度测量不敏感。
- 残余海杂波使 SCNR 在大功率下饱和。海况升高时，继续增大 $P_{\mathrm U}$ 的边际收益下降，航迹带来的几何改善仍然有用，因为杂波系数和浮标距离都随水平距离变化。
- 等功率与优化功率的差距通常小于固定航迹与优化航迹的差距。最差浮标的位置和距离主要由无人机放在哪里决定。
- 平台抖动加入式 (18) 之后，PCRB 和 RMSE 出现不随功率下降的地板。这是 Tang 等的现象在测量方差里的对应结果。
- $8\times 8$、18 GHz、0.5 s 帧长下，主瓣内的速率差异来自路径损耗和偶发失锁，不来自亚米级振荡的平均指向损失。$32\times 32$ 或把预测间隔加到数秒时，Airy–Morison 预测与 Singer、匀速模型的波束增益差才会清楚显示出来。

这些是模型结构决定的趋势。具体的 RMSE、速率差值和求解时间，以实际仿真为准。

---

## 10 结论中可以写的内容

本文在深水 Airy 波和相对运动 Morison 方程上建立了浮标的四维状态方程，并把该方程的非线性流和变分矩阵分别用于 EKF 的均值预测与协方差预测。雷达侧保留了视距双程回波、残余海杂波和方向余弦、时延、多普勒四类测量；通信侧使用由预测坐标生成的匹配波束，上行速率由路径损耗和波束失配共同决定。感知性能用下一帧位置的 PCRB 描述，其中海洋模型的状态转移矩阵决定速度误差如何变成指向误差。在此指标上，每个感知帧联合设计感知功率和无人机水平位置，速率、功率和速度约束经滞后逐次凸近似后成为凸锥规划。

这个框架里，复杂海洋模型承担的是状态转移，不是另一套波束成形算法。波束成形、杂波和 PCRB 优化沿用通感预测波束的现有结构。海洋模型带来的可检验收益，是更小的过程噪声、可预测的波浪相位，以及多普勒与升沉–漂移速度之间的正确耦合。

后续若继续扩展，比较自然的三件事是：把 $(a,\omega,k)$ 放进增广状态做联合估计；在角间隔小于波束宽度时恢复浮标间干扰；用多架无人机提供不同视线，以分离单站多普勒无法直接观测的切向速度。平台抖动若要进一步做成波束形状设计，再引入 Zhou 等的主瓣覆盖约束或 Tang 等的波束搜索，作为本文匹配波束之外的鲁棒层。

---

## 附录 A 实现时的计算顺序

对每个感知帧 $n$，对每个浮标 $k$：

1. 以 $\hat{\boldsymbol{s}}_{k,n-1}$ 为初值，步长 $\Delta_s$，积分式 (27) 至 $t_n$，同步积分式 (35)，并累加式 (36)。得到 $\hat{\boldsymbol{s}}_{k,n|n-1}$、$\boldsymbol{P}_{k,n|n-1}$。
2. 用预测状态再积分一个帧长，得到两步预测位置，供式 (50) 使用；同时保存这一段的 $\boldsymbol{\Psi}_{k,n}$，组装式 (44) 的 $\boldsymbol{S}_{k,n}$。这一步的 $\boldsymbol{\Psi}_{k,n}$ 在收到测量之前计算，线性化点是 $\hat{\boldsymbol{s}}_{k,n|n-1}$ 向前积分的名义轨迹。
3. 求解 (P3)，得到 $p_{k,n}^{\star}$ 和下一悬停点。
4. 按式 (2) 形成发射波束并接收回波，由式 (11) 得到 $\hat u_x,\hat u_y,\hat\tau,\hat f^{\mathrm D}$。方向余弦由匹配滤波后的阵列快拍估计，仿真中可直接用真实值加式 (17)、式 (18) 的噪声来等效，以避免在理论验证阶段再实现一整套谱估计。
5. 按式 (39)、式 (40) 更新。
6. 从后验状态积分到 $t_{n+1}$，按式 (4) 形成接收波束，按式 (49) 计算真实速率。

第 4 步若改用真实值加噪声，对比的是滤波与优化，不是角度估计算法。论文里应写明这一等效。若后续补上 Capon 或 GLRT 角度估计，只需把式 (18) 的理论方差换成样本方差，EKF 和 (P1) 的形式不变。

## 附录 B 符号

| 符号 | 含义 |
| --- | --- |
| $K,\,n,\,\Delta_t,\,\Delta_s$ | 浮标数、帧序号、无线帧长、积分子步 |
| $R,\,\bar R,\,d_a,\,\lambda$ | 每维阵元数、总阵元数、阵元间距、波长 |
| $\boldsymbol{s},\,\boldsymbol{p},\,\zeta$ | 运动状态、散射点坐标、相对静水面的高度 |
| $a,\omega,k,U_c$ | 波幅、角频率、波数、沿 $x$ 的海流 |
| $M_e,K_s,C_h,C_D,C_M$ | 等效质量、恢复刚度、垂向阻尼、阻力系数、惯性力系数 |
| $\beta,\,\beta^{\mathrm{clu}},\,\rho,\,\kappa_s$ | 浮标反射系数、杂波反射系数、残余杂波比例、海况 |
| $u_x,u_y,\tau,f^{\mathrm D}$ | 方向余弦、双程时延、双程多普勒 |
| $\boldsymbol{\Phi},\,\boldsymbol{\Psi},\,\boldsymbol{Q}$ | 非线性状态流、变分矩阵、离散过程噪声协方差 |
| $\boldsymbol{J},\,\Theta,\,\boldsymbol{S}$ | 后验信息矩阵、下一帧位置 PCRB、位置提取矩阵 |
| $p_{k,n},\,P_{\mathrm U},\,P_b,\,R_s$ | 感知功率、无人机功率预算、浮标功率、速率门槛 |

## 参考文献（本文直接沿用的四篇）

1. S. Zhou, H. Yang, L. Xiang, and K. Yang, “Temporal-assisted beamforming and trajectory prediction in sensing-enabled UAV communications,” *IEEE Trans. Commun.*, vol. 73, no. 7, pp. 5408–5419, Jul. 2025.
2. M. Li, F. Dong, T. Liu, and F. Liu, “EKF-based beamforming design for joint beam tracking and communication systems,” *IEEE Wireless Commun. Lett.*, vol. 14, no. 9, pp. 2937–2941, Sep. 2025.
3. Y. Tang, W. Liu, J. Zhu, and C. Li, “Sensing aided beam tracking for UAV with jittering effect,” *IEEE Wireless Commun. Lett.*, vol. 13, no. 8, pp. 2150–2154, Aug. 2024.
4. B. Li, G. Gao, J. Zhang, M. Ye, Q. Li, H. Yu, J. Wang, P. Xiao, and S. Chen, “Sensing-assisted predictive beamforming for UAV-enabled ocean monitoring networks,” arXiv:2606.16717, 2026.
