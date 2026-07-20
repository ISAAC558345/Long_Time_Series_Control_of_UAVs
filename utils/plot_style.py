def setup_chinese_matplotlib(matplotlib_module):
    """Configure matplotlib to prefer common Chinese fonts.

    The font list is intentionally permissive: matplotlib will fall back to its
    default font when none of these fonts are installed.
    """
    matplotlib_module.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
    ]
    matplotlib_module.rcParams["axes.unicode_minus"] = False

