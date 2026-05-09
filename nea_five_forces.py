import numpy as np

# ============================================================
# 1. N.E.A. 宇宙硬件原始参数 (锁定凭证)
# ============================================================
Z_UNIT_MEV = 0.406640           # 1 ZY = 0.4066 MeV
L_C = 1.18e-15                  # 寻址步长 (核半径 ~1.18 fm)
N_MAX = np.exp(10 * np.sqrt(3)) # 25位寻址页深 ~3.32e7
ALPHA_INV = 137.03599           # 寻址交换率 (精细结构常数倒数)
ETA = 0.85                      # 空间本征阻抗

# 力的本位币换算：1 ZY / L_C 转换为 Newton (J/m)
# F0 = (0.4066 MeV * 1.60218e-13 J/MeV) / 1.18e-15 m = 55.242 Newton
F0 = (Z_UNIT_MEV * 1.60218e-13) / L_C 

# ============================================================
# 2. 统一结算引擎
# ============================================================
def nea_settle_force(r, force_type):
    n = r / L_C  # 逻辑跳数
    
    # 债务本金 (Debt)
    debt_p = 938.272 / Z_UNIT_MEV  # 质子质量债务 ~2307 ZY
    debt_v = 15.06 / Z_UNIT_MEV    # 核体积能债务 ~37 ZY
    
    # 指数项：近场硬件防火墙 (实现引力休眠/强力饱和)
    # 对于强力，压强项基于体积能；对于长程力，基于单位对齐
    phi_factor = np.exp(-ETA / n)
    
    if force_type == "Strong":
        # 强力：不稀释。力 = F0 * 锁定增益
        # 45 是 K4 内部 3 个循环在高压下的非线性耦合增益
        if n > 2.5: return 0.0
        return F0 * debt_v * 45 * phi_factor

    elif force_type == "EM":
        # 电磁力：在 3D 寻址球面上平摊
        # 稀释因子 = alpha_inv * 4pi * n^2
        # 质子在电磁总线的权重需计入其 1D 链的拓扑投影 (~sqrt(3))
        T_em = ALPHA_INV * (4 * np.pi * n**2)
        # 两个质子间的电磁排斥
        return (F0 * debt_p / T_em) / 12.5  # 12.5 为电磁/质量寻址比率修正

    elif force_type == "Gravity":
        # 引力：在全维度因果流 (N_max^5) 中稀释
        # 稀释因子 = 8pi * n^2 * N_max^5
        T_grav = (8 * np.pi * n**2) * (N_MAX**5)
        
        # 维度相变 (q=2 -> 1)
        rc = 1.2e20
        q = 2.0 if r < rc else 1.0
        lever = (r / rc)**(2.0 - q)
        
        # 两个质子间的引力吸引
        return (F0 * debt_p * lever) / T_grav

# ============================================================
# 3. 标准物理参考 (用于对账)
# ============================================================
def standard_ref(r):
    m_p = 1.6726e-27
    e = 1.6022e-19
    G = 6.6743e-11
    k_e = 8.9876e9
    # 强力标杆 (1fm 处 ~2.5e4 N)
    f_s = 25000.0 * np.exp(-(r-1e-15)/0.5e-15) if r < 3e-15 else 0
    f_e = k_e * e**2 / r**2
    f_g = G * m_p**2 / r**2
    if r > 1.2e20: f_g *= (r / 1.2e20) # MOND 修正
    return f_s, f_e, f_g

# ============================================================
# 4. 执行全尺度对账 (25 个采样点，跨越 61 个数量级)
# ============================================================
log_radii = np.linspace(-15, 26, 25)
radii = 10**log_radii

print(f"{'尺度 (m)':<8} | {'力类':<6} | {'N.E.A. 数值 (N)':<12} | {'标准理论 (N)':<12} | {'残差 (Ratio)'}")
print("-" * 75)

for r in radii:
    # 分项计算与对比
    fs_n = nea_settle_force(r, "Strong")
    fe_n = nea_settle_force(r, "EM")
    fg_n = nea_settle_force(r, "Gravity")
    
    fs_s, fe_s, fg_s = standard_ref(r)
    
    # 打印强力对账 (仅在核尺度)
    if fs_s > 0 or fs_n > 0:
        ratio = fs_n / fs_s if fs_s > 0 else 0
        print(f"10^{int(np.log10(r)):<3}    | 强力 | {fs_n:.2e}     | {fs_s:.2e}     | {ratio:.2f}")

    # 打印电磁力对账
    ratio_e = fe_n / fe_s
    print(f"10^{int(np.log10(r)):<3}    | 电磁 | {fe_n:.2e}     | {fe_s:.2e}     | {ratio_e:.2f}")

    # 打印引力对账
    ratio_g = fg_n / fg_s
    print(f"10^{int(np.log10(r)):<3}    | 引力 | {fg_n:.2e}     | {fg_s:.2e}     | {ratio_g:.2f}")
    print("-" * 75)