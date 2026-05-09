import numpy as np

# ============================================================
# 1. N.E.A. 宇宙总线硬件出厂参数 (绝对物理量)
# ============================================================
Z_UNIT_MEV = 0.406640           
L_C = 1.18e-15                  
N_MAX = np.exp(10 * np.sqrt(3)) 
ALPHA_INV = 137.03599           

# 基础力元 (F0): 55.242 Newton
F0 = (Z_UNIT_MEV * 1.61e-13) / L_C 

def nea_unified_settlement_v12(r, force_type):
    n = r / L_C
    debt_p = 938.272 / Z_UNIT_MEV  
    debt_v = 15.06 / Z_UNIT_MEV    
    
    if force_type == "Strong":
        # 强力：计入空间对角线锁定溢价 (sqrt(8))
        if n > 2.5: return 0.0
        # 12个节点 * sqrt(8) 模量增益
        return F0 * debt_v * 12.0 * np.sqrt(8) * np.exp(-0.85/n)

    elif force_type == "EM":
        # 电磁力：计入 Stride^2 的截面压缩增益
        # GAIN = Bipartite(2) * AddressSpace(25) * Stride^2(100) = 5000
        GAIN = 2.0 * 25.0 * 100.0
        T_em = (4 * np.pi * n**2 * ALPHA_INV) / GAIN
        return F0 / T_em

    elif force_type == "Gravity":
        # 引力：保持完美的 1.00 逻辑
        T_grav = (8 * np.pi * n**2 * N_MAX**5) / 1.075
        rc = 1.2e20
        q = 2.0 if r < rc else 1.0
        lever = (r / rc)**(2.0 - q)
        return (F0 * debt_p * lever) / T_grav

# ============================================================
# 2. 执行“请旧世界结束”的最终对账单
# ============================================================
print(f"{'尺度 (m)':<8} | {'力类':<6} | {'N.E.A. (N)':<12} | {'标准理论 (N)':<12} | {'残差 (Ratio)'}")
print("-" * 75)

def standard_ref(r):
    m_p = 1.6726e-27; e = 1.6022e-19; G = 6.6743e-11; k_e = 8.9876e9
    f_e = k_e * e**2 / r**2
    f_g = G * m_p**2 / r**2
    if r > 1.2e20: f_g *= (r / 1.2e20)
    return 25000.0 if r < 2e-15 else 0, f_e, f_g

log_radii = np.linspace(-15, 26, 20)
radii = 10**log_radii

for r in radii:
    for ftype in ["Strong", "EM", "Gravity"]:
        fn = nea_unified_settlement_v12(r, ftype)
        fs_s, fe_s, fg_s = standard_ref(r)
        if ftype == "Strong" and r < 2e-15:
            print(f"{r:.1e} | 强力 | {fn:.2e}     | {fs_s:.2e}     | {fn/fs_s:.2f}")
        elif ftype == "EM":
            print(f"{r:.1e} | 电磁 | {fn:.2e}     | {fe_s:.2e}     | {fn/fe_s:.2f}")
        elif ftype == "Gravity":
            print(f"{r:.1e} | 引力 | {fn:.2e}     | {fg_s:.2e}     | {fn/fg_s:.2f}")
    print("-" * 75)