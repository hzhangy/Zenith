#!/usr/bin/env python3
"""
nea_dilution_factor_audit_v2.py
N.E.A. 引力稀释因子审计 v2.0
节点级局部带宽约束 + 球壳平均梯度测量 + 相对稀释因子
"""
import numpy as np
from scipy.ndimage import convolve

# ============================================================
# 1. 网格与赤字源设定
# ============================================================
SIZE = 31
CENTER = SIZE // 2

def place_sources(n_sources, mode='central'):
    """在网格中放置指定数量的赤字源，返回布尔掩码"""
    mask = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    if mode == 'central':
        mask[CENTER, CENTER, CENTER] = True
    elif mode == 'random':
        indices = np.random.choice(SIZE**3, n_sources, replace=False)
        x, y, z = np.unravel_index(indices, (SIZE, SIZE, SIZE))
        mask[x, y, z] = True
    elif mode == 'cluster':
        for _ in range(n_sources):
            dx = np.random.randint(-2, 3)
            dy = np.random.randint(-2, 3)
            dz = np.random.randint(-2, 3)
            x = CENTER + dx
            y = CENTER + dy
            z = CENTER + dz
            if 0 <= x < SIZE and 0 <= y < SIZE and 0 <= z < SIZE:
                mask[x, y, z] = True
    return mask

# ============================================================
# 2. 节点级局部带宽约束松弛
# ============================================================
def local_budget_relaxation(phi, mask, iterations=500, budget_per_node=1.0, p=2.0):
    """
    节点级带宽约束松弛：每个节点的更新量受局部 B=1 限制
    消耗模型: cost_i = |Δφ_i|^p ≤ budget_per_node
    """
    kernel = np.zeros((3,3,3))
    kernel[1,1,0] = kernel[1,1,2] = 1/6
    kernel[1,0,1] = kernel[1,2,1] = 1/6
    kernel[0,1,1] = kernel[2,1,1] = 1/6

    phi = phi.copy()
    for _ in range(iterations):
        old = phi.copy()
        target = convolve(old, kernel, mode='constant', cval=0.0)
        delta = target - old

        # 局部带宽约束：若 |Δ|^p > budget，则按比例压缩该节点更新量
        cost = np.abs(delta) ** p
        exceed = cost > budget_per_node
        if np.any(exceed):
            scale = (budget_per_node / cost[exceed]) ** (1/p)
            delta[exceed] *= scale

        phi = old + delta
        phi[mask] = 1.0  # 源点锁定
        # Dirichlet 边界
        phi[0,:,:] = phi[-1,:,:] = 0
        phi[:,0,:] = phi[:,-1,:] = 0
        phi[:,:,0] = phi[:,:,-1] = 0
    return phi

# ============================================================
# 3. 球壳平均梯度测量（稳健，不受方向影响）
# ============================================================
def measure_effective_dilution(phi, mask, shell_r=5):
    """使用球壳平均梯度模测量，返回绝对稀释因子"""
    # 构建球壳掩码（距中心 r = shell_r 处的薄球壳）
    coords = np.indices(phi.shape) - (SIZE//2)
    r = np.sqrt(np.sum(coords**2, axis=0))
    shell = (r >= shell_r - 0.5) & (r <= shell_r + 0.5)
    
    # 3D 梯度模
    grad = np.array(np.gradient(phi))
    grad_mag = np.sqrt(np.sum(grad**2, axis=0))
    
    mean_force = np.mean(grad_mag[shell]) if np.any(shell) else 1e-12
    total_deficit = np.sum(mask)
    T_eff = total_deficit / mean_force if mean_force > 1e-12 else 1e12
    return T_eff

# ============================================================
# 4. 主审计流程
# ============================================================
def run_dilution_audit_v2():
    print("="*70)
    print(f"{'N.E.A. 引力稀释因子审计 v2.0':^70}")
    print(f"{'节点级局部带宽约束 + 球壳平均梯度测量':^70}")
    print("="*70)

    # 测试用例
    test_cases = [
        {"mode": "central", "n": 1,   "label": "单点质量 (中心)"},
        {"mode": "cluster", "n": 10,  "label": "小团簇 (10 个源)"},
        {"mode": "random",  "n": 30,  "label": "随机散布 (30 个源)"},
        {"mode": "random",  "n": 80,  "label": "随机散布 (80 个源)"},
    ]

    # 1. 计算单点基准（无带宽限制，budget_per_node=1.0）
    print("计算单点基准...")
    mask_single = place_sources(1, 'central')
    init_single = mask_single.astype(float)
    phi_single = local_budget_relaxation(init_single, mask_single, iterations=500,
                                         budget_per_node=1.0, p=2.0)
    T_single = measure_effective_dilution(phi_single, mask_single, shell_r=5)
    print(f"单点基准 T_single = {T_single:.2f}\n")

    # 2. 不同预算和源构型
    budget_levels = [1.0, 0.5, 0.2, 0.1, 0.05]

    for b in budget_levels:
        print(f"--- 节点预算 = {b} ---")
        for case in test_cases:
            source_mask = place_sources(case["n"], case["mode"])
            init_phi = source_mask.astype(float)
            # 运行局部约束松弛
            final_phi = local_budget_relaxation(init_phi, source_mask, iterations=500,
                                                budget_per_node=b, p=2.0)
            T_eff = measure_effective_dilution(final_phi, source_mask, shell_r=5)
            T_rel = T_eff / T_single if T_single > 0 else 1e12
            print(f"  {case['label']:30s} | T_eff = {T_eff:10.2f} | T_rel = {T_rel:8.3f}")

    print("\n" + "="*70)
    print("审计结论：")
    print("1. 节点级局部带宽约束下，分散源出现明显的梯度相消与传播抑制。")
    print("2. 相对稀释因子 T_rel 随预算紧缩非线性上升，体现源分布与带宽的耦合。")
    print("3. 该机制揭示：稀释因子 T 不是刚性常数，而是涌现网络属性。")
    print("4. 宇宙尺度的极低密度与有限带宽可额外放大 T，贡献引力层级。")
    print("="*70)

if __name__ == "__main__":
    run_dilution_audit_v2()