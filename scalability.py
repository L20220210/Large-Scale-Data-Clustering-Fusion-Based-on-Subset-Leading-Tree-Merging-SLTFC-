"""
可扩展性实验脚本
──────────────────────────────────────────────────────
功能：在不同数据规模下运行 SLTFC，记录耗时，验证近似线性复杂度
运行：python scalability.py
输出：scalability_results.xlsx + scalability_plot.png
"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)
from scipy.optimize import linear_sum_assignment

# 导入你的算法和数据加载函数
from subset_leading_tree_fusion import SubsetLeadingTreeFusionClustering, load_excel_datasets

# ════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════
def cluster_accuracy(y_true, y_pred):
    """匈牙利算法计算聚类准确率"""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    def remap(arr):
        u = np.unique(arr)
        m = {v: i for i, v in enumerate(u)}
        return np.array([m[x] for x in arr])
    y_true = remap(y_true)
    y_pred = remap(y_pred)
    n = max(y_true.max(), y_pred.max()) + 1
    C = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1
    r, c = linear_sum_assignment(-C)
    return C[r, c].sum() / len(y_true)


def stratified_subsample(X, y, n_samples, random_state=42):
    """
    分层抽样：按类别比例从 (X, y) 中抽取 n_samples 个样本。
    保证每个类别至少有 1 个样本。
    """
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)

    # 计算每个类别应该抽多少
    n_per_class = np.maximum(
        np.round(counts / total * n_samples).astype(int), 1
    )

    # 微调使总数恰好等于 n_samples
    diff = n_samples - n_per_class.sum()
    if diff > 0:
        # 从大类里多抽几个
        order = np.argsort(-counts)
        for i in range(diff):
            n_per_class[order[i % len(order)]] += 1
    elif diff < 0:
        # 从大类里少抽几个
        order = np.argsort(-counts)
        for i in range(-diff):
            idx = order[i % len(order)]
            if n_per_class[idx] > 1:
                n_per_class[idx] -= 1

    # 执行抽样
    selected_indices = []
    for cls, n_cls in zip(classes, n_per_class):
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min(n_cls, len(cls_indices)), replace=False)
        selected_indices.append(chosen)

    selected = np.concatenate(selected_indices)
    rng.shuffle(selected)
    return X[selected], y[selected]


# ════════════════════════════════════════════════
# ===修改这里=== 数据路径和参数
# ════════════════════════════════════════════════

DATA_FILE = "KDD10.xlsx"     # ← 改成你的KDD10Per文件名

# 和之前实验用的最优参数一致
PARAMS = dict(
    n_subsets=300,
    keep_ratio=0.02,
    compression="structure",        # ← 改成你实际用的值
    # partition_method="density",
    # density_k=10,
    n_global_clusters=23,           # KDD10Per 有 23 个类
    # n_jobs=-1,
)

# 要测试的数据规模
SIZES = [10_000, 50_000, 100_000, 200_000, 300_000, None]
# None 表示全量数据（494,015）

N_REPEAT = 10   # 每个规模重复 3 次取平均


# ════════════════════════════════════════════════
# 加载数据
# ════════════════════════════════════════════════
print("正在加载数据...")
datasets = load_excel_datasets(DATA_FILE)
sheet_name = list(datasets.keys())[0]
X_full = datasets[sheet_name]["data"]
y_full = datasets[sheet_name]["target"]
X_full = StandardScaler().fit_transform(X_full)
print(f"数据加载完成：{X_full.shape[0]} 样本，{X_full.shape[1]} 维特征")


# ════════════════════════════════════════════════
# 主实验循环
# ════════════════════════════════════════════════
results = []

for size in SIZES:
    actual_size = X_full.shape[0] if size is None else size
    print(f"\n{'='*60}")
    print(f"数据规模：{actual_size:,}")
    print(f"{'='*60}")

    for seed in range(N_REPEAT):
        # 抽样（全量时直接用原数据）
        if size is None or size >= X_full.shape[0]:
            X_sub, y_sub = X_full, y_full
        else:
            X_sub, y_sub = stratified_subsample(X_full, y_full, size, random_state=seed)

        # 动态调整 n_subsets：保证每个子集至少有 100 个样本
        n_sub = min(PARAMS["n_subsets"], actual_size // 100)
        n_sub = max(n_sub, 5)   # 最少 5 个子集

        params = dict(PARAMS)
        params["n_subsets"] = n_sub
        params["random_state"] = seed

        print(f"  第 {seed+1}/{N_REPEAT} 次（n_subsets={n_sub}）...", end=" ", flush=True)

        # 运行并计时
        model = SubsetLeadingTreeFusionClustering(**params)
        t_start = time.time()
        model.fit(X_sub)
        t_total = time.time() - t_start

        # 取标签
        y_pred = model.final_labels_

        # 计算指标
        acc = cluster_accuracy(y_sub, y_pred)
        ari = adjusted_rand_score(y_sub, y_pred)
        nmi = normalized_mutual_info_score(y_sub, y_pred)
        ami = adjusted_mutual_info_score(y_sub, y_pred)

        print(f"耗时={t_total:.1f}s, ACC={acc:.4f}, ARI={ari:.4f}")

        results.append({
            "数据规模": actual_size,
            "n_subsets": n_sub,
            "重复次数": seed + 1,
            "总耗时(s)": round(t_total, 2),
            "ACC": round(acc, 4),
            "ARI": round(ari, 4),
            "NMI": round(nmi, 4),
            "AMI": round(ami, 4),
        })

# ════════════════════════════════════════════════
# 保存结果
# ════════════════════════════════════════════════
df = pd.DataFrame(results)
df.to_excel("scalability_results.xlsx", index=False)
print(f"\n✅ 结果已保存到 scalability_results.xlsx")

# ════════════════════════════════════════════════
# 汇总统计并打印
# ════════════════════════════════════════════════
summary = df.groupby("数据规模").agg(
    平均耗时=("总耗时(s)", "mean"),
    耗时标准差=("总耗时(s)", "std"),
    平均ACC=("ACC", "mean"),
    平均ARI=("ARI", "mean"),
    平均NMI=("NMI", "mean"),
).reset_index()

print("\n汇总结果：")
print(summary.to_string(index=False))

# ════════════════════════════════════════════════
# 画图
# ════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use("Agg")   # 无 GUI 环境也能画
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sizes_plot = summary["数据规模"].values
    times_plot = summary["平均耗时"].values
    stds_plot = summary["耗时标准差"].values

    # 左图：耗时 vs 数据规模
    axes[0].errorbar(sizes_plot / 1e4, times_plot, yerr=stds_plot,
                     marker="o", capsize=4, linewidth=2)
    axes[0].set_xlabel("Data size (×10⁴)", fontsize=12)
    axes[0].set_ylabel("Time (seconds)", fontsize=12)
    axes[0].set_title("SLTFC Scalability", fontsize=13)
    axes[0].grid(True, alpha=0.3)

    # 画一条理论线性参考线（从最小规模等比外推）
    if len(sizes_plot) > 1 and times_plot[0] > 0:
        linear_ref = times_plot[0] * (sizes_plot / sizes_plot[0])
        axes[0].plot(sizes_plot / 1e4, linear_ref, "--", color="gray",
                     alpha=0.5, label="Linear reference")
        axes[0].legend()

    # 右图：ARI 随规模变化（验证性能是否随规模下降）
    aris_plot = summary["平均ARI"].values
    axes[1].plot(sizes_plot / 1e4, aris_plot, "s-", color="green", linewidth=2)
    axes[1].set_xlabel("Data size (×10⁴)", fontsize=12)
    axes[1].set_ylabel("ARI", fontsize=12)
    axes[1].set_title("Clustering Quality vs Scale", fontsize=13)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("scalability_plot.png", dpi=200, bbox_inches="tight")
    print("\n✅ 图已保存到 scalability_plot.png")
except ImportError:
    print("\n⚠️ matplotlib 未安装，跳过画图。结果在 xlsx 文件中。")

print("\n全部完成！")