"""
参数敏感性分析脚本
──────────────────────────────────────────────────────
运行前检查：
  1. 本脚本和 subset_v3.py 在同一文件夹
  2. 把下面"===修改这里==="部分替换成你自己的数据路径
  3. 运行：python sensitivity.py
  结果自动保存到 sensitivity_results.xlsx
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

# 导入你的算法
from subsetv3 import SubsetLeadingTreeFusionClustering


# ════════════════════════════════════════════════
# 工具函数：计算聚类准确率（匈牙利算法）
# ════════════════════════════════════════════════
def cluster_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    # 把标签映射到从0开始的连续整数
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


# ════════════════════════════════════════════════
# ===修改这里=== 加载你的数据
# ════════════════════════════════════════════════
# 直接从你的算法文件里导入原有的加载函数
from subset_leading_tree_fusion import load_excel_datasets

def load_data(name):
    """
    直接复用 subsetv3.py 里的 load_tabular_datasets，
    和之前实验用的加载方式完全一样。
    """
    if name == "KDD10Per":
        datasets = load_excel_datasets("KDD10.xlsx")

    elif name == "Pendigits":
        datasets = load_excel_datasets("Pendigits.xlsx")

    # load_tabular_datasets 返回字典，取第一个 sheet 的数据
    sheet_name = list(datasets.keys())[0]
    X = datasets[sheet_name]["data"]    # 已经是 float，NaN行已删除
    y = datasets[sheet_name]["target"]  # 已经是 int

    # 和原始实验一样做标准化
    X = StandardScaler().fit_transform(X)
    return X, y
# ════════════════════════════════════════════════
# 核心函数：跑单次实验，返回四个指标
# ════════════════════════════════════════════════
def run_once(X, y, params: dict):
    """
    params 是传给 SubsetLeadingTreeFusionClustering 的参数字典。
    返回：(ACC, ARI, NMI, AMI, 耗时秒)
    """
    model = SubsetLeadingTreeFusionClustering(**params)
    t0 = time.time()
    model.fit(X)
    elapsed = time.time() - t0

    y_pred = model.final_labels_
    acc = cluster_accuracy(y, y_pred)
    ari = adjusted_rand_score(y, y_pred)
    nmi = normalized_mutual_info_score(y, y_pred)
    ami = adjusted_mutual_info_score(y, y_pred)
    return acc, ari, nmi, ami, elapsed


# ════════════════════════════════════════════════
# 敏感性实验：改变一个参数，固定其他参数
# ════════════════════════════════════════════════
def sensitivity_experiment(
    dataset_name,     # 数据集名称（字符串）
    base_params,      # 固定参数的字典（最优参数）
    vary_param,       # 要改变的参数名（字符串）
    vary_values,      # 该参数的取值列表
    n_repeat=3,       # 每个值重复几次（取平均），3次够了
):
    """
    返回一个 DataFrame，每行是一个参数值的实验结果。
    """
    X, y = load_data(dataset_name)
    print(f"\n{'='*50}")
    print(f"数据集：{dataset_name}，改变参数：{vary_param}")
    print(f"取值范围：{vary_values}")
    print(f"{'='*50}")

    rows = []
    for val in vary_values:
        # 构造本次实验的参数：在 base_params 基础上替换 vary_param
        params = dict(base_params)   # 复制固定参数
        params[vary_param] = val     # 替换当前测试的参数

        accs, aris, nmis, amis, times = [], [], [], [], []
        for seed in range(n_repeat):
            params["random_state"] = seed
            print(f"  {vary_param}={val}, 第{seed+1}/{n_repeat}次...", end=" ")
            acc, ari, nmi, ami, t = run_once(X, y, params)
            accs.append(acc);  aris.append(ari)
            nmis.append(nmi);  amis.append(ami);  times.append(t)
            print(f"ACC={acc:.4f}, ARI={ari:.4f}, 耗时={t:.1f}s")

        rows.append({
            "数据集": dataset_name,
            "变化参数": vary_param,
            "参数值": val,
            "ACC均值": np.mean(accs),
            "ACC标准差": np.std(accs),
            "ARI均值": np.mean(aris),
            "ARI标准差": np.std(aris),
            "NMI均值": np.mean(nmis),
            "NMI标准差": np.std(nmis),
            "AMI均值": np.mean(amis),
            "AMI标准差": np.std(amis),
            "平均耗时(s)": np.mean(times),
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════
# ===修改这里=== 设置每个数据集的「最优固定参数」
# 就是之前对比实验中跑出最好结果用的参数
# ════════════════════════════════════════════════
BEST_PARAMS = {
    "KDD10Per": dict(
        n_subsets=300,
        keep_ratio=0.02,
        compression="structure",      # 改成完整单词，"str" 不被识别
        # partition_method="density",
        # density_k=10,
        n_global_clusters=23,         # ← 加上这行，直接告诉算法 K=23
        # n_jobs=-1,
    ),
    "Pendigits": dict(
        n_subsets=700,
        keep_ratio=0.012,
        compression="structure",
        partition_method="density",
        density_k=10,
        n_global_clusters=10,         # ← Pendigits 是 10 类
        n_jobs=-1,
    ),
}


# ════════════════════════════════════════════════
# 主程序：依次做三个参数的敏感性实验
# ════════════════════════════════════════════════
if __name__ == "__main__":

    all_results = []   # 收集所有结果

    # ─── 实验一：n_subsets（子集数 S）的敏感性 ───
    # 固定 keep_ratio 和 density_k，改变 n_subsets
    for ds in ["Pendigits"]:
        df = sensitivity_experiment(
            dataset_name=ds,
            base_params=BEST_PARAMS[ds],
            vary_param="n_subsets",
            vary_values=[50, 100, 200, 300, 500, 700, 1000],
            n_repeat=3,
        )
        all_results.append(df)

    # # ─── 实验二：keep_ratio（代表节点保留比 r）的敏感性 ───
    for ds in ["Pendigits"]:
        df = sensitivity_experiment(
            dataset_name=ds,
            base_params=BEST_PARAMS[ds],
            vary_param="keep_ratio",
            vary_values=[0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050],
            n_repeat=3,
        )
        all_results.append(df)

    # ─── 实验三：density_k（密度估计近邻数）的敏感性 ───
    # 仅对使用 density 策略的配置有意义
    for ds in ["Pendigits"]:
        df = sensitivity_experiment(
            dataset_name=ds,
            base_params=BEST_PARAMS[ds],
            vary_param="density_k",
            vary_values=[5, 10, 15, 20, 30, 50],
            n_repeat=3,
        )
        all_results.append(df)

    # ─── 保存所有结果到 Excel ───
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_excel("sensitivity_results.xlsx", index=False)
    print("\n✅ 所有实验完成，结果已保存到 sensitivity_results.xlsx")
    print(final_df.to_string())