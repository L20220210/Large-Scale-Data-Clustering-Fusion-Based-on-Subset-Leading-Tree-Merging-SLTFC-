"""
子集引领树融合聚类算法 —— v3（空间感知划分版）
论文：《基于子集引领树合并的聚类融合》

v3 相对 v2 新增内容（其余完全保留）
─────────────────────────────────────
  新增参数  partition_method：控制子集划分策略
    "random"  ── 原始随机划分（基线，保持向后兼容）
    "spatial" ── KD 树递归空间划分（保证同一子集内的点在空间上相邻）
    "density" ── 密度感知 Voronoi 划分（在密度高的区域放更多子集种子，
                  使高密度区域获得更细粒度的局部结构建模）
  新增参数  density_k：density 策略中估计局部密度时使用的 KNN 近邻数
  新增方法  _partition_data：统一入口，按 partition_method 调度
  新增方法  _random_partition / _spatial_partition / _density_partition
  新增方法  plot_partition：可视化当前模型实际使用的子集划分结果
  新增函数  compare_partition_strategies：并排对比三种划分策略（适合论文插图）

v2 已有改进（全部保留）
──────────────────────
  1. O(n²) delta 计算 → 近似 O(n·k·log n)（小子集精确，大子集 KNN 近似）
  2. 子集树并行训练（joblib.Parallel）
  3. hidden_leaf 压缩逻辑独立分支，消除原版隐患
  4. 标签传播：top-k 加权投票（利用代表节点 weight 字段）
  5. 自动估计全局聚类数（gamma 拐点法）
"""

import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

import matplotlib.pyplot as plt
from matplotlib import font_manager
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# ══════════════════════════════════════════════
# 字体配置
# ══════════════════════════════════════════════
def configure_chinese_font():
    preferred = [
        "SimHei", "Microsoft YaHei", "Microsoft JhengHei",
        "NSimSun", "SimSun", "KaiTi", "FangSong",
        "Arial Unicode MS", "WenQuanYi Micro Hei", "Heiti TC",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["font.family"] = "sans-serif"
            break
    plt.rcParams["axes.unicode_minus"] = False


configure_chinese_font()


# ══════════════════════════════════════════════
# 数据类
# ══════════════════════════════════════════════
@dataclass
class RepresentativeNode:
    """从子集引领树中保留的压缩节点。"""
    original_index: int
    subset_id: int
    local_index: int
    local_label: int
    x: np.ndarray
    rho: float
    delta: float
    gamma: float
    weight: float = 1.0
    importance: float = 0.0


# ══════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════
def cluster_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """匈牙利算法最优映射下的聚类准确率。"""
    true_labels = _ensure_continuous(true_labels)
    pred_labels = _ensure_continuous(pred_labels)
    n = max(len(np.unique(true_labels)), len(np.unique(pred_labels)))
    C = np.zeros((n, n), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        C[t, p] += 1
    r, c = linear_sum_assignment(-C)
    return C[r, c].sum() / len(true_labels)


def _ensure_continuous(labels: np.ndarray) -> np.ndarray:
    unique = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique)}
    return np.array([mapping[x] for x in labels], dtype=int)


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    return np.zeros_like(values) if hi - lo < 1e-12 else (values - lo) / (hi - lo)


# ══════════════════════════════════════════════
# ▌ 新增：三种子集划分策略（v3 核心新增内容）
# ══════════════════════════════════════════════

def _random_partition(
    X: np.ndarray, n_subsets: int, rng: np.random.Generator
) -> List[np.ndarray]:
    """
    策略 1 ── 随机划分（原始基线）
    ─────────────────────────────
    把所有点随机打乱，再等份切割成 n_subsets 段。
    优点：实现简单，子集大小均匀。
    缺点：完全忽略空间结构，同一真实聚类的点可能被切到不同子集，
          导致局部引领树无法捕捉完整的密度峰值。
    """
    idx = rng.permutation(X.shape[0])
    return [s for s in np.array_split(idx, n_subsets) if len(s) > 0]


def _spatial_partition(
    X: np.ndarray, n_subsets: int, rng: np.random.Generator
) -> List[np.ndarray]:
    """
    策略 2 ── KD 树递归空间划分
    ─────────────────────────────
    思路
    ----
    每次选方差最大的维度，在中位数处将当前点集一分为二，递归直到产生 n_subsets 个子集。
    这相当于手动构建一棵 KD 树，每个叶节点就是一个子集。

    优点：
      · 同一子集内的点在特征空间中保证相邻 → 局部密度估计更准确
      · 子集大小接近均匀（每次在中位数处切割）
      · 完全不需要额外参数，纯几何操作

    缺点：
      · 高维数据中"方差最大维度"可能不稳定（维度诅咒），
        但因为每次都重新选，整体上仍然有效

    复杂度：O(N · log(N) · log(n_subsets))，非常快
    """
    all_indices = np.arange(X.shape[0])

    def _split(idx_array: np.ndarray, n_parts: int) -> List[np.ndarray]:
        # 递归终止条件：只需要1份，或者点数不足再分
        if n_parts <= 1 or len(idx_array) == 0:
            return [idx_array] if len(idx_array) > 0 else []

        sub_X = X[idx_array]

        # 找方差最大的维度（在该维度上切割最能分开点）
        best_dim = int(np.argmax(sub_X.var(axis=0)))
        median_val = np.median(sub_X[:, best_dim])

        left_mask  = sub_X[:, best_dim] <= median_val
        right_mask = ~left_mask

        # 边界情况：所有点在该维度上值完全相同 → 强制按索引对半分
        if left_mask.all() or right_mask.all():
            mid = len(idx_array) // 2
            left_idx  = idx_array[:mid]
            right_idx = idx_array[mid:]
        else:
            left_idx  = idx_array[left_mask]
            right_idx = idx_array[right_mask]

        # 左右各分配一半的子集数量
        n_left  = n_parts // 2
        n_right = n_parts - n_left

        return _split(left_idx, n_left) + _split(right_idx, n_right)

    return _split(all_indices, n_subsets)


def _density_partition(
    X: np.ndarray, n_subsets: int, rng: np.random.Generator, density_k: int = 10
) -> List[np.ndarray]:
    """
    策略 3 ── 密度感知 Voronoi 划分（v3 新创新点）
    ─────────────────────────────────────────────
    核心思想
    --------
    随机划分把每个点被选为"种子"的概率视为相等。
    但在聚类任务中，密度高的区域（真实簇的核心）结构更复杂，
    更需要精细建模 → 应该分配更多子集（更细的粒度）。
    密度低的稀疏区域结构简单，少量子集即可覆盖。

    算法步骤
    --------
    Step 1. 用 KNN 估计每个点的局部密度：
              ρ(i) = 1 / (第 k 近邻距离 + ε)
            密度越高 → ρ 越大

    Step 2. 以 ρ 为概率权重，从 N 个点中无放回地采样 n_subsets 个"种子"。
              密度高的区域被选为种子的概率更高
              → 同等面积内高密度区得到更多种子（更细划分）

    Step 3. 对每个点，找其最近的种子，归属到该种子所在的子集。
              这形成了一张"密度加权 Voronoi 图"。

    优点：
      · 高密度区（真实簇核心）获得更细粒度的子集 → 局部树更准确
      · 低密度区（噪声/稀疏）合并为大子集 → 减少冗余计算
      · 子集边界大致沿密度梯度走，不会将密度峰值劈开

    缺点：
      · 多一步密度预估（O(N·log N)），但开销很小
      · 子集大小不均匀（密度高的地方子集小，密度低的地方子集大）

    参数
    ----
    density_k : 估计局部密度时使用的 KNN 近邻数（默认 10）
    """
    n = X.shape[0]
    n_seeds = min(n_subsets, n)   # 种子数不超过总点数

    # ── Step 1：KNN 估计局部密度 ──────────────
    k_eff = min(density_k + 1, n)
    nbrs = NearestNeighbors(n_neighbors=k_eff).fit(X)
    distances, _ = nbrs.kneighbors(X)
    # 用第 k 近邻距离的倒数作为密度；加 ε 防止除零
    density = 1.0 / (distances[:, -1] + 1e-8)

    # ── Step 2：密度加权无放回采样种子 ────────
    probs = density / density.sum()          # 归一化为概率
    seed_indices = rng.choice(n, size=n_seeds, replace=False, p=probs)
    seed_X = X[seed_indices]

    # ── Step 3：Voronoi 划分 ──────────────────
    # 每个点找最近种子 → 归属到该种子的子集
    nbrs_seed = NearestNeighbors(n_neighbors=1).fit(seed_X)
    _, assignments = nbrs_seed.kneighbors(X)
    assignments = assignments[:, 0]          # shape (N,)，值域 [0, n_seeds)

    # 按 assignment 分组，过滤空子集
    splits = []
    for s in range(n_seeds):
        idx = np.where(assignments == s)[0]
        if len(idx) > 0:
            splits.append(idx)

    return splits


# ══════════════════════════════════════════════
# 引领树（LeadingTree，与 v2 完全相同）
# ══════════════════════════════════════════════
class LeadingTree:
    """密度峰值风格的引领树聚类（v2 版本，未改动）。"""

    APPROX_THRESHOLD = 300

    def __init__(
        self,
        dc: Optional[float] = None,
        rho_threshold: Optional[float] = None,
        delta_threshold: Optional[float] = None,
        min_centers: int = 1,
        knn_k: int = 20,
    ):
        self.dc = dc
        self.rho_threshold = rho_threshold
        self.delta_threshold = delta_threshold
        self.min_centers = min_centers
        self.knn_k = knn_k
        self.rho = self.delta = self.gamma = None
        self.parent_ = self.centers = self.labels_ = None
        self.tree = self.data = None

    def fit(self, X: np.ndarray, metric: str = "euclidean") -> "LeadingTree":
        self.data = np.asarray(X)
        self.rho = self._compute_local_density(self.data, metric=metric)
        self.delta, self.parent_ = self._compute_delta_and_parent(
            self.data, self.rho, metric=metric
        )
        self.gamma = self.rho * self.delta
        self.centers = self._identify_centers(self.rho, self.delta, self.gamma)
        self.tree = self._build_tree(self.data, self.parent_, self.centers)
        self.labels_ = self._assign_labels(self.tree, self.centers)
        return self

    def _compute_local_density(self, X, metric="euclidean", k=10):
        n = X.shape[0]
        if n <= 1:
            return np.ones(n)
        if self.dc is None:
            k_eff = min(k + 1, n)
            nbrs = NearestNeighbors(n_neighbors=k_eff, metric=metric).fit(X)
            distances, _ = nbrs.kneighbors(X)
            return 1.0 / (distances[:, -1] + 1e-8)
        distances = cdist(X, X, metric=metric)
        return np.sum(np.exp(-((distances / self.dc) ** 2)), axis=1) - 1.0

    def _compute_delta_and_parent(self, X, rho, metric="euclidean"):
        n = X.shape[0]
        if n <= 1:
            return np.zeros(n), np.arange(n)
        if n <= self.APPROX_THRESHOLD:
            return self._delta_exact(X, rho, metric)
        return self._delta_approx(X, rho, metric)

    def _delta_exact(self, X, rho, metric):
        n = X.shape[0]
        delta, parent = np.zeros(n), np.arange(n)
        distances = cdist(X, X, metric=metric)
        order = np.argsort(-rho)
        for rank, idx in enumerate(order):
            if rank == 0:
                delta[idx] = np.max(distances[idx])
                parent[idx] = idx
                continue
            higher = order[:rank]
            d = distances[idx, higher]
            best = np.argmin(d)
            delta[idx] = d[best]
            parent[idx] = higher[best]
        return delta, parent

    def _delta_approx(self, X, rho, metric):
        n = X.shape[0]
        delta, parent = np.zeros(n), np.arange(n)
        order = np.argsort(-rho)
        rho_rank = np.empty(n, dtype=int)
        rho_rank[order] = np.arange(n)

        k_eff = min(self.knn_k + 1, n)
        nbrs = NearestNeighbors(n_neighbors=k_eff, metric=metric).fit(X)
        knn_d, knn_i = nbrs.kneighbors(X)
        knn_d, knn_i = knn_d[:, 1:], knn_i[:, 1:]

        top_idx = order[0]
        delta[top_idx] = np.max(cdist(X[top_idx:top_idx+1], X, metric=metric))
        parent[top_idx] = top_idx

        fallback = []
        for idx in order[1:]:
            nbr, d = knn_i[idx], knn_d[idx]
            mask = rho_rank[nbr] < rho_rank[idx]
            if mask.any():
                valid_d, valid_n = d[mask], nbr[mask]
                best = np.argmin(valid_d)
                delta[idx], parent[idx] = valid_d[best], valid_n[best]
            else:
                fallback.append(idx)

        for idx in fallback:
            hi_mask = rho > rho[idx]
            if not hi_mask.any():
                delta[idx] = np.max(cdist(X[idx:idx+1], X, metric=metric))
                parent[idx] = idx
            else:
                hi_X = X[hi_mask]
                hi_i = np.where(hi_mask)[0]
                d = cdist(X[idx:idx+1], hi_X, metric=metric)[0]
                best = np.argmin(d)
                delta[idx], parent[idx] = d[best], hi_i[best]

        return delta, parent

    def _identify_centers(self, rho, delta, gamma):
        if len(rho) == 0:
            return np.array([], dtype=int)
        rho_thr = self.rho_threshold if self.rho_threshold is not None else np.percentile(rho, 70)
        delta_thr = self.delta_threshold if self.delta_threshold is not None else np.percentile(delta, 70)
        centers = np.where((rho >= rho_thr) & (delta >= delta_thr))[0]
        if len(centers) < self.min_centers:
            centers = np.argsort(-gamma)[:min(self.min_centers, len(gamma))]
        return np.asarray(sorted(set(centers.tolist())), dtype=int)

    def _build_tree(self, X, parent, centers):
        tree = nx.DiGraph()
        center_set = set(centers.tolist())
        for i in range(X.shape[0]):
            tree.add_node(i, pos=X[i],
                          rho=float(self.rho[i]),
                          delta=float(self.delta[i]),
                          gamma=float(self.gamma[i]))
        for i, p in enumerate(parent):
            if i not in center_set and i != p:
                tree.add_edge(int(p), int(i))
        return tree

    def _assign_labels(self, tree, centers):
        labels = np.full(tree.number_of_nodes(), -1, dtype=int)
        for label, center in enumerate(centers):
            labels[center] = label
            for node in nx.descendants(tree, center):
                labels[node] = label
        labels[labels < 0] = 0
        return labels


# ══════════════════════════════════════════════
# 图卷积平滑器（与 v2 完全相同）
# ══════════════════════════════════════════════
class GraphSmoother:
    def __init__(self, n_layers: int = 2, alpha: float = 0.6):
        self.n_layers = n_layers
        self.alpha = alpha

    def transform(self, tree: nx.DiGraph) -> Tuple[List[int], np.ndarray]:
        nodes = list(tree.nodes)
        if not nodes:
            return nodes, np.empty((0, 3))
        features = np.array(
            [[tree.nodes[n]["rho"], tree.nodes[n]["delta"], tree.nodes[n]["gamma"]]
             for n in nodes], dtype=float
        )
        features = (features - features.mean(0)) / (features.std(0) + 1e-8)
        adj = nx.to_numpy_array(tree.to_undirected(), nodelist=nodes) + np.eye(len(nodes))
        norm_adj = adj / (adj.sum(1)[:, None] + 1e-8)
        h = features
        for _ in range(self.n_layers):
            h = self.alpha * h + (1 - self.alpha) * norm_adj @ h
        return nodes, h


# ══════════════════════════════════════════════
# 主算法（v3）
# ══════════════════════════════════════════════
class SubsetLeadingTreeFusionClustering:
    """
    基于子集引领树合并的聚类融合 v3。

    新增参数（相对 v2）
    ──────────────────
    partition_method : str
        子集划分策略，三选一：
        · "random"  ── 随机划分（原始基线，默认兼容 v2）
        · "spatial" ── KD 树递归空间划分（空间相邻性保证）
        · "density" ── 密度感知 Voronoi 划分（推荐用于高密度变化数据集）

    density_k : int
        仅当 partition_method="density" 时生效。
        估计局部密度时的 KNN 近邻数，默认 10。
        数据规模大时可适当增大（如 20）以得到更稳定的密度估计。
    """

    def __init__(
        self,
        n_subsets: int = 10,
        dc: Optional[float] = None,
        rho_threshold: Optional[float] = None,
        delta_threshold: Optional[float] = None,
        global_dc: Optional[float] = None,
        global_rho_threshold: Optional[float] = None,
        global_delta_threshold: Optional[float] = None,
        compression: str = "hidden_leaf",
        keep_ratio: float = 0.35,
        min_representatives: int = 5,
        structure_weights: Tuple[float, float, float, float] = (0.45, 0.20, 0.25, 0.10),
        boundary_k: int = 10,
        n_global_clusters: Optional[int] = None,
        n_jobs: int = -1,
        partition_method: str = "random",   # ← v3 新增
        density_k: int = 10,                # ← v3 新增
        random_state: int = 42,
    ):
        self.n_subsets = n_subsets
        self.dc = dc
        self.rho_threshold = rho_threshold
        self.delta_threshold = delta_threshold
        self.global_dc = global_dc
        self.global_rho_threshold = global_rho_threshold
        self.global_delta_threshold = global_delta_threshold
        self.compression = compression
        self.keep_ratio = keep_ratio
        self.min_representatives = min_representatives
        self.structure_weights = structure_weights
        self.boundary_k = boundary_k
        self.n_global_clusters = n_global_clusters
        self.n_jobs = n_jobs
        self.partition_method = partition_method
        self.density_k = density_k
        self.random_state = random_state

        # 训练后填充
        self.subset_indices_: List[np.ndarray] = []
        self.subset_models_: List[LeadingTree] = []
        self.representatives_: List[RepresentativeNode] = []
        self.global_model_: Optional[LeadingTree] = None
        self.final_labels_: Optional[np.ndarray] = None
        self.X_: Optional[np.ndarray] = None
        self._partition_assignment_: Optional[np.ndarray] = None  # 用于可视化

    # ── 主入口 ───────────────────────────────
    def fit(self, X: np.ndarray) -> "SubsetLeadingTreeFusionClustering":
        self.X_ = np.asarray(X)
        self._fit_subset_trees(self.X_)
        self._compress_subset_trees()
        self._fuse_representatives()
        self._propagate_labels()
        self._relabel_continuously()
        return self

    # ══════════════════════════════════════════
    # ▌ v3 核心新增：步骤 1 使用可选划分策略
    # ══════════════════════════════════════════
    def _partition_data(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> List[np.ndarray]:
        """
        统一划分入口，根据 partition_method 调用对应的划分函数。

        返回值：List[np.ndarray]，每个元素是一个子集的原始索引数组。
        """
        method = self.partition_method.lower()

        if method == "spatial":
            print(f"  [划分策略] KD 树递归空间划分")
            splits = _spatial_partition(X, self.n_subsets, rng)

        elif method == "density":
            print(f"  [划分策略] 密度感知 Voronoi 划分（density_k={self.density_k}）")
            splits = _density_partition(X, self.n_subsets, rng, density_k=self.density_k)

        else:   # "random" 或其他未知值均退化为随机
            if method != "random":
                print(f"  [警告] 未知划分策略 '{self.partition_method}'，退化为随机划分")
            else:
                print(f"  [划分策略] 随机划分")
            splits = _random_partition(X, self.n_subsets, rng)

        # 统计并打印划分信息
        sizes = [len(s) for s in splits]
        print(
            f"  [划分结果] 实际子集数={len(splits)}，"
            f"子集大小：min={min(sizes)}，max={max(sizes)}，"
            f"均值={np.mean(sizes):.1f}，标准差={np.std(sizes):.1f}"
        )

        # 保存每个点的子集编号（用于可视化）
        assignment = np.zeros(X.shape[0], dtype=int)
        for sid, idx in enumerate(splits):
            assignment[idx] = sid
        self._partition_assignment_ = assignment

        return splits

    def _fit_subset_trees(self, X: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)

        # ← v3：调用统一划分入口，替换原来的 rng.permutation + array_split
        splits = self._partition_data(X, rng)

        def _train_one(subset_indices: np.ndarray) -> LeadingTree:
            return LeadingTree(
                dc=self.dc,
                rho_threshold=self.rho_threshold,
                delta_threshold=self.delta_threshold,
                min_centers=1,
            ).fit(X[subset_indices])

        models = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_train_one)(idx) for idx in splits
        )

        self.subset_indices_ = splits
        self.subset_models_ = models

    # ── 步骤 2：压缩子集树 ───────────────────
    def _compress_subset_trees(self) -> None:
        self.representatives_ = []
        for subset_id, (indices, model) in enumerate(
            zip(self.subset_indices_, self.subset_models_)
        ):
            kept = self._select_representative_nodes(model)
            for local_idx in kept:
                self.representatives_.append(
                    RepresentativeNode(
                        original_index=int(indices[local_idx]),
                        subset_id=subset_id,
                        local_index=int(local_idx),
                        local_label=int(model.labels_[local_idx]),
                        x=self.X_[indices[local_idx]].copy(),
                        rho=float(model.rho[local_idx]),
                        delta=float(model.delta[local_idx]),
                        gamma=float(model.gamma[local_idx]),
                        weight=float(self._subtree_size(model.tree, local_idx)),
                        importance=float(
                            getattr(model, "representative_scores_", {}).get(
                                int(local_idx), 0.0
                            )
                        ),
                    )
                )

    def _select_representative_nodes(self, model: LeadingTree) -> np.ndarray:
        n_nodes = model.tree.number_of_nodes()
        if n_nodes <= self.min_representatives:
            return np.arange(n_nodes, dtype=int)

        keep_count = min(
            max(self.min_representatives, int(np.ceil(n_nodes * self.keep_ratio))),
            n_nodes,
        )
        center_nodes = set(model.centers.tolist())
        branch_nodes = {n for n in model.tree.nodes if model.tree.out_degree(n) >= 2}

        if self.compression == "structure":
            scores = self._compute_structural_scores(model)
            model.representative_scores_ = {int(i): float(scores[i]) for i in range(n_nodes)}
            ranked = list(np.argsort(-scores).astype(int))
            selected = list(center_nodes)

        elif self.compression == "conv":
            nodes, hidden = GraphSmoother().transform(model.tree)
            s = np.linalg.norm(hidden, axis=1)
            model.representative_scores_ = {int(n): float(v) for n, v in zip(nodes, s)}
            ranked = [nodes[i] for i in np.argsort(-s)]
            selected = list(center_nodes | branch_nodes)

        elif self.compression == "hidden_leaf":
            gamma_rank = list(np.argsort(-model.gamma).astype(int))
            model.representative_scores_ = {i: float(model.gamma[i]) for i in range(n_nodes)}
            non_leaf = [n for n in gamma_rank
                        if model.tree.out_degree(n) > 0 or n in center_nodes]
            leaf_only = [n for n in gamma_rank if n not in non_leaf]
            ranked = non_leaf + leaf_only
            selected = list(center_nodes | branch_nodes)

        else:   # 默认：纯 gamma 排名
            gamma_rank = list(np.argsort(-model.gamma).astype(int))
            model.representative_scores_ = {i: float(model.gamma[i]) for i in range(n_nodes)}
            ranked = gamma_rank
            selected = list(center_nodes | branch_nodes)

        selected_set = set(selected)
        for node in ranked:
            if node not in selected_set:
                selected.append(node)
                selected_set.add(node)
            if len(selected) >= keep_count:
                break

        return np.array(selected[:keep_count], dtype=int)

    def _compute_structural_scores(self, model: LeadingTree) -> np.ndarray:
        alpha, beta, gamma_w, eta = self.structure_weights
        tree = model.tree
        n = tree.number_of_nodes()
        scores = (
            alpha * _minmax(model.gamma)
            + beta  * _minmax(np.array([tree.out_degree(i) for i in range(n)], dtype=float))
            + gamma_w * _minmax(self._all_subtree_sizes(tree, n))
            + eta  * self._boundary_scores(model)
        )
        for c in model.centers:
            scores[int(c)] += 1.0
        return scores

    def _all_subtree_sizes(self, tree: nx.DiGraph, n: int) -> np.ndarray:
        sizes = np.ones(n, dtype=float)
        for node in reversed(list(nx.topological_sort(tree))):
            for child in tree.successors(node):
                sizes[node] += sizes[child]
        return sizes

    def _boundary_scores(self, model: LeadingTree) -> np.ndarray:
        n = model.data.shape[0]
        if n <= 2 or len(np.unique(model.labels_)) <= 1:
            return np.zeros(n, dtype=float)
        k_eff = min(self.boundary_k + 1, n)
        nbrs = NearestNeighbors(n_neighbors=k_eff).fit(model.data)
        _, ni = nbrs.kneighbors(model.data)
        ni = ni[:, 1:]
        return np.array(
            [np.mean(model.labels_[ni[i]] != model.labels_[i]) for i in range(n)],
            dtype=float,
        )

    def _subtree_size(self, tree: nx.DiGraph, node: int) -> int:
        return 1 + len(nx.descendants(tree, node)) if node in tree else 1

    # ── 步骤 3：融合 ─────────────────────────
    def _fuse_representatives(self) -> None:
        if not self.representatives_:
            raise ValueError("无代表节点，请检查 keep_ratio 和 min_representatives 参数。")
        rep_X = np.vstack([r.x for r in self.representatives_])
        n_cls = self.n_global_clusters or self._estimate_n_clusters(rep_X)

        self.global_model_ = LeadingTree(
            dc=self.global_dc if self.global_dc is not None else self.dc,
            rho_threshold=self.global_rho_threshold,
            delta_threshold=self.global_delta_threshold,
            min_centers=n_cls,
        ).fit(rep_X)

        top_centers = np.argsort(-self.global_model_.gamma)[:n_cls]
        self.global_model_.centers = np.asarray(top_centers, dtype=int)
        self.global_model_.tree = self.global_model_._build_tree(
            rep_X, self.global_model_.parent_, self.global_model_.centers
        )
        self.global_model_.labels_ = self.global_model_._assign_labels(
            self.global_model_.tree, self.global_model_.centers
        )

    def _estimate_n_clusters(self, rep_X: np.ndarray) -> int:
        tmp = LeadingTree(dc=self.dc, min_centers=1).fit(rep_X)
        sg = np.sort(tmp.gamma)[::-1]
        if len(sg) < 3:
            return 1
        elbow = int(np.argmin(np.diff(sg))) + 1
        n_est = max(1, min(elbow, 20))
        print(f"  [自动估计聚类数] gamma 拐点法 k = {n_est}")
        return n_est

    # ── 步骤 4：标签传播（top-k 加权投票）──
    def _propagate_labels(self) -> None:
        rep_X = np.vstack([r.x for r in self.representatives_])
        rep_labels = np.array(self.global_model_.labels_, dtype=int)
        rep_weights = np.array([r.weight for r in self.representatives_], dtype=float)

        top_k = min(5, len(self.representatives_))
        nbrs = NearestNeighbors(n_neighbors=top_k).fit(rep_X)
        dists, nbr_idx = nbrs.kneighbors(self.X_)
        dists = np.clip(dists, 1e-10, None)

        n_cls = int(rep_labels.max()) + 1
        final_labels = np.zeros(len(self.X_), dtype=int)
        for i in range(len(self.X_)):
            vote = np.zeros(n_cls)
            for j, lbl in enumerate(rep_labels[nbr_idx[i]]):
                vote[lbl] += rep_weights[nbr_idx[i, j]] / dists[i, j]
            final_labels[i] = int(np.argmax(vote))
        self.final_labels_ = final_labels

    def _relabel_continuously(self) -> None:
        unique = np.unique(self.final_labels_)
        mapping = {old: new for new, old in enumerate(unique)}
        self.final_labels_ = np.array([mapping[l] for l in self.final_labels_], dtype=int)

    # ── 评估 ─────────────────────────────────
    def evaluate(self, true_labels: np.ndarray) -> Dict:
        return {
            "准确率 (ACC)": cluster_accuracy(true_labels, self.final_labels_),
            "调整兰德指数 (ARI)": adjusted_rand_score(true_labels, self.final_labels_),
            "标准化互信息 (NMI)": normalized_mutual_info_score(true_labels, self.final_labels_),
            "调整互信息 (AMI)": adjusted_mutual_info_score(true_labels, self.final_labels_),
            "代表节点数": len(self.representatives_),
            "最终簇数": len(np.unique(self.final_labels_)),
        }

    # ══════════════════════════════════════════
    # ▌ v3 新增：划分可视化
    # ══════════════════════════════════════════
    def plot_partition(
        self,
        X: np.ndarray,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        可视化本模型实际使用的子集划分结果。
        每个子集用不同颜色表示，可直观看出划分是否保持了空间连贯性。

        参数
        ----
        X         : 原始特征矩阵（2D）或高维数据（自动 PCA 降维到 2D）
        title     : 图标题（None 则自动生成）
        save_path : 保存路径（None 则直接显示）
        """
        if self._partition_assignment_ is None:
            raise ValueError("请先调用 fit() 再调用 plot_partition()。")

        X = np.asarray(X)
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            X = PCA(n_components=2).fit_transform(X)

        n_colors = len(self.subset_indices_)
        cmap = plt.cm.get_cmap("tab20", n_colors)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            X[:, 0], X[:, 1],
            c=self._partition_assignment_,
            cmap=cmap,
            s=15, alpha=0.7,
        )
        title = title or f"子集划分可视化（策略：{self.partition_method}，子集数：{n_colors}）"
        plt.title(title)
        plt.xlabel("X坐标")
        plt.ylabel("Y坐标")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # ── 其他可视化（与 v2 相同）─────────────
    def plot_clusters(self, X, title="子集引领树融合聚类结果", save_path=None):
        if self.final_labels_ is None:
            raise ValueError("请先调用 fit()。")
        X = np.asarray(X)
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=self.final_labels_, cmap="tab10", s=35, alpha=0.85)
        rep_X = np.vstack([r.x for r in self.representatives_])
        plt.scatter(rep_X[:, 0], rep_X[:, 1], c="black", s=18, marker="x", label="代表节点")
        plt.title(title)
        plt.xlabel("X坐标"); plt.ylabel("Y坐标")
        plt.grid(True, alpha=0.25); plt.legend(); plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close()
        else:
            plt.show()

    def plot_comparison(self, X, true_labels, title="真实标签与聚类结果对比", save_path=None):
        if self.final_labels_ is None:
            raise ValueError("请先调用 fit()。")
        X = np.asarray(X); true_labels = np.asarray(true_labels)
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            r = PCA(n_components=2); X = r.fit_transform(X)
            xl = f"主成分1 ({r.explained_variance_ratio_[0]:.2%})"
            yl = f"主成分2 ({r.explained_variance_ratio_[1]:.2%})"
            title += " (PCA降维)"
        else:
            xl, yl = "X坐标", "Y坐标"
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
        for ax, lbl, sub in zip(axes, [true_labels, self.final_labels_], ["真实标签", "预测标签"]):
            ax.scatter(X[:, 0], X[:, 1], c=lbl, cmap="tab10", s=35, alpha=0.85)
            ax.set_title(sub); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=0.25)
        fig.suptitle(title, fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close()
        else:
            plt.show()


# ══════════════════════════════════════════════
# ▌ v3 新增：三种划分策略并排对比（论文插图）
# ══════════════════════════════════════════════
def compare_partition_strategies(
    X: np.ndarray,
    n_subsets: int = 20,
    random_state: int = 42,
    save_path: Optional[str] = None,
) -> None:
    """
    在同一份数据上并排展示三种划分策略的结果，适合直接放入论文。

    参数
    ----
    X           : 2D 特征矩阵（高维请先降维）
    n_subsets   : 子集数量（建议取较小值如 10~30，方便可视化）
    random_state: 随机种子
    save_path   : 保存路径（None 则直接显示）

    使用示例
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=1000, centers=5, random_state=0)
    >>> compare_partition_strategies(X, n_subsets=15, save_path="partition_compare.png")
    """
    rng = np.random.default_rng(random_state)

    strategies = [
        ("random",  "随机划分\n（基线）",       _random_partition(X, n_subsets, rng)),
        ("spatial", "KD 树空间划分\n", _spatial_partition(X, n_subsets, rng)),
        ("density", "密度感知划分\n",  _density_partition(X, n_subsets, rng)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    cmap = plt.cm.get_cmap("tab20", n_subsets)

    for ax, (method, label, splits) in zip(axes, strategies):
        # 给每个点着色：颜色 = 子集编号
        assignment = np.zeros(X.shape[0], dtype=int)
        for sid, idx in enumerate(splits):
            assignment[idx] = sid

        sizes = [len(s) for s in splits]

        ax.scatter(X[:, 0], X[:, 1], c=assignment, cmap=cmap, s=12, alpha=0.75)
        ax.set_title(
            f"{label}\n子集数={len(splits)}  "
            f"大小 min={min(sizes)} / max={max(sizes)} / std={np.std(sizes):.1f}",
            fontsize=11,
        )
        ax.set_xlabel("X坐标"); ax.set_ylabel("Y坐标")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"三种子集划分策略对比（n_subsets={n_subsets}）", fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ══════════════════════════════════════════════
# 实验入口（与 v2 相同，新增 partition_method）
# ══════════════════════════════════════════════
def load_excel_datasets(file_path: str) -> Dict:
    datasets = {}
    excel_file = pd.ExcelFile(file_path)
    for sheet in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)
        if df.shape[1] < 3:
            print(f"跳过 {sheet}: 至少需要两列特征和一列标签")
            continue
        X = df.iloc[:, :-1].to_numpy(dtype=float)
        y = df.iloc[:, -1].to_numpy()
        mask = ~(np.any(pd.isna(X), axis=1) | pd.isna(y))
        X, y = X[mask], y[mask].astype(int)
        datasets[sheet] = {"data": X, "target": y}
        print(f"{sheet}: X={X.shape}, 类别数={len(np.unique(y))}")
    return datasets

def load_csv_dataset(file_path):
    df = pd.read_csv(file_path)
    if df.shape[1] < 3:
        raise ValueError("CSV至少需要两列特征和一列标签")

    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy()
    valid_mask = ~(np.any(pd.isna(X), axis=1) | pd.isna(y))
    X = X[valid_mask]
    y = y[valid_mask].astype(int)

    name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"{name}: X={X.shape}, 类别数={len(np.unique(y))}")
    return {name: {"data": X, "target": y}}


def load_tabular_datasets(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        return load_excel_datasets(file_path)
    if ext == ".csv":
        return load_csv_dataset(file_path)
    raise ValueError(f"不支持的文件类型: {ext}")


def print_results_table(results: Dict) -> None:
    if not results:
        print("没有有效结果"); return
    print("\n" + "=" * 96)
    print("子集引领树融合聚类算法性能总结 v3")
    print("=" * 96)
    print(f"{'数据集':<12}{'ACC':>10}{'ARI':>10}{'NMI':>10}"
          f"{'AMI':>10}{'代表节点数':>12}{'最终簇数':>10}{'耗时(秒)':>12}")
    print("-" * 96)
    for name, m in results.items():
        print(f"{name:<12}{m['准确率 (ACC)']:>10.4f}{m['调整兰德指数 (ARI)']:>10.4f}"
              f"{m['标准化互信息 (NMI)']:>10.4f}{m['调整互信息 (AMI)']:>10.4f}"
              f"{int(m['代表节点数']):>12}{int(m['最终簇数']):>10}{m['耗时(秒)']:>12.4f}")
    print("=" * 96)


def run_excel_experiment(
    file_path: str = "数据集.xlsx",
    n_subsets: int = 500,
    compression: str = "hidden_leaf",
    keep_ratio: float = 0.01,
    min_representatives: int = 15,
    structure_weights: Tuple = (0.45, 0.25, 0.2, 0.1),
    boundary_k: int = 10,
    n_global_clusters: Optional[int] = 7,
    n_jobs: int = -1,
    partition_method: str = "density",   # ← v3 新增，推荐 density
    density_k: int = 10,                 # ← v3 新增
    output_dir: str = "results_v3",
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    datasets = load_tabular_datasets(file_path)
    results = {}

    for name, dataset in datasets.items():
        print(f"\n{'='*50}\n开始处理数据集: {name}")
        start = time.time()
        X, y = dataset["data"], dataset["target"]
        X_scaled = StandardScaler().fit_transform(X)

        model = SubsetLeadingTreeFusionClustering(
            n_subsets=n_subsets,
            compression=compression,
            keep_ratio=keep_ratio,
            min_representatives=min_representatives,
            structure_weights=structure_weights,
            boundary_k=boundary_k,
            n_global_clusters=n_global_clusters or len(np.unique(y)),
            n_jobs=n_jobs,
            partition_method=partition_method,
            density_k=density_k,
            random_state=42,
        )
        model.fit(X_scaled)
        metrics = model.evaluate(y)
        metrics["耗时(秒)"] = time.time() - start
        results[name] = metrics

        for key, val in metrics.items():
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

        # 子集划分可视化（仅对 2D 数据有意义）
        if X.shape[1] == 2:
            model.plot_partition(
                X_scaled, save_path=os.path.join(output_dir, f"{name}_partition.png")
            )

        model.plot_clusters(
            X, title=f"{name} 聚类结果（{partition_method}划分）",
            save_path=os.path.join(output_dir, f"{name}_clusters.png")
        )
        model.plot_comparison(
            X, y, save_path=os.path.join(output_dir, f"{name}_comparison.png")
        )

    summary = pd.DataFrame(results).T[
        ["准确率 (ACC)", "调整兰德指数 (ARI)", "标准化互信息 (NMI)",
         "调整互信息 (AMI)", "代表节点数", "最终簇数", "耗时(秒)"]
    ]
    summary.to_excel(os.path.join(output_dir, "summary.xlsx"))
    print_results_table(results)
    print(f"\n结果已保存到：{output_dir}")
    return results


if __name__ == "__main__":
    # ── 实验 1：正常运行（density 划分）─────────────────────────
    run_excel_experiment(
        file_path="KDD10.xlsx",
        n_subsets=300,
        compression="strcture",   # 可选: "hidden_leaf", "conv" 或 "structure"
        keep_ratio=0.02,
        min_representatives=15,
        n_global_clusters=23,    # 自动估计
        partition_method="density", # ← 新：密度感知划分
        density_k=15,
        n_jobs=-1,
        output_dir="results_KDD10_den",
    )

    # ── 实验 2（可选）：用 make_blobs 对比三种划分策略 ────────────
    from sklearn.datasets import make_blobs
    X_demo, _ = make_blobs(n_samples=2000, centers=6, random_state=42)
    compare_partition_strategies(X_demo, n_subsets=20, save_path="partition_compare.png")