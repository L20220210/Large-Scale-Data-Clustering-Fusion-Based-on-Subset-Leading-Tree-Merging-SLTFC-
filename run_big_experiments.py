# -*- coding: utf-8 -*-
"""
SLTFC 大规模合成实验一键脚本
================================
与用户当前 subsetv3.py 配套，完成 4 组实验：
1) 多密度 + 类别不均衡 Gaussian Mixture
2) 多密度 + 类别不均衡 + 10% 均匀噪声
3) 1M -> 5M -> 10M -> 50M -> 100M 扩展性曲线
4) 固定 N，逐渐增加类别不均衡与密度差异，观察 SLTFC - RandSamplingLT 的性能差

重要说明
--------
- 10M 质量实验默认使用“采样候选池估计密度 + Voronoi 分配”的大规模工程实现。
  它保留了 density-aware Voronoi 的核心思想，但不是 subsetv3.py 中对所有样本做全局 KNN 的逐点精确实现。
  原因是对 10M/100M 样本直接做全量 KNN 预估会成为新的主要瓶颈。
- 第 3 组 50M/100M 扩展性实验使用“随机顺序数据的连续分块 = 随机子集”的流式实现，
  目的仅是验证“子集建树 -> 代表节点压缩 -> 全局融合”的可扩展性；
  density partition 的质量优势由实验 1/2/4 单独验证。
- 所有大规模实现仍使用 subsetv3.py 中相同的 LeadingTree、代表节点规则、全局融合和 top-k 标签传播思想。

先运行：
    python run_big_experiments.py --experiment quick
确认环境正常后再跑正式实验。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# 限制底层 BLAS 线程，避免与 joblib 叠加后线程爆炸。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from subsetv3 import (
    LeadingTree,
    RepresentativeNode,
    SubsetLeadingTreeFusionClustering,
)


# ============================================================
# 0. 通用工具
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class PeakMemoryMonitor:
    """后台轮询当前 Python 进程的 RSS，记录峰值内存。"""
    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self._stop = threading.Event()
        self.peak = 0
        self.proc = psutil.Process(os.getpid())
        self.thread = None

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self.proc.memory_info().rss
                # joblib 这里优先 threads；若有子进程，也把子进程算进去
                for c in self.proc.children(recursive=True):
                    try:
                        rss += c.memory_info().rss
                    except Exception:
                        pass
                self.peak = max(self.peak, rss)
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = 0
        self._stop.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)

    @property
    def peak_gb(self) -> float:
        return self.peak / (1024 ** 3)


def chunk_slices(n: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    for st in range(0, n, chunk_size):
        yield st, min(st + chunk_size, n)


def geometric_weights(k: int, max_min_ratio: float) -> np.ndarray:
    """生成最大类/最小类约为 max_min_ratio 的几何递减权重。"""
    if max_min_ratio <= 1:
        w = np.ones(k, dtype=float)
    else:
        w = np.geomspace(1.0, 1.0 / max_min_ratio, num=k)
    return w / w.sum()


def generate_centers(k: int, d: int, rng: np.random.Generator,
                     min_dist: float = 12.0, box: float = 24.0) -> np.ndarray:
    """拒绝采样生成彼此分离的中心，避免随机中心偶然重叠。"""
    centers: List[np.ndarray] = []
    attempts = 0
    while len(centers) < k:
        attempts += 1
        if attempts > 200000:
            raise RuntimeError("无法生成足够分离的中心，请减小 min_dist 或增大 box。")
        c = rng.uniform(-box, box, size=d)
        if not centers:
            centers.append(c)
            continue
        ds = np.linalg.norm(np.vstack(centers) - c, axis=1)
        if np.min(ds) >= min_dist:
            centers.append(c)
    return np.vstack(centers).astype(np.float32)


def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    # 固定符号，增强可重复性
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs
    return Q.astype(np.float32)


DIFFICULTY_PROFILES = {
    # easy 与原版基本一致：簇间距离很大，主要用于 sanity check。
    "easy":   {"center_box": 24.0, "min_center_dist": 12.0, "base_std": 0.45},
    # medium：让稀疏簇之间产生适量重叠，但仍保留低密度谷。
    "medium": {"center_box": 4.2,  "min_center_dist": 3.0,  "base_std": 0.65},
    # hard：簇更靠近且整体更扩散，用于压力测试；不预设 SLTFC 必须胜出。
    "hard":   {"center_box": 3.0,  "min_center_dist": 2.0,  "base_std": 0.75},
}


def make_gmm_params(
    k: int = 10,
    d: int = 8,
    imbalance_ratio: float = 50.0,
    density_ratio: float = 6.0,
    anisotropy: float = 2.0,
    random_state: int = 42,
    difficulty: str = "medium",
) -> Dict[str, np.ndarray]:
    """
    构造多密度、大小不均衡、不同协方差方向的 GMM 参数。

    difficulty 控制簇间重叠程度：
      easy   : 中心远、重叠很少（只用于正确性检查）；
      medium : 有适量重叠，适合作为论文主质量实验的起点；
      hard   : 重叠明显，用于压力测试和适用边界分析。

    注意：难度档位改变的是“中心空间范围 + 最小中心距离 + 基础标准差”，
    imbalance_ratio 与 density_ratio 仍独立控制类别数量不均衡和簇尺度差异。
    """
    difficulty = difficulty.lower()
    if difficulty not in DIFFICULTY_PROFILES:
        raise ValueError(f"未知 difficulty={difficulty!r}，应为 easy/medium/hard")
    prof = DIFFICULTY_PROFILES[difficulty]

    rng = np.random.default_rng(random_state)
    centers = generate_centers(
        k, d, rng,
        min_dist=prof["min_center_dist"],
        box=prof["center_box"],
    )
    weights = geometric_weights(k, imbalance_ratio).astype(np.float64)

    base_std = float(prof["base_std"])
    base_stds = np.geomspace(base_std, base_std * density_ratio, num=k).astype(np.float32)

    rotations = []
    axis_scales = []
    for j in range(k):
        Q = random_orthogonal(d, rng)
        # 让每个簇不是完全球形。几何均值约 1，避免整体尺度失控。
        if anisotropy <= 1:
            axes = np.ones(d, dtype=np.float32)
        else:
            raw = np.geomspace(1.0 / math.sqrt(anisotropy),
                               math.sqrt(anisotropy), d)
            rng.shuffle(raw)
            axes = raw.astype(np.float32)
        rotations.append(Q)
        axis_scales.append(axes)

    return {
        "centers": centers,
        "weights": weights,
        "base_stds": base_stds,
        "rotations": np.stack(rotations),
        "axis_scales": np.stack(axis_scales),
    }


def _sample_gmm_chunk(
    m: int,
    params: Dict[str, np.ndarray],
    rng: np.random.Generator,
    noise_fraction: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    centers = params["centers"]
    weights = params["weights"]
    base_stds = params["base_stds"]
    rotations = params["rotations"]
    axis_scales = params["axis_scales"]
    k, d = centers.shape

    probs = np.concatenate([weights * (1.0 - noise_fraction), [noise_fraction]])
    probs = probs / probs.sum()
    raw_labels = rng.choice(k + 1, size=m, p=probs)
    X = np.empty((m, d), dtype=np.float32)
    y = raw_labels.astype(np.int32)

    for c in range(k):
        idx = np.flatnonzero(raw_labels == c)
        if len(idx) == 0:
            continue
        z = rng.normal(size=(len(idx), d)).astype(np.float32)
        z *= axis_scales[c]
        z = z @ rotations[c].T
        z *= base_stds[c]
        z += centers[c]
        X[idx] = z

    # 均匀背景噪声，不把它当作一个“真实簇”；标签设为 -1。
    noise_idx = np.flatnonzero(raw_labels == k)
    if len(noise_idx) > 0:
        lo = centers.min(axis=0) - 8.0 * base_stds.max()
        hi = centers.max(axis=0) + 8.0 * base_stds.max()
        X[noise_idx] = rng.uniform(lo, hi, size=(len(noise_idx), d)).astype(np.float32)
        y[noise_idx] = -1

    return X, y


def generate_gmm_in_memory(
    n: int,
    params: Dict[str, np.ndarray],
    noise_fraction: float = 0.0,
    random_state: int = 123,
    chunk_size: int = 250_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """内存版生成器。10M×8 float32 约 0.32GB，仅原始 X。"""
    d = params["centers"].shape[1]
    X = np.empty((n, d), dtype=np.float32)
    y = np.empty(n, dtype=np.int32)
    rng = np.random.default_rng(random_state)
    for st, ed in chunk_slices(n, chunk_size):
        Xc, yc = _sample_gmm_chunk(ed - st, params, rng, noise_fraction)
        X[st:ed] = Xc
        y[st:ed] = yc
    return X, y


def generate_gmm_memmap(
    out_dir: str | Path,
    n: int,
    params: Dict[str, np.ndarray],
    noise_fraction: float = 0.0,
    random_state: int = 123,
    chunk_size: int = 250_000,
    force: bool = False,
) -> Tuple[np.memmap, np.memmap, Path, Path]:
    """磁盘 memmap 版，供 50M/100M 使用。"""
    out_dir = ensure_dir(out_dir)
    d = params["centers"].shape[1]
    x_path = out_dir / f"X_n{n}_d{d}_noise{noise_fraction:.2f}.f32"
    y_path = out_dir / f"y_n{n}_d{d}_noise{noise_fraction:.2f}.i32"

    expected_x = n * d * 4
    expected_y = n * 4
    if (not force and x_path.exists() and y_path.exists()
            and x_path.stat().st_size == expected_x
            and y_path.stat().st_size == expected_y):
        X = np.memmap(x_path, dtype=np.float32, mode="r+", shape=(n, d))
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=(n,))
        return X, y, x_path, y_path

    X = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(n, d))
    y = np.memmap(y_path, dtype=np.int32, mode="w+", shape=(n,))
    rng = np.random.default_rng(random_state)
    t0 = time.time()
    for i, (st, ed) in enumerate(chunk_slices(n, chunk_size), 1):
        Xc, yc = _sample_gmm_chunk(ed - st, params, rng, noise_fraction)
        X[st:ed] = Xc
        y[st:ed] = yc
        if i % 20 == 0 or ed == n:
            print(f"  生成数据: {ed:,}/{n:,} ({ed/n:.1%}), {time.time()-t0:.1f}s")
    X.flush(); y.flush()
    return X, y, x_path, y_path


# ============================================================
# 1. 指标：大样本下尽量用 contingency matrix，避免额外大对象
# ============================================================

def contingency_matrix_fast(y_true: np.ndarray, y_pred: np.ndarray,
                            mask_non_noise: bool = True) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if mask_non_noise:
        mask = y_true >= 0
        yt = y_true[mask]
        yp = y_pred[mask]
    else:
        yt = y_true
        yp = y_pred

    true_vals, yt2 = np.unique(yt, return_inverse=True)
    pred_vals, yp2 = np.unique(yp, return_inverse=True)
    kp = len(pred_vals)
    flat = yt2.astype(np.int64) * kp + yp2.astype(np.int64)
    cm = np.bincount(flat, minlength=len(true_vals) * kp)
    return cm.reshape(len(true_vals), kp)


def metrics_from_contingency(cm: np.ndarray) -> Dict[str, float]:
    cm = cm.astype(np.float64)
    n = cm.sum()
    if n <= 1:
        return {"ACC": np.nan, "ARI": np.nan, "NMI": np.nan}

    # ACC
    rr, cc = linear_sum_assignment(-cm)
    acc = cm[rr, cc].sum() / n

    # ARI
    def comb2(x):
        return x * (x - 1.0) / 2.0
    sum_ij = comb2(cm).sum()
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    sum_row = comb2(row).sum()
    sum_col = comb2(col).sum()
    total = comb2(n)
    expected = (sum_row * sum_col / total) if total > 0 else 0.0
    max_index = 0.5 * (sum_row + sum_col)
    denom = max_index - expected
    ari = (sum_ij - expected) / denom if abs(denom) > 1e-15 else 1.0

    # NMI (arithmetic normalization，与 sklearn 默认一致)
    p = cm / n
    pi = row / n
    pj = col / n
    nz = p > 0
    denom_outer = np.outer(pi, pj)
    mi = np.sum(p[nz] * np.log(p[nz] / denom_outer[nz]))
    hi = -np.sum(pi[pi > 0] * np.log(pi[pi > 0]))
    hj = -np.sum(pj[pj > 0] * np.log(pj[pj > 0]))
    nmi = 1.0 if hi + hj == 0 else mi / ((hi + hj) / 2.0)

    return {"ACC": float(acc), "ARI": float(ari), "NMI": float(nmi)}


def evaluate_labels(y_true: np.ndarray, y_pred: np.ndarray,
                    noise_primary: bool = True,
                    ami_max_n: int = 2_000_000,
                    random_state: int = 42) -> Dict[str, float]:
    """
    噪声实验主指标默认只在真实簇样本(y>=0)上计算。
    AMI 对超大 n 采用最多 ami_max_n 的固定随机子样本，并记录 AMI_eval_n。
    """
    cm = contingency_matrix_fast(y_true, y_pred, mask_non_noise=noise_primary)
    out = metrics_from_contingency(cm)

    if noise_primary:
        mask = np.asarray(y_true) >= 0
        idx_all = np.flatnonzero(mask)
    else:
        idx_all = np.arange(len(y_true))

    if len(idx_all) <= ami_max_n:
        idx = idx_all
    else:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(idx_all, size=ami_max_n, replace=False)
    out["AMI"] = float(adjusted_mutual_info_score(np.asarray(y_true)[idx], np.asarray(y_pred)[idx]))
    out["AMI_eval_n"] = int(len(idx))
    out["eval_n"] = int(cm.sum())
    return out


# ============================================================
# 2. 可扩展版 SLTFC：不改变核心流程，只避免保存所有局部 NetworkX 树
# ============================================================

class ScalableSLTFC(SubsetLeadingTreeFusionClustering):
    """
    大规模工程实现：
    - 局部树训练后立即压缩，只保留 RepresentativeNode，不保存全部局部树；
    - 标签传播按 chunk 向量化；
    - density 分区在 n 很大时使用候选池估计密度，再做 Voronoi 分配。

    注意：候选池 density 是大规模近似实现。实验报告里必须说明。
    """
    def __init__(self, *args,
                 density_candidate_size: int = 300_000,
                 query_chunk_size: int = 200_000,
                 candidate_pool_seed: int = 2026,
                 density_seed_method: str = "weighted_random",
                 exact_density_limit: int = 500_000,
                 local_knn_k: int = 30,
                 max_subset_factor: float = 2.0,
                 refine_oversized_density: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.density_candidate_size = int(density_candidate_size)
        self.query_chunk_size = int(query_chunk_size)
        self.candidate_pool_seed = int(candidate_pool_seed)
        self.density_seed_method = str(density_seed_method).lower()
        self.exact_density_limit = int(exact_density_limit)
        self.local_knn_k = int(local_knn_k)
        self.max_subset_factor = float(max_subset_factor)
        self.refine_oversized_density = bool(refine_oversized_density)

    def _kd_split_until_capacity(
        self, X: np.ndarray, idx: np.ndarray, max_size: int
    ) -> List[np.ndarray]:
        """
        仅对过大的 density-Voronoi 单元继续做 KD 中位数二分，
        直到每个局部任务不超过 max_size。这样保留第一阶段 density-aware
        分区的空间语义，同时避免单个巨型子集拖慢并行建树。
        """
        pending = [np.asarray(idx, dtype=np.int64)]
        out: List[np.ndarray] = []
        while pending:
            g = pending.pop()
            if len(g) <= max_size:
                out.append(g)
                continue

            # 对超大单元只抽样一部分点估计最大方差维度，降低临时内存。
            if len(g) > 50_000:
                sample_pos = np.linspace(0, len(g) - 1, 50_000, dtype=np.int64)
                probe = np.asarray(X[g[sample_pos]], dtype=np.float32)
            else:
                probe = np.asarray(X[g], dtype=np.float32)
            best_dim = int(np.argmax(probe.var(axis=0)))
            del probe

            vals = np.asarray(X[g, best_dim], dtype=np.float32)
            med = float(np.median(vals))
            mask = vals <= med
            if mask.all() or (~mask).all():
                # 极端退化情况：按索引对半切，保证一定能终止。
                mid = len(g) // 2
                left, right = g[:mid], g[mid:]
            else:
                left, right = g[mask], g[~mask]
            del vals, mask
            if len(left):
                pending.append(left)
            if len(right):
                pending.append(right)
        return out

    def _refine_density_capacity(
        self, X: np.ndarray, groups: List[np.ndarray], requested_s: int
    ) -> List[np.ndarray]:
        if not self.refine_oversized_density or self.max_subset_factor <= 0:
            return groups
        mean_target = len(X) / max(1, requested_s)
        max_size = max(2, int(math.ceil(self.max_subset_factor * mean_target)))
        before = np.asarray([len(g) for g in groups], dtype=np.int64)
        if len(before) == 0 or before.max() <= max_size:
            return groups

        refined: List[np.ndarray] = []
        n_split_cells = 0
        for g in groups:
            if len(g) > max_size:
                n_split_cells += 1
                refined.extend(self._kd_split_until_capacity(X, g, max_size))
            else:
                refined.append(g)

        after = np.asarray([len(g) for g in refined], dtype=np.int64)
        print(
            f"  [容量约束] max_subset_factor={self.max_subset_factor:g}, "
            f"上限={max_size:,}; 细分原始大单元={n_split_cells}; "
            f"子集数 {len(groups)} -> {len(refined)}, "
            f"max {before.max():,} -> {after.max():,}, "
            f"CV {before.std()/before.mean():.3f} -> {after.std()/after.mean():.3f}"
        )
        return refined

    def _density_diverse_seed_indices(
        self, cand_X: np.ndarray, density: np.ndarray, n_seeds: int
    ) -> np.ndarray:
        """
        确定性的“密度 + 空间多样性”贪心选种子。

        第1个种子：局部密度最高的候选点。
        第2..S个种子：优先选择“密度较高且离现有种子较远”的候选点。
        score_i = rho_score_i * distance_score_i

        rho_score 用 5%~95% 分位裁剪后归一化，避免极端密度值支配全部选择。
        distance_score 为到当前已选种子集合的最近欧氏距离（归一化）。

        该过程无随机抽样，因此在候选池固定时完全可复现；同时避免单纯 top-density
        导致大量种子扎堆在同一个高密度簇。
        """
        m = len(cand_X)
        n_seeds = min(int(n_seeds), m)
        if n_seeds <= 0:
            return np.empty(0, dtype=np.int64)

        rho = np.asarray(density, dtype=np.float64)
        lo, hi = np.percentile(rho, [5.0, 95.0])
        if hi - lo < 1e-12:
            rho_score = np.ones(m, dtype=np.float64)
        else:
            rho_score = np.clip((rho - lo) / (hi - lo), 0.0, 1.0)
            # 给低密度候选保留一个很小的基础权重，避免完全失去空间覆盖。
            rho_score = 0.05 + 0.95 * rho_score

        chosen = np.empty(n_seeds, dtype=np.int64)
        first = int(np.argmax(rho))
        chosen[0] = first

        # 使用 float32 计算距离以控制 40万/80万候选池的临时内存。
        diff = cand_X - cand_X[first]
        min_d2 = np.einsum('ij,ij->i', diff, diff, optimize=True).astype(np.float32, copy=False)
        min_d2[first] = 0.0

        selected_mask = np.zeros(m, dtype=bool)
        selected_mask[first] = True

        for j in range(1, n_seeds):
            max_d2 = float(min_d2.max())
            if max_d2 <= 1e-20:
                # 完全退化时按密度顺序补足。
                score = rho_score.copy()
            else:
                distance_score = np.sqrt(min_d2 / max_d2, dtype=np.float32)
                score = rho_score * distance_score
            score[selected_mask] = -1.0
            nxt = int(np.argmax(score))
            chosen[j] = nxt
            selected_mask[nxt] = True

            diff = cand_X - cand_X[nxt]
            d2 = np.einsum('ij,ij->i', diff, diff, optimize=True).astype(np.float32, copy=False)
            np.minimum(min_d2, d2, out=min_d2)
            min_d2[selected_mask] = 0.0

        return chosen

    def _density_spatial_seed_indices(
        self, cand_X: np.ndarray, density: np.ndarray, n_seeds: int
    ) -> np.ndarray:
        """
        可扩展的确定性“空间分层 + 局部密度峰”选种子。

        先把候选池用平衡 KD 递归划成 n_seeds 个空间区域；
        再在每个区域选择局部密度最大的候选点作为 Voronoi seed。

        与 density-diverse 的逐轮全候选扫描 O(C*S) 相比，
        该过程更接近 O(C*log S)，适合 S 随 N 增长到 1000/10000 的场景。
        """
        m = len(cand_X)
        n_seeds = min(int(n_seeds), m)
        if n_seeds <= 0:
            return np.empty(0, dtype=np.int64)

        all_idx = np.arange(m, dtype=np.int64)
        leaves = []
        stack = [(all_idx, n_seeds)]

        while stack:
            idx, parts = stack.pop()
            if parts <= 1 or len(idx) <= 1:
                leaves.append(idx)
                continue

            # 用最多 20k 个点估计最大方差维，降低大候选池临时开销。
            if len(idx) > 20_000:
                pos = np.linspace(0, len(idx)-1, 20_000, dtype=np.int64)
                probe = cand_X[idx[pos]]
            else:
                probe = cand_X[idx]
            best_dim = int(np.argmax(np.asarray(probe, dtype=np.float32).var(axis=0)))

            vals = np.asarray(cand_X[idx, best_dim], dtype=np.float32)
            n_left = parts // 2
            n_right = parts - n_left
            # 按所需叶子数比例分配点数，保证每个叶子最终有点。
            cut = max(1, min(len(idx)-1, int(round(len(idx) * n_left / parts))))
            order = np.argpartition(vals, cut-1)
            left = idx[order[:cut]]
            right = idx[order[cut:]]
            stack.append((right, n_right))
            stack.append((left, n_left))

        chosen = []
        rho = np.asarray(density)
        for idx in leaves:
            if len(idx) == 0:
                continue
            chosen.append(int(idx[np.argmax(rho[idx])]))

        chosen = np.asarray(chosen, dtype=np.int64)
        # 正常情况下恰好 n_seeds；极端退化时按全局密度补足。
        if len(chosen) < n_seeds:
            mask = np.ones(m, dtype=bool)
            mask[chosen] = False
            remain = np.where(mask)[0]
            add = remain[np.argsort(-rho[remain])[:n_seeds-len(chosen)]]
            chosen = np.concatenate([chosen, add.astype(np.int64)])
        elif len(chosen) > n_seeds:
            chosen = chosen[:n_seeds]
        return chosen

    def _density_partition_large(self, X: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
        n = len(X)
        s = min(self.n_subsets, n)
        candidate_n = min(n, max(self.density_candidate_size, s * 50))

        # 随机源拆分诊断：候选池使用独立且固定的随机种子，
        # 不再随 partition seed 改变。这样 stability 实验只改变
        # “密度加权种子抽样”这一处随机性。
        candidate_rng = np.random.default_rng(self.candidate_pool_seed)
        candidate_order = candidate_rng.permutation(n)
        cand_idx = candidate_order[:candidate_n]
        cand_X = np.asarray(X[cand_idx], dtype=np.float32)
        k_eff = min(self.density_k + 1, candidate_n)
        print(f"  [大规模 density] 候选池={candidate_n:,}, k={self.density_k}, seeds={s}, "
              f"candidate_pool_seed={self.candidate_pool_seed}")
        nbrs = NearestNeighbors(n_neighbors=k_eff, n_jobs=1).fit(cand_X)
        dist, _ = nbrs.kneighbors(cand_X)
        density = 1.0 / (dist[:, -1] + 1e-8)
        method = self.density_seed_method
        seed_t0 = time.perf_counter()
        if method == "diverse":
            print("  [density种子] 确定性 density-diverse 贪心选择（近似 O(C*S)）")
            chosen_local = self._density_diverse_seed_indices(cand_X, density, s)
        elif method == "spatial_density":
            print("  [density种子] 可扩展 density-spatial：KD空间分层 + 区域内最高密度（近似 O(C log S)）")
            chosen_local = self._density_spatial_seed_indices(cand_X, density, s)
        else:
            probs = density / density.sum()
            print("  [density种子] density-weighted random sampling")
            chosen_local = rng.choice(candidate_n, size=s, replace=False, p=probs)
        self.seed_selection_time_ = time.perf_counter() - seed_t0
        print(f"  [种子选择耗时] {self.seed_selection_time_:.3f}s")
        seed_X = cand_X[chosen_local]

        # 种子很少，kd_tree 在低维合成数据上通常比全量 KNN 稳定。
        alg = "kd_tree" if X.shape[1] <= 20 else "auto"
        seed_nn = NearestNeighbors(n_neighbors=1, algorithm=alg, n_jobs=1).fit(seed_X)
        assign = np.empty(n, dtype=np.int32)
        for st, ed in chunk_slices(n, self.query_chunk_size):
            _, a = seed_nn.kneighbors(np.asarray(X[st:ed], dtype=np.float32))
            assign[st:ed] = a[:, 0].astype(np.int32)

        # 一次排序分组，避免对每个 seed 做一次 np.where（那会变成 O(n*S)）。
        order = np.argsort(assign, kind="stable")
        counts = np.bincount(assign, minlength=s)
        cuts = np.cumsum(counts)[:-1]
        groups = np.split(order, cuts)
        groups = [g.astype(np.int64, copy=False) for g in groups if len(g) > 0]
        groups = self._refine_density_capacity(X, groups, requested_s=s)
        sizes = np.array([len(g) for g in groups])
        self.last_partition_stats_ = {
            "actual_subsets": int(len(groups)),
            "min_subset_size": int(sizes.min()),
            "max_subset_size": int(sizes.max()),
            "mean_subset_size": float(sizes.mean()),
            "std_subset_size": float(sizes.std()),
            "subset_cv": float(sizes.std()/sizes.mean()),
        }
        print(f"  [density 分区] 实际子集={len(groups)}, min={sizes.min()}, "
              f"max={sizes.max()}, mean={sizes.mean():.1f}, "
              f"std={sizes.std():.1f}, CV={sizes.std()/sizes.mean():.3f}")
        return groups

    def _partition_data_large(self, X: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
        n = len(X)
        method = self.partition_method.lower()
        if method == "density":
            if n <= self.exact_density_limit:
                groups = super()._partition_data(X, rng)
                return self._refine_density_capacity(X, groups, requested_s=min(self.n_subsets, n))
            return self._density_partition_large(X, rng)
        if method == "random":
            idx = rng.permutation(n)
            return [a.astype(np.int64, copy=False) for a in np.array_split(idx, self.n_subsets) if len(a)]
        if method == "spatial":
            print("  [提示] 大规模 spatial 仍调用原 KD 递归实现，50M/100M 不建议使用。")
            return super()._partition_data(X, rng)
        raise ValueError(f"未知 partition_method={self.partition_method}")

    def _train_compress_one(self, indices: np.ndarray, subset_id: int) -> List[RepresentativeNode]:
        local_X = np.asarray(self.X_[indices], dtype=np.float32)
        model = LeadingTree(
            dc=self.dc,
            rho_threshold=self.rho_threshold,
            delta_threshold=self.delta_threshold,
            min_centers=1,
            knn_k=self.local_knn_k,
        ).fit(local_X)
        kept = self._select_representative_nodes(model)
        subtree_sizes = self._all_subtree_sizes(model.tree, len(local_X))
        score_dict = getattr(model, "representative_scores_", {})

        out: List[RepresentativeNode] = []
        for li in kept:
            gi = int(indices[li])
            out.append(RepresentativeNode(
                original_index=gi,
                subset_id=subset_id,
                local_index=int(li),
                local_label=int(model.labels_[li]),
                x=np.asarray(self.X_[gi], dtype=np.float32).copy(),
                rho=float(model.rho[li]),
                delta=float(model.delta[li]),
                gamma=float(model.gamma[li]),
                weight=float(subtree_sizes[li]),
                importance=float(score_dict.get(int(li), 0.0)),
            ))
        del model, local_X
        return out

    def _propagate_labels_chunked(self) -> None:
        rep_X = np.vstack([r.x for r in self.representatives_]).astype(np.float32, copy=False)
        rep_labels = np.asarray(self.global_model_.labels_, dtype=np.int32)
        rep_weights = np.asarray([r.weight for r in self.representatives_], dtype=np.float32)
        top_k = min(5, len(rep_X))
        nn = NearestNeighbors(n_neighbors=top_k, n_jobs=1).fit(rep_X)
        n_cls = int(rep_labels.max()) + 1
        n = len(self.X_)
        final = np.empty(n, dtype=np.int32)

        for st, ed in chunk_slices(n, self.query_chunk_size):
            dists, idx = nn.kneighbors(np.asarray(self.X_[st:ed], dtype=np.float32))
            dists = np.maximum(dists.astype(np.float32), 1e-8)
            labs = rep_labels[idx]
            scores = rep_weights[idx] / dists
            m = ed - st
            vote = np.zeros((m, n_cls), dtype=np.float32)
            rows = np.repeat(np.arange(m, dtype=np.int64), top_k)
            np.add.at(vote, (rows, labs.reshape(-1)), scores.reshape(-1))
            final[st:ed] = np.argmax(vote, axis=1).astype(np.int32)
        self.final_labels_ = final
        self._relabel_continuously()

    def fit(self, X: np.ndarray) -> "ScalableSLTFC":
        """大规模 fit，并记录四个主要阶段耗时，便于 10M+ 可扩展性诊断。"""
        fit_t0 = time.perf_counter()
        self.X_ = X
        rng = np.random.default_rng(self.random_state)

        t0 = time.perf_counter()
        splits = self._partition_data_large(X, rng)
        partition_time = time.perf_counter() - t0

        self.subset_indices_ = []  # 有意不保存全部局部树
        self.subset_models_ = []

        print(f"  [局部建树] {len(splits)} 个子集, n_jobs={self.n_jobs}, compression={self.compression}")
        t0 = time.perf_counter()
        blocks = Parallel(n_jobs=self.n_jobs, prefer="threads", batch_size=1)(
            delayed(self._train_compress_one)(idx, sid) for sid, idx in enumerate(splits)
        )
        self.representatives_ = [r for block in blocks for r in block]
        local_tree_time = time.perf_counter() - t0
        print(f"  [代表节点] {len(self.representatives_):,}")
        # 释放巨大的划分索引
        del splits, blocks
        gc.collect()

        t0 = time.perf_counter()
        self._fuse_representatives()
        fusion_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._propagate_labels_chunked()
        propagation_time = time.perf_counter() - t0

        self.fit_stage_times_ = {
            "partition_time_sec": float(partition_time),
            "seed_selection_time_sec": float(getattr(self, "seed_selection_time_", 0.0)),
            "local_tree_time_sec": float(local_tree_time),
            "fusion_time_sec": float(fusion_time),
            "propagation_time_sec": float(propagation_time),
            "fit_total_time_sec": float(time.perf_counter() - fit_t0),
        }
        print(
            "  [阶段耗时] "
            f"partition={partition_time:.1f}s, "
            f"local_tree={local_tree_time:.1f}s, "
            f"fusion={fusion_time:.1f}s, "
            f"propagation={propagation_time:.1f}s, "
            f"fit_total={self.fit_stage_times_['fit_total_time_sec']:.1f}s"
        )
        return self

    def fit_contiguous_random_stream(self, X: np.ndarray, subset_size: int = 20_000) -> "ScalableSLTFC":
        """
        用于 50M/100M scaling：假设合成数据生成顺序本身已按 mixture 随机打散，
        因此连续切片在统计上等价于随机划分，且无需构造 100M 长度的 permutation。
        """
        self.X_ = X
        n = len(X)
        slices = [(st, ed) for st, ed in chunk_slices(n, subset_size)]
        print(f"  [流式随机子集] n={n:,}, subset_size={subset_size:,}, subsets={len(slices):,}")

        def one_slice(st: int, ed: int, sid: int) -> List[RepresentativeNode]:
            indices = np.arange(st, ed, dtype=np.int64)
            return self._train_compress_one(indices, sid)

        # 分批调用，避免一次创建海量 future。
        all_reps: List[RepresentativeNode] = []
        batch_tasks = max(32, self.n_jobs * 8 if self.n_jobs > 0 else 32)
        for b0 in range(0, len(slices), batch_tasks):
            batch = slices[b0:b0 + batch_tasks]
            blocks = Parallel(n_jobs=self.n_jobs, prefer="threads", batch_size=1)(
                delayed(one_slice)(st, ed, b0 + i) for i, (st, ed) in enumerate(batch)
            )
            for block in blocks:
                all_reps.extend(block)
            if b0 % (batch_tasks * 10) == 0:
                print(f"    已处理子集 {min(b0+len(batch),len(slices)):,}/{len(slices):,}, reps={len(all_reps):,}")
        self.representatives_ = all_reps
        print(f"  [代表节点] {len(all_reps):,}")
        self._fuse_representatives()
        self._propagate_labels_chunked()
        return self


# ============================================================
# 3. 基线
# ============================================================

def force_k_centers(model: LeadingTree, X: np.ndarray, k: int) -> LeadingTree:
    centers = np.argsort(-model.gamma)[:k]
    model.centers = np.asarray(centers, dtype=int)
    model.tree = model._build_tree(X, model.parent_, model.centers)
    model.labels_ = model._assign_labels(model.tree, model.centers)
    return model


def predict_from_reps_chunked(
    X: np.ndarray,
    rep_X: np.ndarray,
    rep_labels: np.ndarray,
    rep_weights: Optional[np.ndarray] = None,
    top_k: int = 5,
    chunk_size: int = 200_000,
) -> np.ndarray:
    rep_X = np.asarray(rep_X, dtype=np.float32)
    rep_labels = np.asarray(rep_labels, dtype=np.int32)
    if rep_weights is None:
        rep_weights = np.ones(len(rep_X), dtype=np.float32)
    else:
        rep_weights = np.asarray(rep_weights, dtype=np.float32)
    top_k = min(top_k, len(rep_X))
    nn = NearestNeighbors(n_neighbors=top_k, n_jobs=1).fit(rep_X)
    n_cls = int(rep_labels.max()) + 1
    out = np.empty(len(X), dtype=np.int32)
    for st, ed in chunk_slices(len(X), chunk_size):
        dists, idx = nn.kneighbors(np.asarray(X[st:ed], dtype=np.float32))
        dists = np.maximum(dists.astype(np.float32), 1e-8)
        labs = rep_labels[idx]
        scores = rep_weights[idx] / dists
        m = ed - st
        vote = np.zeros((m, n_cls), dtype=np.float32)
        rows = np.repeat(np.arange(m, dtype=np.int64), top_k)
        np.add.at(vote, (rows, labs.reshape(-1)), scores.reshape(-1))
        out[st:ed] = np.argmax(vote, axis=1)
    return out


def run_random_sampling_lt(X: np.ndarray, k: int, sample_size: int,
                           random_state: int = 42,
                           chunk_size: int = 200_000) -> Tuple[np.ndarray, Dict]:
    rng = np.random.default_rng(random_state)
    sample_size = min(int(sample_size), len(X))
    idx = rng.choice(len(X), size=sample_size, replace=False)
    Xs = np.asarray(X[idx], dtype=np.float32)
    model = LeadingTree(min_centers=k, knn_k=30).fit(Xs)
    model = force_k_centers(model, Xs, k)
    pred = predict_from_reps_chunked(X, Xs, model.labels_, None, top_k=5, chunk_size=chunk_size)
    return pred, {"sample_size": sample_size}


def run_minibatch_kmeans(X: np.ndarray, k: int, random_state: int = 42,
                         batch_size: int = 65536,
                         max_iter: int = 100,
                         chunk_size: int = 200_000) -> Tuple[np.ndarray, Dict]:
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init=3,
        reassignment_ratio=0.01,
    )
    model.fit(X)
    pred = np.empty(len(X), dtype=np.int32)
    for st, ed in chunk_slices(len(X), chunk_size):
        pred[st:ed] = model.predict(np.asarray(X[st:ed], dtype=np.float32)).astype(np.int32)
    return pred, {}


def run_minibatch_kmeans_stream(X: np.ndarray, k: int, random_state: int = 42,
                                batch_size: int = 65536,
                                epochs: int = 2,
                                chunk_size: int = 200_000) -> Tuple[np.ndarray, Dict]:
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=batch_size,
        n_init=1,
        reassignment_ratio=0.01,
    )
    for ep in range(epochs):
        for st, ed in chunk_slices(len(X), chunk_size):
            model.partial_fit(np.asarray(X[st:ed], dtype=np.float32))
    pred = np.empty(len(X), dtype=np.int32)
    for st, ed in chunk_slices(len(X), chunk_size):
        pred[st:ed] = model.predict(np.asarray(X[st:ed], dtype=np.float32)).astype(np.int32)
    return pred, {"epochs": epochs}


# ============================================================
# 4. 单次方法运行包装
# ============================================================

def timed_method(name: str, func, y_true: np.ndarray,
                 noise_primary: bool = True) -> Dict:
    gc.collect()
    print(f"\n>>> 开始: {name}")
    t0 = time.time()
    with PeakMemoryMonitor() as mon:
        pred, extra = func()
    elapsed = time.time() - t0
    metrics = evaluate_labels(y_true, pred, noise_primary=noise_primary)
    row = {
        "method": name,
        **metrics,
        "time_sec": elapsed,
        "peak_mem_gb": mon.peak_gb,
        **extra,
    }
    print("    " + ", ".join(
        [f"ACC={row['ACC']:.4f}", f"ARI={row['ARI']:.4f}", f"NMI={row['NMI']:.4f}",
         f"AMI={row['AMI']:.4f}", f"time={elapsed:.1f}s", f"peak={mon.peak_gb:.2f}GB"]
    ))
    del pred
    gc.collect()
    return row


def run_quality_suite(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    out_dir: Path,
    n_subsets: int,
    keep_ratio: float,
    compression: str,
    density_k: int,
    n_jobs: int,
    random_state: int,
    include_minibatch: bool = True,
    max_subset_factor: float = 2.0,
    density_candidate_size: Optional[int] = None,
    density_seed_method: str = "weighted_random",
    candidate_pool_seed: int = 2026,
) -> pd.DataFrame:
    rows = []

    # 先跑 density SLTFC，以实际代表节点数作为 RandomSamplingLT 的采样预算。
    holder = {}
    def sltfc_density():
        model = ScalableSLTFC(
            n_subsets=n_subsets,
            compression=compression,
            keep_ratio=keep_ratio,
            min_representatives=5,
            n_global_clusters=k,
            n_jobs=n_jobs,
            partition_method="density",
            density_k=density_k,
            random_state=random_state,
            density_candidate_size=(
                int(density_candidate_size)
                if density_candidate_size is not None
                else min(400_000, max(100_000, len(X)//20))
            ),
            query_chunk_size=200_000,
            candidate_pool_seed=candidate_pool_seed,
            density_seed_method=density_seed_method,
            max_subset_factor=max_subset_factor,
            refine_oversized_density=True,
        )
        model.fit(X)
        holder["rep_count"] = len(model.representatives_)
        pred = model.final_labels_.copy()
        extra = {"rep_count": len(model.representatives_), "n_subsets": n_subsets}
        del model
        return pred, extra

    density_label = "SLTFC-density-diverse" if density_seed_method == "diverse" else "SLTFC-density"
    rows.append(timed_method(density_label, sltfc_density, y, noise_primary=True))

    # 公平性修正：RandSamplingLT 的采样预算固定为 N * keep_ratio，
    # 不再随 SLTFC-density 每次实际代表节点数的轻微变化而改变。
    rep_budget = max(1, int(round(len(X) * keep_ratio)))
    print(f"  [RandSamplingLT 固定预算] sample_size={rep_budget:,} (= N * keep_ratio)")

    def sltfc_random():
        model = ScalableSLTFC(
            n_subsets=n_subsets,
            compression=compression,
            keep_ratio=keep_ratio,
            min_representatives=5,
            n_global_clusters=k,
            n_jobs=n_jobs,
            partition_method="random",
            density_k=density_k,
            random_state=random_state,
            query_chunk_size=200_000,
        )
        model.fit(X)
        pred = model.final_labels_.copy()
        extra = {"rep_count": len(model.representatives_), "n_subsets": n_subsets}
        del model
        return pred, extra
    rows.append(timed_method("SLTFC-random", sltfc_random, y, noise_primary=True))

    rows.append(timed_method(
        "RandSamplingLT",
        lambda: run_random_sampling_lt(X, k, rep_budget, random_state, 200_000),
        y,
        noise_primary=True,
    ))

    if include_minibatch:
        rows.append(timed_method(
            "MiniBatchKMeans",
            lambda: run_minibatch_kmeans(X, k, random_state, 65536, 100, 200_000),
            y,
            noise_primary=True,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False, encoding="utf-8-sig")
    plot_quality_results(df, out_dir)
    return df


def plot_quality_results(df: pd.DataFrame, out_dir: Path) -> None:
    for metric in ["ARI", "NMI", "ACC", "time_sec", "peak_mem_gb"]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(df["method"], df[metric])
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.png", dpi=220)
        plt.close(fig)


# ============================================================
# 5. 四组实验
# ============================================================

def experiment_1(args) -> pd.DataFrame:
    out = ensure_dir(Path(args.output) / f"exp1_{args.difficulty}_multidensity_imbalanced")
    params = make_gmm_params(
        k=args.k, d=args.d,
        imbalance_ratio=args.imbalance,
        density_ratio=args.density_ratio,
        anisotropy=args.anisotropy,
        random_state=42,
        difficulty=args.difficulty,
    )
    np.savez(out / "gmm_params.npz", **params)
    print(f"\n[实验1] n={args.n:,}, d={args.d}, K={args.k}, difficulty={args.difficulty}, "
          f"imbalance={args.imbalance}:1, density_ratio={args.density_ratio}:1")
    X, y = generate_gmm_in_memory(args.n, params, noise_fraction=0.0, random_state=args.seed)
    n_subsets = args.n_subsets or max(100, int(math.ceil(args.n / args.target_subset_size)))
    df = run_quality_suite(X, y, args.k, out, n_subsets, args.keep_ratio,
                           args.compression, args.density_k, args.jobs, args.seed,
                           max_subset_factor=args.max_subset_factor,
                           density_candidate_size=args.candidate_pool_size,
                           density_seed_method=args.density_seed_method)
    del X, y
    gc.collect()
    return df


def experiment_2(args) -> pd.DataFrame:
    out = ensure_dir(Path(args.output) / f"exp2_{args.difficulty}_multidensity_noise10")
    params = make_gmm_params(
        k=args.k, d=args.d,
        imbalance_ratio=args.imbalance,
        density_ratio=args.density_ratio,
        anisotropy=args.anisotropy,
        random_state=42,
        difficulty=args.difficulty,
    )
    print(f"\n[实验2] n={args.n:,}, noise=10%; 主指标只在 y>=0 的真实簇样本上计算")
    X, y = generate_gmm_in_memory(args.n, params, noise_fraction=0.10, random_state=args.seed)
    n_subsets = args.n_subsets or max(100, int(math.ceil(args.n / args.target_subset_size)))
    df = run_quality_suite(X, y, args.k, out, n_subsets, args.keep_ratio,
                           args.compression, args.density_k, args.jobs, args.seed,
                           max_subset_factor=args.max_subset_factor,
                           density_candidate_size=args.candidate_pool_size,
                           density_seed_method=args.density_seed_method)
    del X, y
    gc.collect()
    return df


def experiment_3(args) -> pd.DataFrame:
    out = ensure_dir(Path(args.output) / "exp3_scaling")
    data_dir = ensure_dir(out / "memmap_data")
    params = make_gmm_params(
        k=args.k, d=args.d,
        imbalance_ratio=args.imbalance,
        density_ratio=args.density_ratio,
        anisotropy=args.anisotropy,
        random_state=42,
        difficulty=args.difficulty,
    )
    rows = []
    sizes = args.scale_sizes
    print("\n[实验3] scaling curve")
    print("  注意：50M/100M 的 SLTFC 使用随机顺序数据连续分块，验证融合框架可扩展性；")
    print("        density-aware partition 的质量优势请看实验1/2/4。")

    for n in sizes:
        print(f"\n===== SCALE n={n:,} =====")
        X, y, xp, yp = generate_gmm_memmap(data_dir, n, params, 0.0, args.seed)
        subset_size = args.scaling_subset_size
        keep_ratio = args.scaling_keep_ratio

        def sltfc_stream():
            model = ScalableSLTFC(
                n_subsets=max(1, math.ceil(n / subset_size)),
                compression=args.scaling_compression,
                keep_ratio=keep_ratio,
                min_representatives=3,
                n_global_clusters=args.k,
                n_jobs=args.jobs,
                partition_method="random",
                random_state=args.seed,
                query_chunk_size=args.query_chunk,
                local_knn_k=30,
            )
            model.fit_contiguous_random_stream(X, subset_size=subset_size)
            pred = model.final_labels_.copy()
            extra = {"n": n, "rep_count": len(model.representatives_), "subset_size": subset_size}
            del model
            return pred, extra

        row = timed_method("SLTFC-stream", sltfc_stream, y, noise_primary=True)
        row["n"] = n
        rows.append(row)

        # MiniBatchKMeans 是可真正全量流式运行的强基线。
        row = timed_method(
            "MiniBatchKMeans",
            lambda: run_minibatch_kmeans_stream(X, args.k, args.seed, 65536, 2, args.query_chunk),
            y,
            noise_primary=True,
        )
        row["n"] = n
        rows.append(row)

        # RandSamplingLT 统一给与 SLTFC 近似代表预算。
        rep_budget = max(5000, min(100000, int(n * keep_ratio)))
        row = timed_method(
            "RandSamplingLT",
            lambda: run_random_sampling_lt(X, args.k, rep_budget, args.seed, args.query_chunk),
            y,
            noise_primary=True,
        )
        row["n"] = n
        rows.append(row)

        # 释放映射
        del X, y
        gc.collect()

        pd.DataFrame(rows).to_csv(out / "scaling_results_partial.csv", index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    df.to_csv(out / "scaling_results.csv", index=False, encoding="utf-8-sig")
    plot_scaling(df, out)
    return df


def plot_scaling(df: pd.DataFrame, out: Path) -> None:
    for metric in ["time_sec", "peak_mem_gb", "ARI", "NMI"]:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for method, g in df.groupby("method"):
            g = g.sort_values("n")
            ax.plot(g["n"], g[metric], marker="o", label=method)
        ax.set_xscale("log")
        if metric in ["time_sec", "peak_mem_gb"]:
            ax.set_yscale("log")
        ax.set_xlabel("样本数 n")
        ax.set_ylabel(metric)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"scaling_{metric}.png", dpi=220)
        plt.close(fig)


def experiment_4(args) -> pd.DataFrame:
    out = ensure_dir(Path(args.output) / "exp4_distribution_stress")
    rows = []
    imbalance_levels = args.imbalance_levels
    density_levels = args.density_levels

    print(f"\n[实验4] 固定 n={args.n:,}，扫描 imbalance × density_ratio")
    for imb in imbalance_levels:
        for den in density_levels:
            print(f"\n===== imbalance={imb}:1, density_ratio={den}:1 =====")
            params = make_gmm_params(
                k=args.k, d=args.d,
                imbalance_ratio=imb,
                density_ratio=den,
                anisotropy=args.anisotropy,
                random_state=42,
            )
            X, y = generate_gmm_in_memory(args.n, params, 0.0, args.seed)
            n_subsets = args.n_subsets or max(100, int(math.ceil(args.n / args.target_subset_size)))

            holder = {}
            def density_run():
                model = ScalableSLTFC(
                    n_subsets=n_subsets,
                    compression=args.compression,
                    keep_ratio=args.keep_ratio,
                    min_representatives=5,
                    n_global_clusters=args.k,
                    n_jobs=args.jobs,
                    partition_method="density",
                    density_k=args.density_k,
                    random_state=args.seed,
                    density_candidate_size=min(400_000, max(100_000, len(X)//20)),
                    query_chunk_size=args.query_chunk,
                    max_subset_factor=args.max_subset_factor,
                    refine_oversized_density=True,
                )
                model.fit(X)
                holder["rep_count"] = len(model.representatives_)
                pred = model.final_labels_.copy()
                extra = {"rep_count": len(model.representatives_)}
                del model
                return pred, extra

            a = timed_method("SLTFC-density", density_run, y, True)
            b = timed_method(
                "RandSamplingLT",
                lambda: run_random_sampling_lt(X, args.k, holder["rep_count"], args.seed, args.query_chunk),
                y,
                True,
            )
            for row in [a, b]:
                row["imbalance_ratio"] = imb
                row["density_ratio"] = den
                rows.append(row)
            del X, y
            gc.collect()
            pd.DataFrame(rows).to_csv(out / "stress_partial.csv", index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    df.to_csv(out / "stress_results.csv", index=False, encoding="utf-8-sig")
    plot_gap_heatmaps(df, out)
    return df


def plot_gap_heatmaps(df: pd.DataFrame, out: Path) -> None:
    for metric in ["ARI", "NMI"]:
        pivot = {}
        ims = sorted(df["imbalance_ratio"].unique())
        dens = sorted(df["density_ratio"].unique())
        mat = np.full((len(dens), len(ims)), np.nan)
        for iy, den in enumerate(dens):
            for ix, imb in enumerate(ims):
                sub = df[(df.imbalance_ratio == imb) & (df.density_ratio == den)]
                a = sub[sub.method == "SLTFC-density"]
                b = sub[sub.method == "RandSamplingLT"]
                if len(a) and len(b):
                    mat[iy, ix] = float(a.iloc[0][metric] - b.iloc[0][metric])
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        im = ax.imshow(mat, aspect="auto", origin="lower")
        ax.set_xticks(range(len(ims)), [f"{x}:1" for x in ims])
        ax.set_yticks(range(len(dens)), [f"{x}:1" for x in dens])
        ax.set_xlabel("类别不平衡 max:min")
        ax.set_ylabel("密度差异 std_max:std_min")
        ax.set_title(f"SLTFC - RandSamplingLT 的 {metric} 差值")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:+.3f}", ha="center", va="center")
        fig.colorbar(im, ax=ax, label=f"Δ{metric}")
        fig.tight_layout()
        fig.savefig(out / f"gap_{metric}.png", dpi=220)
        plt.close(fig)




def experiment_formal(args) -> pd.DataFrame:
    """
    正式 1M（或用户指定 N）质量实验。

    这里故意冻结开发阶段已经确定的设置，避免继续根据标签结果调参：
      - distribution: hard GMM, d=8, K=10, imbalance=50:1, density_ratio=6:1, anisotropy=2
      - density seed: deterministic density-diverse
      - candidate pool cap: 400,000
      - candidate pool RNG seed: 2026
      - method RNG seed: 2026
      - target subset size: 10,000
      - keep ratio: 0.01
      - compression: hidden_leaf
      - density_k: 15
      - max_subset_factor: 2.0
      - independent data sample seeds: 101,102,103,104,105

    注意：五次重复改变的是“数据采样实现”，而不是算法超参数。
    """
    # ----- Frozen formal configuration -----
    n = int(args.n)
    d = 8
    k = 10
    imbalance = 50.0
    density_ratio = 6.0
    anisotropy = 2.0
    difficulty = "hard"
    candidate_pool_size = min(400_000, n)
    candidate_pool_seed = 2026
    method_seed = 2026
    target_subset_size = 10_000
    keep_ratio = 0.01
    compression = "hidden_leaf"
    density_k = 15
    max_subset_factor = 2.0
    data_seeds = [101, 102, 103, 104, 105]

    out = ensure_dir(Path(args.output) / f"formal_hard_n{n}")
    config = {
        "n": n, "d": d, "k": k,
        "imbalance": imbalance,
        "density_ratio": density_ratio,
        "anisotropy": anisotropy,
        "difficulty": difficulty,
        "candidate_pool_size": candidate_pool_size,
        "candidate_pool_seed": candidate_pool_seed,
        "method_seed": method_seed,
        "target_subset_size": target_subset_size,
        "keep_ratio": keep_ratio,
        "compression": compression,
        "density_k": density_k,
        "max_subset_factor": max_subset_factor,
        "data_seeds": data_seeds,
        "jobs": int(args.jobs),
    }
    with open(out / "formal_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 76)
    print("正式实验：5 个未用于开发调参的数据随机种子")
    print("=" * 76)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print("\n[冻结规则] 本实验不再根据单次结果修改 candidate pool / keep ratio / density_k 等参数。")

    # Same underlying distribution geometry; only the sampled observations change.
    params = make_gmm_params(
        k=k, d=d,
        imbalance_ratio=imbalance,
        density_ratio=density_ratio,
        anisotropy=anisotropy,
        random_state=42,
        difficulty=difficulty,
    )
    np.savez(out / "gmm_params.npz", **params)

    n_subsets = max(100, int(math.ceil(n / target_subset_size)))
    all_rows = []

    for idx, data_seed in enumerate(data_seeds, 1):
        print("\n" + "#" * 76)
        print(f"正式重复 {idx}/{len(data_seeds)}：data_seed={data_seed}")
        print("#" * 76)

        seed_out = ensure_dir(out / f"seed_{data_seed}")
        X, y = generate_gmm_in_memory(
            n, params, noise_fraction=0.0, random_state=data_seed
        )

        df_seed = run_quality_suite(
            X=X,
            y=y,
            k=k,
            out_dir=seed_out,
            n_subsets=n_subsets,
            keep_ratio=keep_ratio,
            compression=compression,
            density_k=density_k,
            n_jobs=args.jobs,
            random_state=method_seed,
            include_minibatch=True,
            max_subset_factor=max_subset_factor,
            density_candidate_size=candidate_pool_size,
            density_seed_method="diverse",
            candidate_pool_seed=candidate_pool_seed,
        )
        df_seed["data_seed"] = int(data_seed)
        df_seed["repeat_index"] = idx
        all_rows.append(df_seed)

        raw_partial = pd.concat(all_rows, ignore_index=True)
        raw_partial.to_csv(
            out / "formal_results_raw_partial.csv",
            index=False, encoding="utf-8-sig"
        )

        del X, y, df_seed
        gc.collect()

    raw = pd.concat(all_rows, ignore_index=True)
    raw.to_csv(out / "formal_results_raw.csv", index=False, encoding="utf-8-sig")

    metrics = ["ACC", "ARI", "NMI", "AMI", "time_sec", "peak_mem_gb"]
    summary = raw.groupby("method", sort=False)[metrics].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "formal_summary.csv", index=False, encoding="utf-8-sig")

    # A compact mean ± std table for terminal output and paper drafting.
    print("\n" + "=" * 92)
    print("正式实验汇总（5个数据随机种子，mean ± std）")
    print("=" * 92)
    for _, r in summary.iterrows():
        print(
            f"{r['method']:<24} "
            f"ACC={r['ACC_mean']:.4f}±{r['ACC_std']:.4f}  "
            f"ARI={r['ARI_mean']:.4f}±{r['ARI_std']:.4f}  "
            f"NMI={r['NMI_mean']:.4f}±{r['NMI_std']:.4f}  "
            f"AMI={r['AMI_mean']:.4f}±{r['AMI_std']:.4f}  "
            f"time={r['time_sec_mean']:.1f}±{r['time_sec_std']:.1f}s"
        )
    print("=" * 92)

    # Paper-friendly error-bar plots.
    for metric in ["ACC", "ARI", "NMI", "AMI", "time_sec", "peak_mem_gb"]:
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        x = np.arange(len(summary))
        means = summary[f"{metric}_mean"].to_numpy(dtype=float)
        stds = summary[f"{metric}_std"].fillna(0).to_numpy(dtype=float)
        ax.bar(x, means, yerr=stds, capsize=4)
        ax.set_xticks(x, summary["method"], rotation=18, ha="right")
        ax.set_ylabel(f"{metric} (mean ± std)")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"formal_{metric}_mean_std.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

    print(f"\n正式结果目录：{out}")
    print("最重要的两个文件：formal_results_raw.csv 和 formal_summary.csv")
    print("请把终端最后的『正式实验汇总（5个数据随机种子，mean ± std）』发给我。")
    return raw




def experiment_formal10m(args) -> pd.DataFrame:
    """
    10M 正式质量实验：3 个独立数据 seed（201/202/203），冻结开发阶段参数。

    特点：
    - 比较 SLTFC-density-diverse / SLTFC-random / RandSamplingLT / MiniBatchKMeans；
    - density/random 记录 partition/local-tree/fusion/propagation 分阶段耗时；
    - 每完成一个方法立即 checkpoint，意外中断后重跑会自动跳过已完成方法；
    - 不根据单次结果继续调 candidate pool / keep ratio / density_k。
    """
    n = int(args.n)
    d = 8
    k = 10
    imbalance = 50.0
    density_ratio = 6.0
    anisotropy = 2.0
    difficulty = "hard"
    candidate_pool_size = min(400_000, n)
    candidate_pool_seed = 2026
    method_seed = 2026
    target_subset_size = 10_000
    keep_ratio = 0.01
    compression = "hidden_leaf"
    density_k = 15
    max_subset_factor = 2.0
    data_seeds = [201, 202, 203]
    query_chunk = 200_000

    out = ensure_dir(Path(args.output) / f"formal10m_hard_n{n}")
    config = {
        "n": n, "d": d, "k": k,
        "imbalance": imbalance,
        "density_ratio": density_ratio,
        "anisotropy": anisotropy,
        "difficulty": difficulty,
        "candidate_pool_size": candidate_pool_size,
        "candidate_pool_seed": candidate_pool_seed,
        "method_seed": method_seed,
        "target_subset_size": target_subset_size,
        "keep_ratio": keep_ratio,
        "compression": compression,
        "density_k": density_k,
        "max_subset_factor": max_subset_factor,
        "data_seeds": data_seeds,
        "jobs": int(args.jobs),
        "query_chunk": query_chunk,
    }
    with open(out / "formal10m_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 96)
    print("10M 正式实验：3 个独立数据 seed；参数冻结；支持按方法断点续跑")
    print("=" * 96)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print("\n[冻结规则] 本实验不再根据结果修改 candidate pool / keep ratio / density_k / seed method。")

    params = make_gmm_params(
        k=k, d=d,
        imbalance_ratio=imbalance,
        density_ratio=density_ratio,
        anisotropy=anisotropy,
        random_state=42,
        difficulty=difficulty,
    )
    np.savez(out / "gmm_params.npz", **params)
    n_subsets = max(100, int(math.ceil(n / target_subset_size)))
    methods_expected = ["SLTFC-density-diverse", "SLTFC-random", "RandSamplingLT", "MiniBatchKMeans"]
    all_seed_frames = []

    def load_seed_checkpoint(seed_out: Path) -> pd.DataFrame:
        fp = seed_out / "results_checkpoint.csv"
        if fp.exists():
            try:
                return pd.read_csv(fp)
            except Exception:
                pass
        return pd.DataFrame()

    def save_seed_checkpoint(seed_out: Path, rows: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df.to_csv(seed_out / "results_checkpoint.csv", index=False, encoding="utf-8-sig")
        return df

    for idx, data_seed in enumerate(data_seeds, 1):
        print("\n" + "#" * 96)
        print(f"10M 正式重复 {idx}/{len(data_seeds)}：data_seed={data_seed}")
        print("#" * 96)
        seed_out = ensure_dir(out / f"seed_{data_seed}")
        existing = load_seed_checkpoint(seed_out)
        completed = set(existing["method"].tolist()) if (not existing.empty and "method" in existing.columns) else set()
        rows = existing.to_dict("records") if not existing.empty else []
        if completed:
            print("  [断点续跑] 已完成：" + ", ".join(sorted(completed)))

        X, y = generate_gmm_in_memory(n, params, noise_fraction=0.0, random_state=data_seed)

        if "SLTFC-density-diverse" not in completed:
            def density_run():
                model = ScalableSLTFC(
                    n_subsets=n_subsets,
                    compression=compression,
                    keep_ratio=keep_ratio,
                    min_representatives=5,
                    n_global_clusters=k,
                    n_jobs=args.jobs,
                    partition_method="density",
                    density_k=density_k,
                    random_state=method_seed,
                    density_candidate_size=candidate_pool_size,
                    query_chunk_size=query_chunk,
                    candidate_pool_seed=candidate_pool_seed,
                    density_seed_method="diverse",
                    max_subset_factor=max_subset_factor,
                    refine_oversized_density=True,
                )
                model.fit(X)
                pred = model.final_labels_.copy()
                stats = dict(getattr(model, "last_partition_stats_", {}))
                stage = dict(getattr(model, "fit_stage_times_", {}))
                extra = {
                    "rep_count": len(model.representatives_),
                    "requested_subsets": n_subsets,
                    **stats, **stage,
                }
                del model
                return pred, extra
            row = timed_method("SLTFC-density-diverse", density_run, y, noise_primary=True)
            row["data_seed"] = data_seed
            rows.append(row)
            save_seed_checkpoint(seed_out, rows)

        if "SLTFC-random" not in completed:
            def random_run():
                model = ScalableSLTFC(
                    n_subsets=n_subsets,
                    compression=compression,
                    keep_ratio=keep_ratio,
                    min_representatives=5,
                    n_global_clusters=k,
                    n_jobs=args.jobs,
                    partition_method="random",
                    density_k=density_k,
                    random_state=method_seed,
                    query_chunk_size=query_chunk,
                )
                model.fit(X)
                pred = model.final_labels_.copy()
                stage = dict(getattr(model, "fit_stage_times_", {}))
                extra = {
                    "rep_count": len(model.representatives_),
                    "requested_subsets": n_subsets,
                    **stage,
                }
                del model
                return pred, extra
            row = timed_method("SLTFC-random", random_run, y, noise_primary=True)
            row["data_seed"] = data_seed
            rows.append(row)
            save_seed_checkpoint(seed_out, rows)

        rep_budget = max(1, int(round(n * keep_ratio)))
        if "RandSamplingLT" not in completed:
            print(f"  [RandSamplingLT 固定预算] sample_size={rep_budget:,} (= N * keep_ratio)")
            row = timed_method(
                "RandSamplingLT",
                lambda: run_random_sampling_lt(X, k, rep_budget, method_seed, query_chunk),
                y,
                noise_primary=True,
            )
            row["data_seed"] = data_seed
            rows.append(row)
            save_seed_checkpoint(seed_out, rows)

        if "MiniBatchKMeans" not in completed:
            row = timed_method(
                "MiniBatchKMeans",
                lambda: run_minibatch_kmeans(X, k, method_seed, 65536, 100, query_chunk),
                y,
                noise_primary=True,
            )
            row["data_seed"] = data_seed
            rows.append(row)
            save_seed_checkpoint(seed_out, rows)

        df_seed = pd.DataFrame(rows)
        # 同一 method 若因人工重复产生多行，保留最后一次。
        if not df_seed.empty:
            df_seed = df_seed.drop_duplicates(subset=["method"], keep="last")
        df_seed.to_csv(seed_out / "results.csv", index=False, encoding="utf-8-sig")
        all_seed_frames.append(df_seed)
        del X, y, df_seed
        gc.collect()

        raw_partial = pd.concat(all_seed_frames, ignore_index=True)
        raw_partial.to_csv(out / "formal10m_results_raw_partial.csv", index=False, encoding="utf-8-sig")

    raw = pd.concat(all_seed_frames, ignore_index=True)
    raw.to_csv(out / "formal10m_results_raw.csv", index=False, encoding="utf-8-sig")

    metrics = ["ACC", "ARI", "NMI", "AMI", "time_sec", "peak_mem_gb"]
    stage_metrics = [
        "partition_time_sec", "seed_selection_time_sec", "local_tree_time_sec",
        "fusion_time_sec", "propagation_time_sec", "fit_total_time_sec",
        "rep_count", "actual_subsets", "max_subset_size", "subset_cv",
    ]
    agg_cols = [c for c in metrics + stage_metrics if c in raw.columns]
    summary = raw.groupby("method", sort=False)[agg_cols].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "formal10m_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 112)
    print("10M 正式实验汇总（3个数据随机种子，mean ± std）")
    print("=" * 112)
    for _, r in summary.iterrows():
        print(
            f"{r['method']:<24} "
            f"ACC={r['ACC_mean']:.4f}±{r['ACC_std']:.4f}  "
            f"ARI={r['ARI_mean']:.4f}±{r['ARI_std']:.4f}  "
            f"NMI={r['NMI_mean']:.4f}±{r['NMI_std']:.4f}  "
            f"AMI={r['AMI_mean']:.4f}±{r['AMI_std']:.4f}  "
            f"time={r['time_sec_mean']:.1f}±{r['time_sec_std']:.1f}s  "
            f"peak={r['peak_mem_gb_mean']:.2f}±{r['peak_mem_gb_std']:.2f}GB"
        )
    print("=" * 112)

    dens = summary[summary["method"] == "SLTFC-density-diverse"]
    if len(dens):
        r = dens.iloc[0]
        pieces = []
        for key, label in [
            ("partition_time_sec_mean", "partition"),
            ("seed_selection_time_sec_mean", "seed_select"),
            ("local_tree_time_sec_mean", "local_tree"),
            ("fusion_time_sec_mean", "fusion"),
            ("propagation_time_sec_mean", "propagation"),
        ]:
            if key in r.index and pd.notna(r[key]):
                pieces.append(f"{label}={r[key]:.1f}s")
        if pieces:
            print("SLTFC-density-diverse 平均分阶段耗时：" + ", ".join(pieces))

    # 轻量图：只画正式论文可能直接使用的几项。
    for metric in ["ACC", "ARI", "NMI", "time_sec", "peak_mem_gb"]:
        if f"{metric}_mean" not in summary.columns:
            continue
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        x = np.arange(len(summary))
        means = summary[f"{metric}_mean"].to_numpy(dtype=float)
        stds = summary[f"{metric}_std"].fillna(0).to_numpy(dtype=float)
        ax.bar(x, means, yerr=stds, capsize=4)
        ax.set_xticks(x, summary["method"], rotation=18, ha="right")
        ax.set_ylabel(f"{metric} (mean ± std)")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"formal10m_{metric}_mean_std.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

    print(f"\n10M 正式结果目录：{out}")
    print("最重要：formal10m_results_raw.csv / formal10m_summary.csv")
    print("如果中途中断，直接重新运行同一条命令，会按方法跳过已经完成的 checkpoint。")
    return raw

def experiment_seedscale(args) -> pd.DataFrame:
    """
    开发/诊断实验：在固定 seed=42 的同一份数据上，对比
    1) density-diverse：逐轮密度×距离贪心，近似 O(C*S)
    2) density-spatial：KD 空间分层 + 区域最高密度，近似 O(C log S)

    目的不是挑分数最高的方法，而是判断可扩展种子选择是否在显著降低
    seed-selection 成本的同时保持可接受的聚类质量。
    """
    out = ensure_dir(Path(args.output) / f"seedscale_{args.difficulty}_n{args.n}")
    params = make_gmm_params(
        k=args.k, d=args.d, imbalance_ratio=args.imbalance,
        density_ratio=args.density_ratio, anisotropy=args.anisotropy,
        random_state=42, difficulty=args.difficulty,
    )
    # 固定开发数据 seed=42，避免使用正式 seed=101~105 做算法修改。
    data_seed = 42
    X, y = generate_gmm_in_memory(args.n, params, noise_fraction=0.0, random_state=data_seed)
    n_subsets = args.n_subsets or max(100, int(math.ceil(args.n / args.target_subset_size)))
    candidate_n = int(args.candidate_pool_size or 400_000)

    print(f"\n[seedscale] 固定开发数据 seed={data_seed}, n={args.n:,}, difficulty={args.difficulty}")
    print(f"  S={n_subsets}, candidate_pool={candidate_n:,}, candidate_pool_seed={args.candidate_pool_seed}")
    print("  对比 diverse(O(C*S)) vs spatial_density(O(C log S))")

    rows = []
    for method, label in [("diverse", "density-diverse"), ("spatial_density", "density-spatial")]:
        def run_one(method=method):
            model = ScalableSLTFC(
                n_subsets=n_subsets,
                compression=args.compression,
                keep_ratio=args.keep_ratio,
                min_representatives=5,
                n_global_clusters=args.k,
                n_jobs=args.jobs,
                partition_method="density",
                density_k=args.density_k,
                random_state=2026,
                density_candidate_size=candidate_n,
                query_chunk_size=args.query_chunk,
                candidate_pool_seed=args.candidate_pool_seed,
                density_seed_method=method,
                exact_density_limit=0,  # seedscale 强制走大规模 density 路径，便于小样本自检
                max_subset_factor=args.max_subset_factor,
                refine_oversized_density=True,
            )
            model.fit(X)
            pred = model.final_labels_.copy()
            ps = getattr(model, "last_partition_stats_", {})
            extra = {
                "rep_count": len(model.representatives_),
                "seed_selection_sec": float(getattr(model, "seed_selection_time_", float("nan"))),
                **ps,
            }
            del model
            return pred, extra

        row = timed_method(label, run_one, y, noise_primary=True)
        row["density_seed_method"] = method
        row["candidate_pool"] = candidate_n
        row["requested_subsets"] = n_subsets
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out / "seedscale_results.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("seedscale 汇总")
    print("=" * 100)
    for _, r in df.iterrows():
        print(
            f"{r['method']:<20} ACC={r['ACC']:.4f} ARI={r['ARI']:.4f} "
            f"NMI={r['NMI']:.4f} AMI={r['AMI']:.4f} "
            f"seed_select={r.get('seed_selection_sec', float('nan')):.3f}s "
            f"total={r['time_sec']:.1f}s max_subset={int(r.get('max_subset_size', -1)):,} "
            f"CV={r.get('subset_cv', float('nan')):.3f}"
        )
    print("=" * 100)
    print(f"结果目录: {out}")
    print("判断重点：看 spatial_density 是否显著降低 seed_select 耗时，同时质量没有明显崩坏。")

    # 简单对比图
    for metric in ["ARI", "NMI", "time_sec", "seed_selection_sec", "subset_cv"]:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6.8, 4.5))
        ax.bar(df["method"], df[metric])
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"seedscale_{metric}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    del X, y
    gc.collect()
    return df


def experiment_density_seedtest(args) -> pd.DataFrame:
    """只运行一次 SLTFC-density，用来测试确定性 density-diverse 选种子。"""
    out = ensure_dir(Path(args.output) / f"density_seedtest_{args.difficulty}_n{args.n}")
    params = make_gmm_params(
        k=args.k, d=args.d, imbalance_ratio=args.imbalance,
        density_ratio=args.density_ratio, anisotropy=args.anisotropy,
        random_state=42, difficulty=args.difficulty,
    )
    X, y = generate_gmm_in_memory(args.n, params, noise_fraction=0.0, random_state=args.seed)
    n_subsets = args.n_subsets or max(100, int(math.ceil(args.n / args.target_subset_size)))
    candidate_n = int(args.candidate_pool_size or 400_000)

    print(f"\n[density种子方法诊断] n={args.n:,}, difficulty={args.difficulty}, S={n_subsets}")
    print(f"  candidate_pool={candidate_n:,}, candidate_pool_seed={args.candidate_pool_seed}")
    print(f"  seed method={args.density_seed_method}")

    def run_one():
        model = ScalableSLTFC(
            n_subsets=n_subsets,
            compression=args.compression,
            keep_ratio=args.keep_ratio,
            min_representatives=5,
            n_global_clusters=args.k,
            n_jobs=args.jobs,
            partition_method="density",
            density_k=args.density_k,
            random_state=args.seed,
            density_candidate_size=candidate_n,
            query_chunk_size=args.query_chunk,
            candidate_pool_seed=args.candidate_pool_seed,
            density_seed_method=args.density_seed_method,
            max_subset_factor=args.max_subset_factor,
            refine_oversized_density=True,
        )
        model.fit(X)
        pred = model.final_labels_.copy()
        extra = {"rep_count": len(model.representatives_)}
        del model
        return pred, extra

    row = timed_method("SLTFC-density", run_one, y, noise_primary=True)
    row["candidate_pool"] = candidate_n
    row["density_seed_method"] = args.density_seed_method
    df = pd.DataFrame([row])
    df.to_csv(out / "density_seedtest_result.csv", index=False, encoding="utf-8-sig")
    print(f"\n结果目录: {out}")
    del X, y
    gc.collect()
    return df


def experiment_density_stability(args) -> pd.DataFrame:
    """
    固定同一份合成数据，只改变 density 分区随机种子，并同时比较多个候选池大小。
    用于判断单次结果波动究竟来自候选池大小，还是来自 density 加权随机选种子。
    """
    out = ensure_dir(Path(args.output) / f"density_stability_{args.difficulty}_n{args.n}")
    params = make_gmm_params(
        k=args.k,
        d=args.d,
        imbalance_ratio=args.imbalance,
        density_ratio=args.density_ratio,
        anisotropy=args.anisotropy,
        random_state=42,
        difficulty=args.difficulty,
    )
    # 数据固定：始终只用 args.seed 生成一次
    X, y = generate_gmm_in_memory(args.n, params, noise_fraction=0.0, random_state=args.seed)
    n_subsets = args.n_subsets or max(100, int(math.ceil(args.n / args.target_subset_size)))

    print(
        f"\n[density 稳定性诊断] 固定数据 seed={args.seed}, n={args.n:,}, "
        f"difficulty={args.difficulty}, S={n_subsets}"
    )
    print(f"  candidate pools = {args.candidate_pool_sizes}")
    print(f"  candidate pool seed 固定 = {args.candidate_pool_seed}")
    print(f"  weighted-seed RNG seeds = {args.partition_seeds}")
    print("  只运行 SLTFC-density；候选池组成固定，仅改变密度加权种子抽样随机性。")

    rows = []
    for candidate_n in args.candidate_pool_sizes:
        candidate_n = min(int(candidate_n), len(X))
        for pseed in args.partition_seeds:
            print(f"\n========== candidate={candidate_n:,}, partition_seed={pseed} ==========")

            def run_one(candidate_n=candidate_n, pseed=int(pseed)):
                model = ScalableSLTFC(
                    n_subsets=n_subsets,
                    compression=args.compression,
                    keep_ratio=args.keep_ratio,
                    min_representatives=5,
                    n_global_clusters=args.k,
                    n_jobs=args.jobs,
                    partition_method="density",
                    density_k=args.density_k,
                    random_state=pseed,  # 只改变分区随机性；数据本身不变
                    density_candidate_size=candidate_n,
                    query_chunk_size=args.query_chunk,
                    candidate_pool_seed=args.candidate_pool_seed,
                    density_seed_method=args.density_seed_method,
                    max_subset_factor=args.max_subset_factor,
                    refine_oversized_density=True,
                )
                model.fit(X)
                pred = model.final_labels_.copy()
                extra = {
                    "rep_count": len(model.representatives_),
                    "n_subsets_requested": n_subsets,
                }
                del model
                return pred, extra

            row = timed_method("SLTFC-density", run_one, y, noise_primary=True)
            row["candidate_pool"] = int(candidate_n)
            row["partition_seed"] = int(pseed)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out / "density_stability_raw.csv", index=False, encoding="utf-8-sig")

    metrics = [m for m in ["ACC", "ARI", "NMI", "AMI", "time_sec", "peak_mem_gb"] if m in df.columns]
    agg = df.groupby("candidate_pool")[metrics].agg(["mean", "std"]).reset_index()
    # Flatten MultiIndex columns for an easy-to-read CSV
    agg.columns = ["candidate_pool" if c[0] == "candidate_pool" else f"{c[0]}_{c[1]}" for c in agg.columns]
    agg.to_csv(out / "density_stability_summary.csv", index=False, encoding="utf-8-sig")

    print("\n[density 稳定性汇总：mean ± std]")
    for _, r in agg.iterrows():
        cp = int(r["candidate_pool"])
        parts = [f"candidate={cp:,}"]
        for m in ["ACC", "ARI", "NMI", "AMI", "time_sec"]:
            mk, sk = f"{m}_mean", f"{m}_std"
            if mk in agg.columns:
                parts.append(f"{m}={r[mk]:.4f}±{r[sk]:.4f}")
        print("  " + ", ".join(parts))

    # Plot mean ± std for the clustering metrics
    for metric in ["ACC", "ARI", "NMI", "AMI"]:
        mk, sk = f"{metric}_mean", f"{metric}_std"
        if mk not in agg.columns:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.errorbar(agg["candidate_pool"], agg[mk], yerr=agg[sk], marker="o", capsize=4)
        ax.set_xlabel("density candidate pool size")
        ax.set_ylabel(f"{metric} (mean ± std)")
        ax.grid(True, alpha=0.25)
        ax.ticklabel_format(style="plain", axis="x")
        fig.tight_layout()
        fig.savefig(out / f"density_stability_{metric}.png", dpi=220)
        plt.close(fig)

    print(f"\n结果目录: {out}")
    del X, y
    gc.collect()
    return df


def experiment_calibrate(args) -> pd.DataFrame:
    """
    小规模难度标定：固定 n（建议 100k），依次跑 easy/medium/hard，
    用来找到“不全是 1.0、也不是全部接近 0”的主实验难度。
    """
    out = ensure_dir(Path(args.output) / "calibration_easy_medium_hard")
    rows = []
    original_difficulty = args.difficulty
    print(f"\n[难度标定] n={args.n:,}; 依次测试 easy / medium / hard")
    for level in ["easy", "medium", "hard"]:
        print(f"\n========== difficulty={level} ==========")
        params = make_gmm_params(
            k=args.k, d=args.d, imbalance_ratio=args.imbalance,
            density_ratio=args.density_ratio, anisotropy=args.anisotropy,
            random_state=42, difficulty=level,
        )
        X, y = generate_gmm_in_memory(args.n, params, 0.0, args.seed)
        n_subsets = args.n_subsets or max(20, int(math.ceil(args.n / args.target_subset_size)))
        level_out = ensure_dir(out / level)
        df = run_quality_suite(
            X, y, args.k, level_out, n_subsets, args.keep_ratio,
            args.compression, args.density_k, args.jobs, args.seed,
            max_subset_factor=args.max_subset_factor,
            density_candidate_size=args.candidate_pool_size,
        )
        df["difficulty"] = level
        rows.append(df)
        del X, y
        gc.collect()

    args.difficulty = original_difficulty
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(out / "calibration_results.csv", index=False, encoding="utf-8-sig")

    order = ["easy", "medium", "hard"]
    for metric in ["ARI", "NMI", "ACC", "time_sec"]:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for method, g in all_df.groupby("method"):
            vals = []
            for level in order:
                q = g[g.difficulty == level]
                vals.append(float(q.iloc[0][metric]) if len(q) else np.nan)
            ax.plot(order, vals, marker="o", label=method)
        ax.set_xlabel("difficulty")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"calibration_{metric}.png", dpi=220)
        plt.close(fig)
    return all_df


# ============================================================
# 6. 快速自检
# ============================================================

def quick_test(args):
    print("\n=== QUICK TEST：先用小数据验证整套代码能跑通 ===")
    args.n = 20_000
    args.k = 6
    args.d = 4
    args.imbalance = 10
    args.density_ratio = 3
    args.n_subsets = 20
    args.keep_ratio = 0.05
    args.target_subset_size = 1000
    args.jobs = min(args.jobs, 2) if args.jobs > 0 else 2
    args.compression = "hidden_leaf"
    args.difficulty = "easy"   # quick 只验证代码链路，不用于性能比较
    args.max_subset_factor = 2.0
    return experiment_1(args)


# ============================================================
# 7. CLI
# ============================================================

def parse_int_list(s: str) -> List[int]:
    mult = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    out = []
    for p in s.split(","):
        p = p.strip().lower()
        if not p:
            continue
        if p[-1:] in mult:
            out.append(int(float(p[:-1]) * mult[p[-1]]))
        else:
            out.append(int(p))
    return out


def build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--experiment", choices=["quick", "calibrate", "formal", "formal10m", "seedscale", "seedtest", "stability", "1", "2", "3", "4", "all"], default="quick")
    p.add_argument("--output", default="big_experiment_results")
    p.add_argument("--n", type=int, default=10_000_000, help="实验1/2/4的样本数")
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--imbalance", type=float, default=50.0)
    p.add_argument("--density-ratio", type=float, default=6.0)
    p.add_argument("--anisotropy", type=float, default=2.0)
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium",
                   help="合成 GMM 的簇间重叠难度；正式质量实验建议先用 calibrate 标定")
    p.add_argument("--max-subset-factor", type=float, default=2.0,
                   help="density Voronoi 子集容量上限 = factor × N/请求子集数；超限单元用 KD 中位数继续细分")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jobs", type=int, default=4, help="建议先4，不要一上来 -1")
    p.add_argument("--n-subsets", type=int, default=None)
    p.add_argument("--target-subset-size", type=int, default=10_000)
    p.add_argument("--keep-ratio", type=float, default=0.01)
    p.add_argument("--compression", choices=["hidden_leaf", "structure", "conv"], default="hidden_leaf")
    p.add_argument("--density-k", type=int, default=15)
    p.add_argument("--query-chunk", type=int, default=200_000)
    p.add_argument("--candidate-pool-size", type=int, default=None,
                   help="大规模 density 候选池大小，例如 100000、200000、400000、800000")
    p.add_argument("--candidate-pool-sizes", type=parse_int_list,
                   default=parse_int_list("400k,800k"),
                   help="stability 实验要比较的候选池，例如 400k,800k")
    p.add_argument("--partition-seeds", type=parse_int_list,
                   default=parse_int_list("1,2,3"),
                   help="stability 实验使用的密度种子抽样随机种子，例如 1,2,3")
    p.add_argument("--candidate-pool-seed", type=int, default=2026,
                   help="固定候选池组成的独立随机种子；stability 中保持不变")
    p.add_argument("--density-seed-method", choices=["weighted_random", "diverse", "spatial_density"], default="weighted_random",
                   help="density Voronoi 种子选择：原始密度加权随机 / 确定性密度-距离多样性")

    # scaling
    p.add_argument("--scale-sizes", type=parse_int_list, default=parse_int_list("1m,5m,10m,50m,100m"))
    p.add_argument("--scaling-subset-size", type=int, default=20_000)
    p.add_argument("--scaling-keep-ratio", type=float, default=0.001)
    p.add_argument("--scaling-compression", choices=["hidden_leaf", "structure", "conv"], default="hidden_leaf")

    # stress grid
    p.add_argument("--imbalance-levels", type=lambda s: [float(x) for x in s.split(",")],
                   default=[1.0, 10.0, 100.0])
    p.add_argument("--density-levels", type=lambda s: [float(x) for x in s.split(",")],
                   default=[1.0, 3.0, 6.0])
    return p


def main():
    args = build_parser().parse_args()
    print("\nSLTFC 大规模实验脚本")
    print(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str))
    ensure_dir(args.output)

    if args.experiment == "quick":
        quick_test(args)
        return
    if args.experiment == "calibrate":
        experiment_calibrate(args)
        return
    if args.experiment == "formal":
        experiment_formal(args)
        return
    if args.experiment == "formal10m":
        experiment_formal10m(args)
        return
    if args.experiment == "seedscale":
        experiment_seedscale(args)
        return
    if args.experiment == "seedtest":
        experiment_density_seedtest(args)
        return
    if args.experiment == "stability":
        experiment_density_stability(args)
        return
    if args.experiment in ["1", "all"]:
        experiment_1(args)
    if args.experiment in ["2", "all"]:
        experiment_2(args)
    if args.experiment in ["3", "all"]:
        experiment_3(args)
    if args.experiment in ["4", "all"]:
        experiment_4(args)


if __name__ == "__main__":
    main()
