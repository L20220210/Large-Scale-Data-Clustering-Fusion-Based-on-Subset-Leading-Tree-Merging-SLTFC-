import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from subset_leading_tree_fusion import LeadingTree, configure_chinese_font


warnings.filterwarnings("ignore")
configure_chinese_font()


def ensure_continuous_labels(labels):
    unique = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique)}
    return np.array([mapping[x] for x in labels], dtype=int)


def cluster_accuracy(true_labels, pred_labels):
    true_labels = ensure_continuous_labels(true_labels)
    pred_labels = ensure_continuous_labels(pred_labels)
    n_classes = max(len(np.unique(true_labels)), len(np.unique(pred_labels)))
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        confusion[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    return confusion[row_ind, col_ind].sum() / len(true_labels)


def evaluate_clustering(true_labels, pred_labels):
    return {
        "准确率 (ACC)": cluster_accuracy(true_labels, pred_labels),
        "调整兰德指数 (ARI)": adjusted_rand_score(true_labels, pred_labels),
        "标准化互信息 (NMI)": normalized_mutual_info_score(true_labels, pred_labels),
        "调整互信息 (AMI)": adjusted_mutual_info_score(true_labels, pred_labels),
        "最终簇数": len(np.unique(pred_labels)),
    }


def load_excel_datasets(file_path):
    datasets = {}
    excel_file = pd.ExcelFile(file_path)
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        if df.shape[1] < 3:
            print(f"跳过 {sheet_name}: 至少需要两列特征和一列标签")
            continue
        X = df.iloc[:, :-1].to_numpy(dtype=float)
        y = df.iloc[:, -1].to_numpy()
        valid_mask = ~(np.any(pd.isna(X), axis=1) | pd.isna(y))
        X = X[valid_mask]
        y = y[valid_mask].astype(int)
        datasets[sheet_name] = {"data": X, "target": y}
        print(f"{sheet_name}: X={X.shape}, 类别数={len(np.unique(y))}")
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

    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"{dataset_name}: X={X.shape}, 类别数={len(np.unique(y))}")
    return {dataset_name: {"data": X, "target": y}}


def load_tabular_datasets(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        return load_excel_datasets(file_path)
    if ext == ".csv":
        return load_csv_dataset(file_path)
    raise ValueError(f"不支持的文件类型: {ext}")


def maybe_subsample_for_fit(X, max_samples=None, random_state=42):
    if max_samples is None or X.shape[0] <= max_samples:
        return X, np.arange(X.shape[0])
    rng = np.random.default_rng(random_state)
    indices = rng.choice(X.shape[0], size=max_samples, replace=False)
    return X[indices], indices


def propagate_labels(X_fit, fit_labels, X_all):
    nearest = NearestNeighbors(n_neighbors=1).fit(X_fit)
    _, indices = nearest.kneighbors(X_all)
    return fit_labels[indices[:, 0]]


class OriginalDPC:
    """Original Density Peaks Clustering with Gaussian local density."""

    def __init__(self, n_clusters, dc_percentile=2.0, random_state=42):
        self.n_clusters = n_clusters
        self.dc_percentile = dc_percentile
        self.random_state = random_state
        self.labels_ = None
        self.centers_ = None
        self.rho_ = None
        self.delta_ = None
        self.gamma_ = None

    def fit(self, X):
        X = np.asarray(X)
        distances = cdist(X, X)
        upper = distances[np.triu_indices_from(distances, k=1)]
        dc = np.percentile(upper, self.dc_percentile)
        dc = max(dc, 1e-12)

        self.rho_ = np.sum(np.exp(-((distances / dc) ** 2)), axis=1) - 1.0
        self.delta_, parent = compute_delta_parent_from_distances(distances, self.rho_)
        self.gamma_ = self.rho_ * self.delta_
        self.centers_ = np.argsort(-self.gamma_)[: self.n_clusters]
        self.labels_ = assign_density_peak_labels(self.rho_, parent, self.centers_)
        return self


class KNNDPC:
    """KNN-DPC: local density is estimated by k-nearest-neighbor distance."""

    def __init__(self, n_clusters, k=10, random_state=42):
        self.n_clusters = n_clusters
        self.k = k
        self.random_state = random_state
        self.labels_ = None
        self.centers_ = None
        self.rho_ = None
        self.delta_ = None
        self.gamma_ = None

    def fit(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        k_eff = min(self.k + 1, n_samples)
        nbrs = NearestNeighbors(n_neighbors=k_eff).fit(X)
        knn_distances, _ = nbrs.kneighbors(X)
        self.rho_ = 1.0 / (knn_distances[:, -1] + 1e-8)

        distances = cdist(X, X)
        self.delta_, parent = compute_delta_parent_from_distances(distances, self.rho_)
        self.gamma_ = self.rho_ * self.delta_
        self.centers_ = np.argsort(-self.gamma_)[: self.n_clusters]
        self.labels_ = assign_density_peak_labels(self.rho_, parent, self.centers_)
        return self


class FullLeadingTreeBaseline:
    """Full-data Leading Tree baseline."""

    def __init__(self, n_clusters, dc=None):
        self.n_clusters = n_clusters
        self.dc = dc
        self.model_ = None
        self.labels_ = None

    def fit(self, X):
        self.model_ = LeadingTree(dc=self.dc, min_centers=self.n_clusters).fit(X)
        centers = np.argsort(-self.model_.gamma)[: self.n_clusters]
        self.model_.centers = np.asarray(centers, dtype=int)
        self.model_.tree = self.model_._build_tree(X, self.model_.parent_, self.model_.centers)
        self.labels_ = self.model_._assign_labels(self.model_.tree, self.model_.centers)
        return self


class RandomSamplingLeadingTreeBaseline:
    """Random sampling plus Leading Tree, then nearest-representative label propagation."""

    def __init__(self, n_clusters, sample_ratio=0.05, sample_size=None, random_state=42, dc=None):
        self.n_clusters = n_clusters
        self.sample_ratio = sample_ratio
        self.sample_size = sample_size
        self.random_state = random_state
        self.dc = dc
        self.sample_indices_ = None
        self.sample_model_ = None
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X)
        rng = np.random.default_rng(self.random_state)
        if self.sample_size is None:
            n_sample = max(self.n_clusters * 20, int(np.ceil(X.shape[0] * self.sample_ratio)))
        else:
            n_sample = self.sample_size
        n_sample = min(max(n_sample, self.n_clusters), X.shape[0])
        self.sample_indices_ = rng.choice(X.shape[0], size=n_sample, replace=False)
        X_sample = X[self.sample_indices_]

        self.sample_model_ = FullLeadingTreeBaseline(n_clusters=self.n_clusters, dc=self.dc).fit(X_sample)
        self.labels_ = propagate_labels(X_sample, self.sample_model_.labels_, X)
        return self


def compute_delta_parent_from_distances(distances, rho):
    n_samples = len(rho)
    delta = np.zeros(n_samples, dtype=float)
    parent = np.arange(n_samples, dtype=int)
    order = np.argsort(-rho)

    for rank, idx in enumerate(order):
        if rank == 0:
            delta[idx] = np.max(distances[idx])
            parent[idx] = idx
            continue

        higher = order[:rank]
        higher_distances = distances[idx, higher]
        nearest_pos = np.argmin(higher_distances)
        delta[idx] = higher_distances[nearest_pos]
        parent[idx] = higher[nearest_pos]

    return delta, parent


def assign_density_peak_labels(rho, parent, centers):
    labels = np.full(len(rho), -1, dtype=int)
    for label, center in enumerate(centers):
        labels[int(center)] = label

    for idx in np.argsort(-rho):
        if labels[idx] >= 0:
            continue
        labels[idx] = labels[parent[idx]]
        if labels[idx] < 0:
            labels[idx] = 0

    return labels


def plot_comparison(X, true_labels, pred_labels, title, save_path):
    X = np.asarray(X)
    if X.shape[1] > 2:
        X_plot = PCA(n_components=2, random_state=42).fit_transform(X)
        x_label = "主成分1"
        y_label = "主成分2"
    else:
        X_plot = X
        x_label = "X坐标"
        y_label = "Y坐标"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    axes[0].scatter(X_plot[:, 0], X_plot[:, 1], c=true_labels, cmap="tab10", s=20, alpha=0.85)
    axes[0].set_title("真实标签")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(X_plot[:, 0], X_plot[:, 1], c=pred_labels, cmap="tab10", s=20, alpha=0.85)
    axes[1].set_title("预测标签")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=14)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_results_table(results):
    print("\n" + "=" * 116)
    print("Baseline 聚类算法性能总结")
    print("=" * 116)
    print(
        f"{'数据集':<14}"
        f"{'方法':<28}"
        f"{'ACC':>10}"
        f"{'ARI':>10}"
        f"{'NMI':>10}"
        f"{'AMI':>10}"
        f"{'最终簇数':>10}"
        f"{'训练样本数':>12}"
        f"{'耗时(秒)':>12}"
    )
    print("-" * 116)
    for row in results:
        print(
            f"{row['数据集']:<14}"
            f"{row['方法']:<28}"
            f"{row['准确率 (ACC)']:>10.4f}"
            f"{row['调整兰德指数 (ARI)']:>10.4f}"
            f"{row['标准化互信息 (NMI)']:>10.4f}"
            f"{row['调整互信息 (AMI)']:>10.4f}"
            f"{int(row['最终簇数']):>10}"
            f"{int(row['训练样本数']):>12}"
            f"{row['耗时(秒)']:>12.4f}"
        )
    print("=" * 116)


def make_baseline(method_name, n_clusters, random_state=42, sample_ratio=0.05, sample_size=None):
    if method_name == "OriginalDPC":
        return OriginalDPC(n_clusters=n_clusters, dc_percentile=2.0, random_state=random_state)
    if method_name == "KNNDPC":
        return KNNDPC(n_clusters=n_clusters, k=10, random_state=random_state)
    if method_name == "FullLeadingTree":
        return FullLeadingTreeBaseline(n_clusters=n_clusters)
    if method_name == "RandomSamplingLeadingTree":
        return RandomSamplingLeadingTreeBaseline(
            n_clusters=n_clusters,
            sample_ratio=sample_ratio,
            sample_size=sample_size,
            random_state=random_state,
        )
    raise ValueError(f"Unknown baseline method: {method_name}")


def run_baseline_excel_experiment(
    file_path,
    methods=None,
    output_dir="results_baselines",
    max_quadratic_samples=12000,
    sample_ratio=0.05,
    sample_size=None,
    random_state=42,
):
    if methods is None:
        methods = ["OriginalDPC", "KNNDPC", "FullLeadingTree", "RandomSamplingLeadingTree"]

    os.makedirs(output_dir, exist_ok=True)
    datasets = load_tabular_datasets(file_path)
    rows = []

    for dataset_name, dataset in datasets.items():
        X_raw = dataset["data"]
        y = dataset["target"]
        X = StandardScaler().fit_transform(X_raw)
        n_clusters = len(np.unique(y))

        for method_name in methods:
            print(f"\n开始处理: {dataset_name} / {method_name}")
            start = time.time()

            model = make_baseline(
                method_name,
                n_clusters=n_clusters,
                random_state=random_state,
                sample_ratio=sample_ratio,
                sample_size=sample_size,
            )

            if method_name in {"OriginalDPC", "KNNDPC", "FullLeadingTree"}:
                X_fit, fit_indices = maybe_subsample_for_fit(
                    X,
                    max_samples=max_quadratic_samples,
                    random_state=random_state,
                )
                if len(fit_indices) < X.shape[0]:
                    print(
                        f"  {method_name} 是二次复杂度方法，使用 {len(fit_indices)} 个样本拟合后传播标签。"
                    )
                model.fit(X_fit)
                pred_labels = propagate_labels(X_fit, model.labels_, X)
                train_size = len(fit_indices)
            else:
                model.fit(X)
                pred_labels = model.labels_
                train_size = len(model.sample_indices_)

            metrics = evaluate_clustering(y, pred_labels)
            metrics["数据集"] = dataset_name
            metrics["方法"] = method_name
            metrics["训练样本数"] = train_size
            metrics["耗时(秒)"] = time.time() - start
            rows.append(metrics)

            method_dir = os.path.join(output_dir, method_name)
            os.makedirs(method_dir, exist_ok=True)
            plot_comparison(
                X_raw,
                y,
                pred_labels,
                title=f"{dataset_name} - {method_name}",
                save_path=os.path.join(method_dir, f"{dataset_name}_{method_name}_comparison.png"),
            )

            for key, value in metrics.items():
                if key in {"数据集", "方法"}:
                    continue
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    summary = pd.DataFrame(rows)
    metric_order = [
        "数据集",
        "方法",
        "准确率 (ACC)",
        "调整兰德指数 (ARI)",
        "标准化互信息 (NMI)",
        "调整互信息 (AMI)",
        "最终簇数",
        "训练样本数",
        "耗时(秒)",
    ]
    summary = summary[metric_order]
    summary.to_excel(os.path.join(output_dir, "baseline_summary.xlsx"), index=False)
    summary.to_csv(os.path.join(output_dir, "baseline_summary.csv"), index=False, encoding="utf-8-sig")
    print_results_table(rows)
    print(f"\nBaseline 结果已保存到: {output_dir}")
    return summary


if __name__ == "__main__":
    run_baseline_excel_experiment(
        file_path="clustering_datasets.xlsx",
        output_dir="results_cb",
        max_quadratic_samples=12000,
        sample_ratio=1,
    )
