import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Windows 常用中文字体。字体缺失只影响画图，不影响聚类结果。
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


class KMeansClustering:
    """K-means 聚类算法封装。"""

    def __init__(
        self,
        n_clusters=3,
        random_state=42,
        use_minibatch="auto",
        batch_size=8192,
        n_init=3,
        max_iter=100,
    ):
        self.n_clusters = int(n_clusters)
        self.random_state = random_state
        self.use_minibatch = use_minibatch
        self.batch_size = batch_size
        self.n_init = n_init
        self.max_iter = max_iter
        self.model = None
        self.labels_ = None

    def fit(self, X):
        use_minibatch = self.use_minibatch == "auto" and len(X) > 100000
        use_minibatch = use_minibatch or self.use_minibatch is True

        if use_minibatch:
            print(f"  使用 MiniBatchKMeans 加速: batch_size={self.batch_size}, n_init={self.n_init}, max_iter={self.max_iter}")
            self.model = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                batch_size=self.batch_size,
                n_init=self.n_init,
                max_iter=self.max_iter,
                reassignment_ratio=0.01,
            )
        else:
            print(f"  使用 KMeans: n_init={self.n_init}, max_iter={self.max_iter}")
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
                algorithm="elkan",
            )

        self.labels_ = self.model.fit_predict(X)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("模型尚未拟合，请先调用 fit 方法")
        return self.model.predict(X)

    def evaluate(self, X, true_labels):
        """评估聚类质量。

        注意：ACC 在类别极不平衡时可能虚高，所以必须结合 ARI/NMI/AMI 一起看。
        """
        if self.labels_ is None:
            raise ValueError("模型尚未拟合，请先调用 fit 方法")

        ari = adjusted_rand_score(true_labels, self.labels_)
        nmi = normalized_mutual_info_score(true_labels, self.labels_)
        ami = adjusted_mutual_info_score(true_labels, self.labels_)
        acc = self._cluster_accuracy(true_labels, self.labels_)

        metrics = {
            "准确率 (ACC)": acc,
            "调整兰德指数 (ARI)": ari,
            "标准化互信息 (NMI)": nmi,
            "调整互信息 (AMI)": ami,
        }

        n_pred = len(np.unique(self.labels_))
        if n_pred > 1 and n_pred < len(X):
            # silhouette_score 在几十万样本上非常慢，这里只抽样计算。
            sample_size = min(10000, len(X))
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(len(X), size=sample_size, replace=False)
            metrics["轮廓系数抽样 (Silhouette)"] = silhouette_score(X[sample_idx], self.labels_[sample_idx])
            metrics["CH指数 (Calinski-Harabasz)"] = calinski_harabasz_score(X, self.labels_)
            metrics["DB指数 (Davies-Bouldin)"] = davies_bouldin_score(X, self.labels_)

        return metrics

    @staticmethod
    def _cluster_accuracy(true_labels, pred_labels):
        """用匈牙利算法计算聚类 ACC。"""
        true_labels = KMeansClustering._ensure_continuous_labels(true_labels)
        pred_labels = KMeansClustering._ensure_continuous_labels(pred_labels)

        n_classes = max(len(np.unique(true_labels)), len(np.unique(pred_labels)))
        confusion = np.zeros((n_classes, n_classes), dtype=np.int64)

        for true_label, pred_label in zip(true_labels, pred_labels):
            confusion[true_label, pred_label] += 1

        row_ind, col_ind = linear_sum_assignment(-confusion)
        return confusion[row_ind, col_ind].sum() / len(true_labels)

    @staticmethod
    def _ensure_continuous_labels(labels):
        return LabelEncoder().fit_transform(labels)

    def plot_clusters(self, X_plot, true_labels=None, title="K-means 聚类结果", save_path=None, max_points=20000):
        """绘制前两个特征上的聚类结果。"""
        if self.labels_ is None:
            raise ValueError("模型尚未拟合，请先调用 fit 方法")

        if X_plot.shape[1] < 2:
            print("特征数少于 2，跳过画图")
            return

        if len(X_plot) > max_points:
            rng = np.random.default_rng(42)
            plot_idx = rng.choice(len(X_plot), size=max_points, replace=False)
            X_plot = X_plot[plot_idx]
            pred_labels = self.labels_[plot_idx]
            true_labels = true_labels[plot_idx] if true_labels is not None else None
        else:
            pred_labels = self.labels_

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        if true_labels is not None:
            plt.scatter(X_plot[:, 0], X_plot[:, 1], c=true_labels, cmap="tab20", s=8, alpha=0.7)
        plt.title("真实标签")
        plt.xlabel("特征1")
        plt.ylabel("特征2")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.scatter(X_plot[:, 0], X_plot[:, 1], c=pred_labels, cmap="tab20", s=8, alpha=0.7)
        plt.title(title)
        plt.xlabel("特征1")
        plt.ylabel("特征2")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


class KMeansOptimizer:
    """基于外部标签指标选择聚类数。

    如果你的目标是无监督聚类，请不要用真实标签优化；这里主要用于实验对比。
    """

    def __init__(self, X, y, min_clusters=2, max_clusters=20, metric="ari"):
        self.X = X
        self.y = y
        self.min_clusters = int(min_clusters)
        self.max_clusters = int(max_clusters)
        self.metric = metric.lower()

    def optimize(self):
        print(f"按 {self.metric.upper()} 搜索 K-means 最优聚类数...")

        best_k = None
        best_score = -np.inf
        rows = []

        for k in range(self.min_clusters, self.max_clusters + 1):
            model = KMeansClustering(n_clusters=k)
            model.fit(self.X)
            metrics = model.evaluate(self.X, self.y)

            score_map = {
                "acc": metrics["准确率 (ACC)"],
                "ari": metrics["调整兰德指数 (ARI)"],
                "nmi": metrics["标准化互信息 (NMI)"],
                "ami": metrics["调整互信息 (AMI)"],
            }
            score = score_map[self.metric]
            rows.append((k, score, metrics["准确率 (ACC)"], metrics["调整兰德指数 (ARI)"], metrics["标准化互信息 (NMI)"], metrics["调整互信息 (AMI)"]))

            if score > best_score:
                best_k = k
                best_score = score

        print(f"{'k':<6} {'优化指标':<10} {'ACC':<10} {'ARI':<10} {'NMI':<10} {'AMI':<10}")
        print("-" * 62)
        for k, score, acc, ari, nmi, ami in rows:
            print(f"{k:<6} {score:<10.4f} {acc:<10.4f} {ari:<10.4f} {nmi:<10.4f} {ami:<10.4f}")

        print(f"K-means 最优参数: n_clusters={best_k}")
        print(f"最优 {self.metric.upper()}: {best_score:.4f}")
        return best_k, best_score


def load_excel_datasets(file_path):
    """从 Excel 文件加载数据集。

    默认约定：最后一列是标签，其余列是特征。
    非数值特征会自动做 one-hot 编码，标签会自动编码为整数。
    """
    datasets = {}

    try:
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names

        print(f"找到 {len(sheet_names)} 个 sheet: {sheet_names}")

        for sheet_name in sheet_names:
            print(f"加载 sheet: {sheet_name}")
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            if df.shape[1] < 3:
                print(f"警告: {sheet_name} 数据列数不足 3 列，跳过")
                continue

            df = df.dropna(axis=0)
            if df.empty:
                print(f"警告: {sheet_name} 清理缺失值后为空，跳过")
                continue

            X_df = df.iloc[:, :-1]
            y_raw = df.iloc[:, -1].values

            X_df = pd.get_dummies(X_df, dummy_na=False)
            X_df = X_df.apply(pd.to_numeric, errors="coerce")
            valid_mask = ~X_df.isna().any(axis=1)

            X = X_df.loc[valid_mask].values.astype(float)
            y = LabelEncoder().fit_transform(y_raw[valid_mask])

            datasets[sheet_name] = {
                "data": X,
                "plot_data": X[:, :2],
                "target": y,
                "feature_names": list(X_df.columns),
                "target_names": [f"类别{i}" for i in np.unique(y)],
            }

            print(f"  - 数据形状: {X.shape}, 类别数: {len(np.unique(y))}")
            print_class_distribution(y)

    except Exception as e:
        print(f"加载 Excel 文件时出错: {e}")
        import traceback

        traceback.print_exc()

    return datasets


def print_class_distribution(y, top_n=10):
    """打印类别分布，帮助判断 ACC 是否可能虚高。"""
    labels, counts = np.unique(y, return_counts=True)
    order = np.argsort(counts)[::-1]
    total = len(y)

    print("  - 类别分布 Top:")
    for idx in order[:top_n]:
        print(f"    类别 {labels[idx]}: {counts[idx]} ({counts[idx] / total:.4%})")
    if len(labels) > top_n:
        print(f"    ... 其余 {len(labels) - top_n} 个类别")

    majority_ratio = counts.max() / total
    if majority_ratio > 0.5:
        print(f"  - 提醒: 最大类别占比 {majority_ratio:.4%}，ACC 很容易虚高，请重点看 ARI/NMI/AMI")


def run_kmeans_datasets(
    file_path,
    use_optimization=False,
    optimize_metric="ari",
    max_clusters=None,
    make_plot=True,
):
    """测试 K-means 在所有数据集上的表现。"""
    print("K-means 聚类算法评估")
    print("=" * 80)

    datasets = load_excel_datasets(file_path)
    if not datasets:
        print("无法加载 Excel 数据集")
        return {}

    results = {}

    for name, dataset in datasets.items():
        start = time.time()
        print(f"\n{'=' * 50}")
        print(f"测试 {name} 数据集")
        print(f"数据形状: {dataset['data'].shape}, 类别数: {len(np.unique(dataset['target']))}")
        print(f"{'=' * 50}")

        X, y = dataset["data"], dataset["target"]
        X_scaled = StandardScaler().fit_transform(X)
        true_n_classes = len(np.unique(y))

        if use_optimization:
            upper = max_clusters or min(30, true_n_classes + 5)
            optimizer = KMeansOptimizer(
                X_scaled,
                y,
                min_clusters=2,
                max_clusters=upper,
                metric=optimize_metric,
            )
            n_clusters, _ = optimizer.optimize()
        else:
            n_clusters = true_n_classes
            print(f"使用真实类别数作为聚类数: n_clusters={n_clusters}")

        try:
            model = KMeansClustering(n_clusters=n_clusters)
            model.fit(X_scaled)
            metrics = model.evaluate(X_scaled, y)
            metrics["耗时(秒)"] = time.time() - start
            results[name] = metrics

            print(f"{name} 数据集聚类结果:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

            if make_plot:
                model.plot_clusters(
                    dataset["plot_data"],
                    true_labels=y,
                    title=f"K-means 聚类结果 (k={n_clusters})",
                    save_path=f"{name}_kmeans_fixed.png",
                )

        except Exception as e:
            print(f"处理 {name} 数据集时出错: {e}")
            results[name] = None

    return results


def print_results_summary(results):
    """打印结果总结。"""
    print("\n" + "=" * 80)
    print("K-means 聚类算法性能总结")
    print("=" * 80)

    if not results:
        print("没有有效的结果可显示")
        return

    print(f"{'数据集':<18} {'ACC':<10} {'ARI':<10} {'NMI':<10} {'AMI':<10} {'耗时(秒)':<10}")
    print("-" * 74)

    for name, metrics in results.items():
        if metrics:
            print(
                f"{name:<18} "
                f"{metrics['准确率 (ACC)']:<10.4f} "
                f"{metrics['调整兰德指数 (ARI)']:<10.4f} "
                f"{metrics['标准化互信息 (NMI)']:<10.4f} "
                f"{metrics['调整互信息 (AMI)']:<10.4f} "
                f"{metrics.get('耗时(秒)', 0):<10.2f}"
            )


def main():
    excel_file_path = "clustering_datasets.xlsx"

    if not os.path.exists(excel_file_path):
        print(f"错误: 找不到文件 {excel_file_path}")
        return

    # 推荐默认方式：不用 ACC 优化 k，直接用真实类别数做外部评估。
    # 如果一定要搜索 k，把 use_optimization 改成 True，并优先用 optimize_metric="ari" 或 "ami"。
    results = run_kmeans_datasets(
        excel_file_path,
        use_optimization=False,
        optimize_metric="ari",
        max_clusters=30,
        make_plot=True,
    )

    print_results_summary(results)
    print("\nK-means 算法评估完成！")


if __name__ == "__main__":
    main()
