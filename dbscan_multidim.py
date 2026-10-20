import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


class DBSCANClustering:
    """DBSCAN聚类算法封装"""

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = int(min_samples)
        self.model = None
        self.labels_ = None

    def fit(self, X):
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = self.model.fit_predict(X)
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def evaluate(self, true_labels):
        """评估聚类质量。噪声点(-1)作为一个单独类别参与评价。"""
        if self.labels_ is None:
            raise ValueError("模型尚未拟合，请先调用fit方法")

        pred_labels = self._ensure_continuous_labels(self.labels_)
        true_labels = self._ensure_continuous_labels(true_labels)

        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ami = adjusted_mutual_info_score(true_labels, pred_labels)
        acc = self._cluster_accuracy(true_labels, pred_labels)

        return {
            "准确率 (ACC)": acc,
            "调整兰德指数 (ARI)": ari,
            "标准化互信息 (NMI)": nmi,
            "调整互信息 (AMI)": ami,
        }

    def _cluster_accuracy(self, true_labels, pred_labels):
        """用匈牙利算法计算聚类标签最佳匹配后的准确率。"""
        true_labels = self._ensure_continuous_labels(true_labels)
        pred_labels = self._ensure_continuous_labels(pred_labels)

        n_true = len(np.unique(true_labels))
        n_pred = len(np.unique(pred_labels))
        n_classes = max(n_true, n_pred)
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

        for true_label, pred_label in zip(true_labels, pred_labels):
            confusion_matrix[true_label, pred_label] += 1

        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
        return confusion_matrix[row_ind, col_ind].sum() / len(true_labels)

    @staticmethod
    def _ensure_continuous_labels(labels):
        """确保标签是从0开始的连续整数。"""
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_mapping[label] for label in labels], dtype=int)

    @staticmethod
    def _to_2d_for_plot(X):
        """二维数据直接绘制；一维补零；三维及以上用PCA降到二维。"""
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X必须是二维数组，形状为: 样本数 x 特征数")

        if X.shape[1] == 1:
            return np.column_stack([X[:, 0], np.zeros(X.shape[0])]), "原始第1维", "0"

        if X.shape[1] == 2:
            return X[:, :2], "特征1", "特征2"

        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
        ratio = pca.explained_variance_ratio_
        x_label = f"PCA1 ({ratio[0]:.1%})"
        y_label = f"PCA2 ({ratio[1]:.1%})"
        return X_2d, x_label, y_label

    @staticmethod
    def _scatter_with_noise(X_2d, labels, title, x_label, y_label):
        """绘制聚类散点图，并将DBSCAN噪声点显示为灰色。"""
        labels = np.asarray(labels)
        noise_mask = labels == -1
        cluster_mask = ~noise_mask

        if np.any(cluster_mask):
            scatter = plt.scatter(
                X_2d[cluster_mask, 0],
                X_2d[cluster_mask, 1],
                c=labels[cluster_mask],
                cmap="tab10",
                s=50,
                alpha=0.8,
            )
            plt.colorbar(scatter, fraction=0.046, pad=0.04)

        if np.any(noise_mask):
            plt.scatter(
                X_2d[noise_mask, 0],
                X_2d[noise_mask, 1],
                c="gray",
                s=50,
                alpha=0.7,
                label="噪声点",
            )
            plt.legend()

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)

    def plot_clusters(self, X, true_labels=None, title="DBSCAN聚类结果", save_path=None):
        """绘制聚类结果。多维数据会用PCA投影到二维后可视化。"""
        if self.labels_ is None:
            raise ValueError("模型尚未拟合，请先调用fit方法")

        X_2d, x_label, y_label = self._to_2d_for_plot(X)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        if true_labels is not None:
            scatter = plt.scatter(
                X_2d[:, 0],
                X_2d[:, 1],
                c=true_labels,
                cmap="tab10",
                s=50,
                alpha=0.8,
            )
            plt.colorbar(scatter, fraction=0.046, pad=0.04)
            plt.title("真实标签")
        else:
            plt.text(
                0.5,
                0.5,
                "无真实标签",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("无真实标签")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        self._scatter_with_noise(X_2d, self.labels_, title, x_label, y_label)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


class DBSCANOptimizer:
    """DBSCAN参数优化器"""

    def __init__(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.best_params = None
        self.best_score = -np.inf

    def objective_function(self, params):
        """目标函数：最大化ARI。differential_evolution本身做最小化，所以返回-ARI。"""
        try:
            eps, min_samples = params
            min_samples = int(round(min_samples))

            if eps <= 0.0 or min_samples < 2:
                return 1.0

            model = DBSCANClustering(eps=eps, min_samples=min_samples)
            model.fit(self.X)
            metrics = model.evaluate(self.y)
            ari = metrics["调整兰德指数 (ARI)"]

            if ari > self.best_score:
                self.best_score = ari
                self.best_params = (eps, min_samples)

            return -ari

        except Exception:
            return 1.0

    def optimize(self, max_iter=20):
        """优化DBSCAN参数"""
        print("优化DBSCAN参数...")

        n_samples = len(self.X)
        if n_samples < 3:
            raise ValueError("样本数太少，无法优化DBSCAN参数")

        n_neighbors = min(5, n_samples)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(self.X)
        distances, _ = nbrs.kneighbors(self.X)
        avg_distance = np.mean(distances[:, 1:]) if n_neighbors > 1 else 0.5

        eps_min = max(0.05, avg_distance * 0.5)
        eps_max = max(eps_min + 0.05, min(5.0, avg_distance * 3))
        max_min_samples = max(2, min(20, n_samples // 10 if n_samples >= 20 else n_samples))

        bounds = [
            (eps_min, eps_max),
            (2, max_min_samples),
        ]

        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=max_iter,
            popsize=10,
            recombination=0.7,
            seed=42,
            disp=True,
        )

        best_eps = float(result.x[0])
        best_min_samples = int(round(result.x[1]))
        print(f"DBSCAN最优参数: eps={best_eps:.3f}, min_samples={best_min_samples}")
        print(f"最优ARI: {-result.fun:.4f}")

        return (best_eps, best_min_samples), -result.fun


def load_excel_datasets(file_path, label_col=-1):
    """
    从Excel文件加载数据集。

    默认约定：每个sheet最后一列是标签，其余所有列都是特征，因此支持多维数据。
    """
    datasets = {}

    try:
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names

        print(f"找到 {len(sheet_names)} 个sheet: {sheet_names}")

        for sheet_name in sheet_names:
            print(f"加载sheet: {sheet_name}")
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            if df.shape[1] < 2:
                print(f"警告: {sheet_name} 至少需要1列特征和1列标签，跳过")
                continue

            label_series = df.iloc[:, label_col]
            feature_df = df.drop(columns=df.columns[label_col])

            feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
            label_series = pd.to_numeric(label_series, errors="coerce")

            valid_mask = ~(feature_df.isna().any(axis=1) | label_series.isna())
            feature_df = feature_df.loc[valid_mask]
            label_series = label_series.loc[valid_mask]

            if feature_df.empty:
                print(f"警告: {sheet_name} 清理缺失值后没有有效数据，跳过")
                continue

            X = feature_df.values.astype(float)
            y = label_series.values.astype(int)
            unique_y = np.unique(y)

            datasets[sheet_name] = {
                "data": X,
                "target": y,
                "feature_names": list(feature_df.columns),
                "target_names": [f"类别{i}" for i in unique_y],
            }

            print(
                f"  - 数据形状: {X.shape}, 特征数: {X.shape[1]}, 类别数: {len(unique_y)}"
            )

    except Exception as e:
        print(f"加载Excel文件时出错: {e}")
        import traceback

        traceback.print_exc()

    return datasets


def _sample_data(X, y, sample_size, random_state=42):
    """按固定随机种子抽样，sample_size为None时使用全量数据。"""
    if sample_size is None or len(X) <= sample_size:
        return X, y, np.arange(len(X))

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X), size=sample_size, replace=False)
    indices.sort()
    return X[indices], y[indices], indices


def run_dbscan_datasets(
    file_path,
    use_optimization=True,
    label_col=-1,
    optimize_sample_size=10000,
    fit_sample_size=None,
    random_state=42,
):
    """测试DBSCAN算法在所有数据集上的表现"""
    print("DBSCAN聚类算法评估")
    print("=" * 80)

    datasets = load_excel_datasets(file_path, label_col=label_col)

    if not datasets:
        print("无法加载Excel数据集")
        return {}

    results = {}

    for name, dataset in datasets.items():
        start = time.time()
        print(f"\n{'=' * 50}")
        print(f"测试 {name} 数据集")
        print(
            f"数据形状: {dataset['data'].shape}, 类别数: {len(np.unique(dataset['target']))}"
        )
        print(f"{'=' * 50}")

        X, y = dataset["data"], dataset["target"]
        full_size = len(X)

        if fit_sample_size is not None and full_size > fit_sample_size:
            print(
                f"数据量较大，DBSCAN将从 {full_size} 条样本中抽取 "
                f"{fit_sample_size} 条进行Baseline评估。"
            )
            X, y, _ = _sample_data(X, y, fit_sample_size, random_state=random_state)

        X_scaled = StandardScaler().fit_transform(X)

        if use_optimization:
            try:
                X_opt, y_opt, _ = _sample_data(
                    X_scaled,
                    y,
                    optimize_sample_size,
                    random_state=random_state,
                )
                if len(X_opt) < len(X_scaled):
                    print(f"参数优化使用 {len(X_opt)} 条抽样数据。")

                optimizer = DBSCANOptimizer(X_opt, y_opt)
                best_params, _ = optimizer.optimize()
                model = DBSCANClustering(eps=best_params[0], min_samples=best_params[1])
            except Exception as e:
                print(f"参数优化失败，使用默认参数: {e}")
                model = DBSCANClustering(eps=0.5, min_samples=5)
        else:
            model = DBSCANClustering(eps=0.5, min_samples=5)

        try:
            model.fit(X_scaled)
            metrics = model.evaluate(y)
            metrics["耗时(秒)"] = time.time() - start
            metrics["样本数"] = len(X)
            results[name] = metrics

            print(f"{name} 数据集聚类结果:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

            n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
            n_noise = int(np.sum(model.labels_ == -1))
            print(f"  聚类数量: {n_clusters}")
            print(f"  噪声点数量: {n_noise}")

            safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
            model.plot_clusters(
                X_scaled,
                true_labels=y,
                title="DBSCAN聚类结果",
                save_path=f"{safe_name}_dbscan_clustering.png",
            )

        except Exception as e:
            print(f"处理 {name} 数据集时出错: {e}")
            import traceback

            traceback.print_exc()
            results[name] = None

    return results


def print_results_summary(results):
    """打印结果总结"""
    print("\n" + "=" * 80)
    print("DBSCAN聚类算法性能总结")
    print("=" * 80)

    if not results:
        print("没有有效的结果可显示")
        return

    print(
        f"{'数据集':<15} {'样本数':<10} {'ACC':<8} {'ARI':<8} "
        f"{'NMI':<8} {'AMI':<8} {'耗时(秒)':<10}"
    )
    print("-" * 88)

    for name, metrics in results.items():
        if metrics:
            print(
                f"{name:<15} {int(metrics.get('样本数', 0)):<10} "
                f"{metrics['准确率 (ACC)']:.4f}  "
                f"{metrics['调整兰德指数 (ARI)']:.4f}  "
                f"{metrics['标准化互信息 (NMI)']:.4f}  "
                f"{metrics['调整互信息 (AMI)']:.4f}  "
                f"{metrics['耗时(秒)']:.2f}"
            )


def main():
    """主函数"""
    excel_file_path = "clustering_datasets.xlsx"

    if not os.path.exists(excel_file_path):
        print(f"错误: 找不到文件 {excel_file_path}")
        return

    results = run_dbscan_datasets(
        excel_file_path,
        use_optimization=True,
        label_col=-1,
        optimize_sample_size=10000,
        fit_sample_size=50000,
    )
    print_results_summary(results)
    print("\nDBSCAN算法评估完成！")


if __name__ == "__main__":
    main()
