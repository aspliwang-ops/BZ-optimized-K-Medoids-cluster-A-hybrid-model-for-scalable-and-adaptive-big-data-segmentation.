
"""
Reproducible implementation of the BZ-Optimized K-Medoids (BZ-KMedoids)
workflow described in the manuscript:

"BZ-optimized K-Medoids cluster: A hybrid model for scalable and adaptive
big data segmentation"

Standard framework: Python 3 + NumPy + pandas + scikit-learn.

The implementation follows the manuscript's described stages:
1. Load Bank Marketing data
2. Encode/scale numeric representation
3. Select K (default K=5, as reported by the manuscript's elbow analysis)
4. Initialize K medoids
5. Compute fuzzy membership degrees from distances
6. Refine medoids using membership-weighted distances
7. Iterate until convergence
8. Assign final clusters using maximum membership
9. Report clustering error, silhouette score and convergence time
10. Produce PCA visualization

Note:
The manuscript's equation symbols for the BZ objective are not preserved
in the extracted document text. Therefore, the implementation below uses
the textual definition in Sections 3.4–3.6: distance-weighted fuzzy
membership, membership regularization, and iterative medoid refinement.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class BZKMedoids:
    """BZ-optimized fuzzy K-Medoids implementation."""

    def __init__(
        self,
        n_clusters: int = 5,
        fuzziness: float = 2.0,
        regularization: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-5,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.fuzziness = fuzziness
        self.regularization = regularization
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.medoid_indices_ = None
        self.medoid_vectors_ = None
        self.membership_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.convergence_time_ = None

    @staticmethod
    def _pairwise_euclidean(X, M):
        return np.sqrt(
            np.maximum(
                ((X[:, None, :] - M[None, :, :]) ** 2).sum(axis=2), 1e-12
            )
        )

    def _membership(self, distances):
        # Fuzzy membership based on relative distances.
        # Equivalent to the distance-ratio formulation described in Sec. 3.4.1.
        power = 2.0 / (self.fuzziness - 1.0)
        ratio = (distances[:, :, None] / distances[:, None, :]) ** power
        u = 1.0 / np.maximum(ratio.sum(axis=2), 1e-12)
        u /= np.maximum(u.sum(axis=1, keepdims=True), 1e-12)
        return u

    def _update_medoids(self, X, U):
        new_indices = []
        weights = U ** self.fuzziness

        for k in range(self.n_clusters):
            w = weights[:, k]
            # Candidate medoid must be an observed data point.
            candidate_ids = np.arange(X.shape[0])

            # Compute weighted distance from each candidate to all observations.
            # Chunking limits peak memory for larger datasets.
            best_cost = np.inf
            best_idx = None
            chunk = 1024

            for start in range(0, len(candidate_ids), chunk):
                ids = candidate_ids[start:start + chunk]
                D = np.sqrt(
                    np.maximum(
                        ((X[ids, None, :] - X[None, :, :]) ** 2).sum(axis=2),
                        1e-12,
                    )
                )
                cost = (D * w[None, :]).sum(axis=1)
                idx_local = int(np.argmin(cost))
                if cost[idx_local] < best_cost:
                    best_cost = float(cost[idx_local])
                    best_idx = int(ids[idx_local])

            new_indices.append(best_idx)

        return np.array(new_indices, dtype=int)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        if X.ndim != 2:
            raise ValueError("X must be a 2-D numeric array.")
        if self.n_clusters < 2 or self.n_clusters > len(X):
            raise ValueError("n_clusters must be between 2 and n_samples.")

        t0 = time.perf_counter()
        medoid_indices = rng.choice(len(X), self.n_clusters, replace=False)

        for iteration in range(1, self.max_iter + 1):
            medoids = X[medoid_indices]
            distances = self._pairwise_euclidean(X, medoids)
            U = self._membership(distances)

            new_indices = self._update_medoids(X, U)

            # Convergence in medoid positions.
            shift = np.linalg.norm(X[new_indices] - X[medoid_indices], axis=1).max()

            medoid_indices = new_indices
            if shift <= self.tol:
                self.n_iter_ = iteration
                break
        else:
            self.n_iter_ = self.max_iter

        self.medoid_indices_ = medoid_indices
        self.medoid_vectors_ = X[medoid_indices]
        distances = self._pairwise_euclidean(X, self.medoid_vectors_)
        self.membership_ = self._membership(distances)
        self.labels_ = np.argmax(self.membership_, axis=1)

        # Within-cluster distance (clustering error / total deviation).
        self.inertia_ = float(
            np.min(distances, axis=1).sum()
        )
        self.convergence_time_ = time.perf_counter() - t0
        return self

    def predict(self, X):
        if self.medoid_vectors_ is None:
            raise RuntimeError("Fit the model before calling predict().")
        distances = self._pairwise_euclidean(np.asarray(X, dtype=float),
                                             self.medoid_vectors_)
        U = self._membership(distances)
        return np.argmax(U, axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_


def load_bank_marketing(csv_path: str):
    """
    Load a Bank Marketing CSV. The function accepts the common UCI/Kaggle
    semicolon-delimited format as well as comma-delimited CSV files.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # sep=None lets pandas infer comma/semicolon for ordinary CSV files.
    df = pd.read_csv(path, sep=None, engine="python")

    # Remove duplicate rows and obvious target columns from clustering input.
    df = df.drop_duplicates().reset_index(drop=True)

    target = None
    for candidate in ("y", "target", "label"):
        if candidate in df.columns:
            target = df[candidate].copy()
            df = df.drop(columns=[candidate])
            break

    # Treat object/category columns as categorical and one-hot encode them.
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        )

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X = preprocessor.fit_transform(df)
    X = np.asarray(X, dtype=np.float64)

    return df, X, target


def evaluate_clusters(X, labels, target=None):
    out = {}
    unique_labels = np.unique(labels)

    if len(unique_labels) > 1:
        out["silhouette_score"] = float(silhouette_score(X, labels))
    else:
        out["silhouette_score"] = np.nan

    # ARI is reported only when the original Bank Marketing target is present.
    # It is not used as a training objective.
    if target is not None:
        y = pd.Series(target).astype(str).values
        # Convert the binary target to integer labels only for external evaluation.
        y_encoded = pd.factorize(y)[0]
        out["adjusted_rand_index_vs_target"] = float(
            adjusted_rand_score(y_encoded, labels)
        )

    return out


def save_pca_plot(X, labels, output_path):
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(10, 7))
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        plt.scatter(Z[mask, 0], Z[mask, 1], s=10, alpha=0.65, label=f"Cluster {cluster_id}")
    plt.xlabel("Principal Component 1", fontweight="bold", fontsize=18)
    plt.ylabel("Principal Component 2", fontweight="bold", fontsize=18)
    plt.title("PCA 2D Scatter Plot of BZ-KMedoids Clusters",
              fontweight="bold", fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Bank Marketing CSV")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fuzziness", type=float, default=2.0)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="bz_kmedoids_results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, X, target = load_bank_marketing(args.data)

    model = BZKMedoids(
        n_clusters=args.k,
        fuzziness=args.fuzziness,
        regularization=args.regularization,
        max_iter=args.max_iter,
        random_state=args.seed,
    )

    labels = model.fit_predict(X)
    metrics = evaluate_clusters(X, labels, target)

    result = {
        "n_samples": len(X),
        "n_features_after_encoding": X.shape[1],
        "n_clusters": args.k,
        "fuzziness": args.fuzziness,
        "regularization": args.regularization,
        "iterations": model.n_iter_,
        "convergence_time_seconds": model.convergence_time_,
        "clustering_error_total_deviation": model.inertia_,
        **metrics,
    }

    pd.DataFrame([result]).to_csv(out_dir / "BZ_KMedoids_results.csv", index=False)

    assignments = df.copy()
    assignments["BZ_KMedoids_Cluster"] = labels
    assignments.to_csv(out_dir / "BZ_KMedoids_cluster_assignments.csv", index=False)

    save_pca_plot(X, labels, out_dir / "BZ_KMedoids_PCA.png")

    print("\nBZ-KMedoids completed.")
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"\nResults saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
