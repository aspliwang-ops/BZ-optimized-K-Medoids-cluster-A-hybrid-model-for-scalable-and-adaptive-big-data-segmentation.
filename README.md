
# Reproducible BZ-KMedoids Implementation

This package provides a Python/scikit-learn implementation corresponding to the
workflow described in the manuscript "BZ-optimized K-Medoids cluster: A hybrid
model for scalable and adaptive big data segmentation."

## Environment

Python >= 3.9

Install:
    pip install numpy pandas scikit-learn matplotlib

## Dataset

Use the Bank Marketing dataset specified in the manuscript:
https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing

Download the CSV and run:

    python bz_kmedoids.py --data /path/to/bank.csv --k 5

The manuscript reports K=5 from its elbow analysis.

## Outputs

The script produces:
- BZ_KMedoids_results.csv
- BZ_KMedoids_cluster_assignments.csv
- BZ_KMedoids_PCA.png

The results CSV reports sample count, encoded feature count, K, fuzziness,
regularization, iterations, convergence time, clustering error, silhouette
score, and (when the original target column is available) ARI against that
target.

## Reproducibility note

The manuscript's extracted text does not preserve the mathematical symbols in
Equations (4)-(8) and (15)-(16). The implementation therefore follows the
textual descriptions: fuzzy membership from relative medoid distances,
membership-weighted medoid refinement, iterative updating, and convergence.

For the revision, the authors should archive the exact executed source code
and environment specification (Python version and package versions) alongside
the manuscript.
