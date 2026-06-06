"""
mlscratch.unsupervised
======================
From-scratch implementations of unsupervised learning algorithms.

New algorithms added
--------------------
DBSCAN                  – Density-based spatial clustering
PCA                     – Principal Component Analysis
GaussianMixtureModel    – GMM via Expectation-Maximization
AgglomerativeClustering – Hierarchical agglomerative clustering
KMedoids                – K-Medoids (PAM) clustering
Apriori                 – Association rule mining
FastICA                 – Independent Component Analysis (FastICA)
TSNE                    – t-SNE dimensionality reduction
"""

from .dbscan import DBSCAN                                          # noqa: F401
from .pca import PCA                                                # noqa: F401
from .gmm import GaussianMixtureModel                               # noqa: F401
from .hierarchical_clustering import AgglomerativeClustering        # noqa: F401
from .kmedoids import KMedoids                                      # noqa: F401
from .apriori import Apriori                                        # noqa: F401
from .ica import FastICA                                            # noqa: F401
from .tsne import TSNE                                              # noqa: F401

__all__ = [
    "DBSCAN",
    "PCA",
    "GaussianMixtureModel",
    "AgglomerativeClustering",
    "KMedoids",
    "Apriori",
    "FastICA",
    "TSNE",
]
