# The Density Awakens: Boosting-Enhanced Clustering

Proxy-Guided Clustering: Enhancing K-Means with Learned Density-Aware Augmentations

A novel density-aware clustering pipeline that uses a learned **proxy signal** to enhance the quality of KMeans segmentation. This approach combines **localized LOF**, **Gradient Boosted Regression**, and **feature-space augmentation** to improve cluster coherence and separation — especially in high-dimensional, mixed or noisy datasets.

---

## 🚀 Overview

Traditional clustering algorithms like KMeans are sensitive to scale, outliers, and density variations. This project introduces a novel hybrid approach:

1. **Estimate Local Density:**  
   Use Local Outlier Factor (LOF), normalized within regions via MiniBatchKMeans.

2. **Learn a Proxy Function:**  
   Train a Gradient Boosted Regressor to approximate the density signal.

3. **Augmented Feature:**  
   Append predicted density as an additional feature to original input.

4. **Cluster in Augmented Space:**  
   Apply KMeans on the enriched feature space for improved segmentation.

---

## Breakdown

✅ **Proxy Learning with GBDTs:**  
KMeans enhanced by density-predicted signals, simulating behavior of density-aware models like DBSCAN.
Train GBR to regress the LOF-derived "density score" using original features.
The regressor now learns a mapping from raw features to density, approximating local structure.

KMeans now clusters based on:
- Original feature geometry
- Learned density structure

✅ **Region-wise Normalized LOF:**  
Improves robustness and interpretability by adapting LOF to local structures.
LOF gives us a way to estimate how central/dense a point is in its neighborhood.
But LOF is non-differentiable, non-parametric, and hard to scale for large datasets.
Local normalization of LOF across KMeans-defined regions creates a smoothed, interpretable proxy for "density score".

✅ **Hybrid Evaluation with Gower Distance:**  
Gower distance adds cohesion interpretability beyond geometric metrics.

GBR bridges the gap between:
- Unsupervised structure (via LOF)
- Supervised function approximation (via regression)
- And density-aware clustering (via augmented KMeans)

✅ **Early-Stopped Regressor:**  
Optimized GBR via validation-based early stopping to prevent proxy overfitting.

---

## 📊 Metrics & Visuals

We compare our method against baseline KMeans using:

- **Silhouette Score**
- **Davies–Bouldin Index**
- **Calinski–Harabasz Index**
- **Gower Distance (intra-cluster)**

Also includes:
- PCA/UMAP cluster plots  
- SHAP explainability for the proxy model  
- Annotated bar charts comparing clustering performance  

---

**Overall Results:**

![image](https://github.com/user-attachments/assets/80dee290-5697-427c-be08-c67d5518afba)

---
   
![image](https://github.com/user-attachments/assets/68684de5-8e47-433d-8869-fcf7bb20f275)

---
If you use this work, please cite it as: 

```
@misc{abhishek_sharma_2025_boosting_enhanced_clustering,
author = {Abhishek Sharma},
title = {The Density Awakens: Boosting-Enhanced Clustering (v2.0.0)},
year = {2025},
publisher = {Zenodo},
doi = {10.5281/zenodo.15129301},
url = {https://doi.org/10.5281/zenodo.15129301}
```
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15129301.svg)](https://doi.org/10.5281/zenodo.15129301)
