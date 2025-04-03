## 📐 Mathematical Intuition

This pipeline innovatively combines unsupervised **density estimation** with supervised **regression modeling** and classic **KMeans clustering**. Below is a breakdown of the core mathematical principles behind each step.

---

### ✅ 1. Local Outlier Factor (LOF) as Density Proxy

For each data point \( \mathbf{x}_i \), we compute the **Local Outlier Factor** score:

\[
\text{LOF}_k(\mathbf{x}_i) = \frac{1}{|N_k(\mathbf{x}_i)|} \sum_{\mathbf{x}_j \in N_k(\mathbf{x}_i)} \frac{\text{lrd}_k(\mathbf{x}_j)}{\text{lrd}_k(\mathbf{x}_i)}
\]

Where:
- \( N_k(\mathbf{x}_i) \) = k-nearest neighbors of \( \mathbf{x}_i \)  
- \( \text{lrd}_k \) = local reachability density of the point  

We normalize these scores **locally** within each KMeans-defined region \( R \), making the density signal more interpretable:

\[
\hat{d}_i = 1 - \frac{\text{LOF}_k(\mathbf{x}_i) - \min_{j \in R} \text{LOF}_k(\mathbf{x}_j)}{\max_{j \in R} \text{LOF}_k(\mathbf{x}_j) - \min_{j \in R} \text{LOF}_k(\mathbf{x}_j) + \varepsilon}
\]

- \( \varepsilon \) is a small constant to prevent division by zero  
- **Higher \( \hat{d}_i \) → more dense / central**

---

### ✅ 2. Learning the Proxy Function with GBR

We train a **regression model** \( f_\theta: \mathbb{R}^d \rightarrow [0,1] \) to approximate the normalized LOF:

\[
f_\theta(\mathbf{x}_i) \approx \hat{d}_i
\]

Objective:  
Minimize **Mean Squared Error (MSE)**:

\[
\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \left( f_\theta(\mathbf{x}_i) - \hat{d}_i \right)^2
\]

Implementation:
- Model: **Gradient Boosting Regressor**
- Tuning: **Early stopping** based on validation MSE over staged predictions

---

### ✅ 3. Feature Augmentation for Clustering

Each input vector \( \mathbf{x}_i \in \mathbb{R}^d \) is augmented with its predicted proxy density:

\[
\mathbf{x}_i' = \left[ \mathbf{x}_i, f_\theta(\mathbf{x}_i) \right] \in \mathbb{R}^{d+1}
\]

This transforms the space to incorporate **both geometry and density** into clustering.

---

### ✅ 4. Clustering in Augmented Space

We run **KMeans** clustering on the enriched feature space:

\[
\min_{\{\mu_k\}} \sum_{i=1}^{n} \min_k \left\| \mathbf{x}_i' - \mu_k \right\|^2
\]

Where:
- \( \mu_k \) are the cluster centroids in \( \mathbb{R}^{d+1} \)

---

### ✅ 5. Evaluation Metrics

We evaluate clustering performance using **geometric** and **semantic** metrics.

#### 📐 Geometric Metrics
- **Silhouette Score**:
  \[
  s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
  \]

- **Davies–Bouldin Index**
- **Calinski–Harabasz Index**

#### 🌐 Semantic Metric (Gower)
- **Gower Distance**:
  \[
  D_{ij}^{\text{Gower}} = \frac{1}{p} \sum_{k=1}^p d_{ijk}
  \]
  Where \( d_{ijk} \) handles numeric, categorical, and binary features.

We compute:
- **Avg. intra-cluster Gower distance**
- **Weighted avg. over all clusters**

---

### ✅ 6. Explainability via SHAP

We use **SHAP values** on the GBR model \( f_\theta \) to understand the feature contribution to predicted densities:

\[
\text{SHAP}_j(\mathbf{x}_i) = \phi_j
\]

- \( \phi_j \): contribution of feature \( j \) to the density prediction for \( \mathbf{x}_i \)
- Visualized as **feature importance bar plots**

---

## 📊 Summary

This approach transforms KMeans into a:

> **Density-aware, interpretable, and semantically-cohesive clustering algorithm**

🔍 **Benefits**:
- Works well for **mixed-type and outlier-rich data**
- Offers **interpretability** via SHAP
- Improves over vanilla KMeans without changing the core algorithm
