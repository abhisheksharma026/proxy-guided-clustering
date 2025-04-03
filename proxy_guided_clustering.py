import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import umap
import gower
import matplotlib.patheffects as path_effects

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, mean_squared_error
)
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split


def generate_data(n_samples=10000, n_outliers=50, random_state=42):
    np.random.seed(random_state)
    latent = np.random.normal(0, 1, (n_samples, 1))
    high_corr = np.hstack([latent + np.random.normal(0, 0.1, (n_samples, 1)) for _ in range(5)])
    low_corr = np.random.normal(0, 1, (n_samples, 5))
    X = np.hstack([high_corr, low_corr])
    columns = [f'feat{i+1}' for i in range(10)]
    df = pd.DataFrame(X, columns=columns)

    outliers = np.random.normal(15, 1, (n_outliers, 10))
    df = pd.concat([df, pd.DataFrame(outliers, columns=columns)], ignore_index=True)

    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, columns


def localized_lof_proxy(X, k=10, region_clusters=20):
    kmeans = MiniBatchKMeans(n_clusters=region_clusters, random_state=42).fit(X)
    regions = kmeans.predict(X)
    lof = LocalOutlierFactor(n_neighbors=k, novelty=False)
    lof_scores = -lof.fit(X).negative_outlier_factor_
    normalized = np.zeros(len(X))
    for r in np.unique(regions):
        mask = regions == r
        scores = lof_scores[mask]
        normalized[mask] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return 1 - normalized


def train_gbr_with_early_stopping(df, columns):
    X_train, X_val, y_train, y_val = train_test_split(df[columns], df['density_score'], test_size=0.2, random_state=42)
    gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42)
    gbr.fit(X_train, y_train)

    val_losses = [mean_squared_error(y_val, y_pred) for y_pred in gbr.staged_predict(X_val)]
    best_n_estimators = np.argmin(val_losses) + 1

    final_model = GradientBoostingRegressor(
        n_estimators=best_n_estimators,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    final_model.fit(df[columns], df['density_score'])
    return final_model


def compute_gower_per_cluster(df, labels, columns):
    scores = []
    for label in np.unique(labels):
        cluster = df[columns][labels == label]
        dist = gower.gower_matrix(cluster)
        mean = np.sum(dist) / (len(cluster) ** 2) if len(cluster) > 1 else 0
        scores.append((label, mean, len(cluster)))
    total = sum(s[2] for s in scores)
    weighted_avg = sum(s[1] * s[2] for s in scores) / total
    return scores, weighted_avg


def visualize_results(df, columns, gbr, sil_km, gower_km_avg, sil_gbr, gower_gbr_avg):
    df_aug = df[columns].copy()
    df_aug['predicted_density'] = df['gbr_predicted_density']

    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(df_aug)
    df['pca1'], df['pca2'] = pca_emb[:, 0], pca_emb[:, 1]

    reducer = umap.UMAP(random_state=42)
    umap_emb = reducer.fit_transform(df_aug)
    df['umap1'], df['umap2'] = umap_emb[:, 0], umap_emb[:, 1]

    shap_values = shap.TreeExplainer(gbr).shap_values(df[columns])
    shap_df = pd.DataFrame(shap_values, columns=columns)
    shap_mean = shap_df.abs().mean().sort_values(ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    sns.scatterplot(ax=axes[0, 0], data=df, x='pca1', y='pca2', hue='gbr_cluster', palette='tab10', s=10, legend=False)
    axes[0, 0].set_title("PCA Projection: GBR Clusters")
    for cluster in df['gbr_cluster'].unique():
        cx, cy = df[df['gbr_cluster'] == cluster][['pca1', 'pca2']].mean()
        text = axes[0, 0].text(cx, cy, str(cluster), fontsize=10, weight='bold', color='black')
        text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])

    sns.scatterplot(ax=axes[0, 1], data=df, x='umap1', y='umap2', hue='gbr_cluster', palette='tab10', s=10, legend=False)
    axes[0, 1].set_title("UMAP Projection: GBR Clusters")
    for cluster in df['gbr_cluster'].unique():
        ux, uy = df[df['gbr_cluster'] == cluster][['umap1', 'umap2']].mean()
        text = axes[0, 1].text(ux, uy, str(cluster), fontsize=10, weight='bold', color='black')
        text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])

    sns.barplot(ax=axes[1, 0], x=shap_mean.values, y=shap_mean.index, palette='viridis')
    axes[1, 0].set_title("SHAP Feature Importance (GBR Proxy)")

    comp_df = pd.DataFrame({
        'Metric': ['Silhouette Score', 'Avg. Gower Distance'],
        'KMeans': [sil_km, gower_km_avg],
        'GBR Clustering': [sil_gbr, gower_gbr_avg]
    }).melt(id_vars='Metric', var_name='Method', value_name='Score')

    sns.barplot(ax=axes[1, 1], data=comp_df, x='Metric', y='Score', hue='Method', palette='Set2')
    axes[1, 1].set_title("Model Comparison: Silhouette vs Gower")
    for container in axes[1, 1].containers:
        axes[1, 1].bar_label(container, fmt='%.3f', label_type='edge', fontsize=9, padding=3)

    plt.tight_layout()
    plt.show()


def main():
    n_clusters = 50
    df, columns = generate_data()
    df['density_score'] = localized_lof_proxy(df[columns])

    gbr = train_gbr_with_early_stopping(df, columns)
    df['gbr_predicted_density'] = gbr.predict(df[columns])

    df_aug = df[columns].copy()
    df_aug['predicted_density'] = df['gbr_predicted_density']

    df['gbr_cluster'] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(df_aug)
    df['kmeans_baseline'] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(df[columns])

    sil_gbr = silhouette_score(df[columns], df['gbr_cluster'])
    db_gbr = davies_bouldin_score(df[columns], df['gbr_cluster'])
    ch_gbr = calinski_harabasz_score(df[columns], df['gbr_cluster'])

    sil_km = silhouette_score(df[columns], df['kmeans_baseline'])
    db_km = davies_bouldin_score(df[columns], df['kmeans_baseline'])
    ch_km = calinski_harabasz_score(df[columns], df['kmeans_baseline'])

    _, gower_gbr_avg = compute_gower_per_cluster(df, df['gbr_cluster'], columns)
    _, gower_km_avg = compute_gower_per_cluster(df, df['kmeans_baseline'], columns)

    comparison = pd.DataFrame([
        {"Metric": "Silhouette Score", "GradientBoosted Clustering": round(sil_gbr, 3), "Vanilla KMeans": round(sil_km, 3)},
        {"Metric": "Davies–Bouldin Index", "GradientBoosted Clustering": round(db_gbr, 3), "Vanilla KMeans": round(db_km, 3)},
        {"Metric": "Calinski–Harabasz Index", "GradientBoosted Clustering": round(ch_gbr, 3), "Vanilla KMeans": round(ch_km, 3)},
        {"Metric": "Avg. Gower Distance", "GradientBoosted Clustering": round(gower_gbr_avg, 3), "Vanilla KMeans": round(gower_km_avg, 3)}
    ])

    visualize_results(df, columns, gbr, sil_km, gower_km_avg, sil_gbr, gower_gbr_avg)
    print("\n=== Cluster Comparison Metrics ===")
    print(comparison)


if __name__ == "__main__":
    main()