import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

# 1. Cấu hình thư mục đầu ra và cài đặt giao diện
OUTPUT_DIR = Path("reports/online_retail/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14
})

# 2. Đọc và làm sạch dữ liệu (Loại bỏ Outliers bằng IQR)
if not os.path.exists("online_retail_rfm.csv"):
    raise FileNotFoundError("Không tìm thấy file 'online_retail_rfm.csv' ở thư mục hiện tại!")

df = pd.read_csv("online_retail_rfm.csv")
df_clean = df.copy()

# Áp dụng IQR cho 3 thuộc tính chính
for col in ["recency_days", "frequency", "monetary_value"]:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df_clean[(df_clean[col] >= Q1 - 1.5 * IQR) & (df_clean[col] <= Q3 + 1.5 * IQR)]
df_clean = df_clean.reset_index(drop=True)

# Chuẩn hóa dữ liệu để phục vụ phân cụm
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[["recency_days", "frequency", "monetary_value"]])

# Gán nhãn cụm tối ưu K=3 từ mô hình K-Means mới
kmeans_best = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean["cluster"] = kmeans_best.fit_predict(X_scaled)


# -------------------------------------------------------------------------
# HÀM HỖ TRỢ LƯU ẢNH
# -------------------------------------------------------------------------
def save_fig(filename):
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"-> Đã lưu biểu đồ: {path}")


print(f"Bắt đầu vẽ biểu đồ trên tập dữ liệu đã làm sạch ({len(df_clean)} khách hàng)...")

# =========================================================================
# LUỒNG 1: BIỂU ĐỒ PHÂN PHỐI (DISTRIBUTION PLOTS)
# =========================================================================

# 1. RFM Recency Distribution
plt.figure(figsize=(7, 4))
sns.histplot(df_clean["recency_days"], kde=True, color="#2b5c8f", bins=25)
plt.title("RFM Recency Distribution (Outliers Removed)")
plt.xlabel("Recency (Days)")
plt.ylabel("Count")
save_fig("recency_days_histogram.png")

# 2. RFM Frequency Distribution
plt.figure(figsize=(7, 4))
sns.histplot(df_clean["frequency"], kde=True, color="#d95f02", bins=15)
plt.title("RFM Frequency Distribution (Outliers Removed)")
plt.xlabel("Frequency (Invoices)")
plt.ylabel("Count")
save_fig("frequency_histogram.png")

# 3. RFM Monetary Distribution
plt.figure(figsize=(7, 4))
sns.histplot(df_clean["monetary_value"], kde=True, color="#7570b3", bins=25)
plt.title("RFM Monetary Distribution (Outliers Removed)")
plt.xlabel("Monetary Value ($)")
plt.ylabel("Count")
save_fig("monetary_value_histogram.png")

# 4. Average Order Value Distribution
plt.figure(figsize=(7, 4))
sns.histplot(df_clean["average_order_value"], kde=True, color="#e7298a", bins=25)
plt.title("Average Order Value Distribution")
plt.xlabel("Average Order Value ($/Order)")
plt.ylabel("Count")
save_fig("average_order_value_histogram.png")


# =========================================================================
# LUỒNG 2: TƯƠNG QUAN & PHÂN TÁN GỐC (CORRELATION & SCATTER)
# =========================================================================

# 5. RFM Correlation Heatmap
plt.figure(figsize=(6, 4.5))
corr = df_clean[["recency_days", "frequency", "monetary_value", "average_order_value"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title("RFM Feature Correlation Heatmap")
save_fig("rfm_correlation_heatmap.png")

# 6. Recency vs Monetary Scatter (Trước phân cụm - Thể hiện cấu trúc thô)
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df_clean, x="recency_days", y="monetary_value", alpha=0.6, color="#4a4a4a")
plt.title("Recency vs Monetary Scatter Plot")
plt.xlabel("Recency (Days)")
plt.ylabel("Monetary Value ($)")
save_fig("recency_vs_monetary_scatter.png")


# =========================================================================
# LUỒNG 3: ĐÁNH GIÁ MÔ HÌNH (MODEL EVALUATION PLOTS)
# =========================================================================

# Tính toán ma trận Elbow và Silhouette phục vụ đồ thị lựa chọn K
k_range = range(2, 10)
inertias = []
sil_scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# 7. RFM K-Means Elbow Plot
plt.figure(figsize=(7, 4))
plt.plot(k_range, inertias, marker="o", linestyle="--", color="#1f77b4", linewidth=2)
plt.axvline(x=3, color="red", linestyle=":", label="Optimal K=3")
plt.title("RFM K-Means Elbow Plot")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (WCSS)")
plt.legend()
save_fig("rfm_kmeans_elbow_plot.png")

# 8. RFM Silhouette Score Comparison
plt.figure(figsize=(7, 4))
plt.plot(k_range, sil_scores, marker="s", linestyle="-", color="#2ca02c", linewidth=2)
plt.axvline(x=3, color="red", linestyle=":", label="Selected K=3")
plt.title("RFM Silhouette Score Comparison")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.legend()
save_fig("rfm_silhouette_score_comparison.png")

# 9. RFM Hierarchical Dendrogram
plt.figure(figsize=(9, 5))
Z = linkage(X_scaled, method="ward")
dendrogram(Z, no_labels=True, color_threshold=Z[-2, 2])  # Tô màu phân tách nhánh lớn
plt.title("RFM Hierarchical Clustering Dendrogram (Ward's Method)")
plt.xlabel("Customer Data Points")
plt.ylabel("Euclidean Distance")
save_fig("rfm_hierarchical_dendrogram.png")


# =========================================================================
# LUỒNG 4: KẾT QUẢ PHÂN CỤM (FINAL CLUSTERING PLOTS)
# =========================================================================

# Định nghĩa màu sắc cố định và tên phân khúc mới
cluster_colors = {0: "#2ca02c", 1: "#1f77b4", 2: "#d62728"}
cluster_labels = {
    0: "Cluster 0: VIP / Loyal",
    1: "Cluster 1: General / Potential",
    2: "Cluster 2: Hibernating / At-Risk"
}

# 10. RFM Clusters: Recency vs Monetary
plt.figure(figsize=(8, 5.5))
for c in sorted(df_clean["cluster"].unique()):
    c_mask = df_clean["cluster"] == c
    plt.scatter(
        df_clean.loc[c_mask, "recency_days"],
        df_clean.loc[c_mask, "monetary_value"],
        label=cluster_labels[c],
        color=cluster_colors[c],
        alpha=0.7,
        edgecolors="w",
        s=40
    )
plt.title("RFM Clusters: Recency vs Monetary")
plt.xlabel("Recency (Days)")
plt.ylabel("Monetary Value ($)")
plt.legend(title="Customer Segments")
save_fig("rfm_clusters_recency_vs_monetary.png")

# 11. RFM Cluster Size Distribution
plt.figure(figsize=(6, 4))
counts = df_clean["cluster"].value_counts().sort_index()
percentages = (counts / len(df_clean)) * 100
bars = plt.bar(counts.index.map(cluster_labels), counts.values, color=[cluster_colors[i] for i in counts.index])
plt.title("RFM Cluster Size Distribution")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15, ha="right")
# Thêm nhãn % lên đầu các cột
for bar, pct in zip(bars, percentages):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 30, f"{pct:.1f}%", ha="center", va="bottom", fontweight="bold")
save_fig("rfm_cluster_size_distribution.png")

# 12. RFM Cluster Profile Heatmap
plt.figure(figsize=(7, 4.5))
profile_stats = df_clean.groupby("cluster")[["recency_days", "frequency", "monetary_value"]].mean()
# Chuẩn hóa cục bộ dạng Min-Max để biểu thị cường độ màu trên Heatmap một cách cân bằng
profile_norm = (profile_stats - profile_stats.min()) / (profile_stats.max() - profile_stats.min())
profile_norm.index = profile_norm.index.map(cluster_labels)
profile_norm.columns = ["Recency (Days)", "Frequency (Invoices)", "Monetary ($)"]

# Điền giá trị trung bình thực tế (raw values) vào làm nhãn văn bản chữ
annot_labels = profile_stats.copy()
annot_labels.index = annot_labels.index.map(cluster_labels)
annot_text = np.array([
    [f"{profile_stats.iloc[i, 0]:.1f} d", f"{profile_stats.iloc[i, 1]:.1f} x", f"${profile_stats.iloc[i, 2]:.1f}"]
    for i in range(len(profile_stats))
])

sns.heatmap(profile_norm, annot=annot_text, fmt="", cmap="YlGnBu", cbar=True, linewidths=1)
plt.title("RFM Cluster Profile Heatmap (Mean Raw Values Annotated)")
plt.ylabel("Customer Segments")
save_fig("cluster_profile_heatmap.png")

print("\n🎉 HOÀN THÀNH! Toàn bộ 12 biểu đồ đã được xuất thành công.")