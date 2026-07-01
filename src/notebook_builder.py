from __future__ import annotations

import json
from pathlib import Path

from config import FINAL_NOTEBOOK


def _markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def _image_cell(title: str, relative_path: str, explanation: str) -> dict:
    return _markdown(
        f"""
### {title}

![{title}]({relative_path})

{explanation}
"""
    )


MALL_FIGURES = [
    (
        "Age Distribution",
        "../reports/figures/age_histogram.png",
        "Histogram này cho biết phân bố độ tuổi của khách hàng trong Mall Customers. Đây là biến nhân khẩu học chính để xem nhóm khách hàng trẻ, trung niên hoặc lớn tuổi có hành vi chi tiêu khác nhau hay không.",
    ),
    (
        "Annual Income Distribution",
        "../reports/figures/annual_income_k_histogram.png",
        "Biểu đồ thể hiện phân bố thu nhập hằng năm theo đơn vị nghìn USD. Thu nhập là một feature quan trọng khi phân cụm vì nó thường liên quan đến khả năng chi tiêu.",
    ),
    (
        "Spending Score Distribution",
        "../reports/figures/spending_score_histogram.png",
        "Spending score thể hiện mức độ chi tiêu hoặc mức độ tương tác mua sắm. Phân bố biến này giúp nhận diện nhóm chi tiêu thấp, trung bình và cao.",
    ),
    (
        "Numerical Feature Boxplots",
        "../reports/figures/numeric_boxplots.png",
        "Boxplot dùng để kiểm tra outlier của age, annual income và spending score. Theo rule của chương trình, outlier được báo cáo nhưng không xóa tự động.",
    ),
    (
        "Gender Distribution",
        "../reports/figures/gender_distribution.png",
        "Biểu đồ đếm số lượng khách hàng theo gender. Gender không phải feature bắt buộc trong mô hình cuối, nhưng được thử trong feature set numeric-plus-gender để so sánh.",
    ),
    (
        "Correlation Heatmap",
        "../reports/figures/correlation_heatmap.png",
        "Heatmap correlation giúp kiểm tra quan hệ tuyến tính giữa các biến numeric. Vì clustering dựa trên khoảng cách, việc hiểu quan hệ giữa các feature giúp diễn giải cụm tốt hơn.",
    ),
    (
        "Income vs Spending Scatter",
        "../reports/figures/income_vs_spending_scatter.png",
        "Scatter plot giữa annual income và spending score là hình quan trọng nhất cho bài toán customer segmentation, vì nó thường thể hiện rõ các nhóm như high income-high spending hoặc high income-low spending.",
    ),
    (
        "Age vs Spending Scatter",
        "../reports/figures/age_vs_spending_scatter.png",
        "Biểu đồ age vs spending score giúp xem nhóm tuổi nào có xu hướng chi tiêu cao hoặc thấp.",
    ),
    (
        "Age vs Income Scatter",
        "../reports/figures/age_vs_income_scatter.png",
        "Biểu đồ age vs annual income hỗ trợ kiểm tra xem thu nhập có thay đổi theo độ tuổi trong dataset hay không.",
    ),
    (
        "K-Means Elbow Plot",
        "../reports/figures/kmeans_elbow_plot.png",
        "Elbow plot theo dõi inertia khi thay đổi số cụm k. Inertia giảm khi k tăng, nhưng điểm gấp khúc giúp chọn số cụm hợp lý hơn thay vì chọn quá nhiều cụm.",
    ),
    (
        "Silhouette Score Comparison",
        "../reports/figures/silhouette_score_comparison.png",
        "Biểu đồ so sánh silhouette score giữa các ứng viên model. Silhouette càng cao thì cụm càng tách biệt và gọn hơn, nhưng vẫn cần cân bằng với khả năng giải thích.",
    ),
    (
        "Hierarchical Dendrogram",
        "../reports/figures/hierarchical_dendrogram.png",
        "Dendrogram minh họa quá trình gộp cụm của hierarchical clustering. Hình này giúp kiểm soát trực quan số cụm và khoảng cách giữa các nhóm.",
    ),
    (
        "Best Clusters: Income vs Spending",
        "../reports/figures/best_clusters_income_vs_spending.png",
        "Đây là hình diễn giải chính của best model trên Mall Customers. Màu sắc biểu thị cluster đã gán cho từng khách hàng.",
    ),
    (
        "Best Clusters: Age vs Spending",
        "../reports/figures/best_clusters_age_vs_spending.png",
        "Hình này giúp giải thích cluster theo độ tuổi và spending score, bổ sung góc nhìn nhân khẩu học cho biểu đồ income-spending.",
    ),
    (
        "Cluster Size Distribution",
        "../reports/figures/cluster_size_distribution.png",
        "Biểu đồ cho biết kích thước từng cluster. Nếu một cụm quá nhỏ hoặc quá lớn, cần kiểm tra lại tính ổn định và ý nghĩa thực tế.",
    ),
    (
        "Cluster Profile Heatmap",
        "../reports/figures/cluster_profile_heatmap.png",
        "Heatmap profile tóm tắt giá trị trung bình của các feature theo từng cluster, giúp đặt tên và diễn giải customer segment.",
    ),
]


ONLINE_RETAIL_FIGURES = [
    (
        "RFM Recency Distribution",
        "../reports/online_retail/figures/recency_days_histogram.png",
        "Recency cho biết số ngày từ lần mua gần nhất đến snapshot date. Recency thấp thường biểu thị khách hàng còn hoạt động gần đây.",
    ),
    (
        "RFM Frequency Distribution",
        "../reports/online_retail/figures/frequency_histogram.png",
        "Frequency là số hóa đơn duy nhất của mỗi khách hàng. Đây là chỉ báo hành vi mua lặp lại.",
    ),
    (
        "RFM Monetary Distribution",
        "../reports/online_retail/figures/monetary_value_histogram.png",
        "Monetary value là tổng giá trị mua hàng của khách hàng sau khi làm sạch giao dịch.",
    ),
    (
        "Average Order Value Distribution",
        "../reports/online_retail/figures/average_order_value_histogram.png",
        "Average order value bổ sung góc nhìn về giá trị trung bình mỗi đơn hàng, tránh chỉ nhìn tổng monetary.",
    ),
    (
        "RFM Correlation Heatmap",
        "../reports/online_retail/figures/rfm_correlation_heatmap.png",
        "Heatmap kiểm tra quan hệ giữa recency, frequency, monetary value và average order value.",
    ),
    (
        "Recency vs Monetary Scatter",
        "../reports/online_retail/figures/recency_vs_monetary_scatter.png",
        "Scatter plot này giúp nhận diện nhóm khách hàng mua gần đây và có giá trị cao.",
    ),
    (
        "RFM K-Means Elbow Plot",
        "../reports/online_retail/figures/rfm_kmeans_elbow_plot.png",
        "Elbow plot cho RFM track giúp đánh giá số cụm hợp lý khi dùng K-Means trên dữ liệu giao dịch.",
    ),
    (
        "RFM Silhouette Score Comparison",
        "../reports/online_retail/figures/rfm_silhouette_score_comparison.png",
        "Biểu đồ so sánh silhouette cho các ứng viên RFM clustering.",
    ),
    (
        "RFM Hierarchical Dendrogram",
        "../reports/online_retail/figures/rfm_hierarchical_dendrogram.png",
        "Dendrogram cho RFM track, dùng để kiểm tra cấu trúc phân cấp của khách hàng giao dịch.",
    ),
    (
        "RFM Clusters: Recency vs Monetary",
        "../reports/online_retail/figures/rfm_clusters_recency_vs_monetary.png",
        "Hình phân cụm chính cho RFM track, biểu diễn cluster theo recency và monetary value.",
    ),
    (
        "RFM Cluster Size Distribution",
        "../reports/online_retail/figures/rfm_cluster_size_distribution.png",
        "Biểu đồ kích thước cluster trên Online Retail RFM.",
    ),
    (
        "RFM Cluster Profile Heatmap",
        "../reports/online_retail/figures/cluster_profile_heatmap.png",
        "Heatmap profile của RFM clusters, dùng để diễn giải nhóm khách hàng theo recency, frequency và monetary.",
    ),
]


def write_final_notebook() -> Path:
    cells = [
        _markdown(
            """
# Final Notebook - Customer Segmentation Clustering

Notebook này tổng hợp đầy đủ chương trình phân cụm khách hàng theo yêu cầu Practice Exercise 3. Cấu trúc được viết theo dạng lab notebook: nêu vấn đề, dữ liệu, làm sạch, EDA, scaling, training, đánh giá model, test, giao diện Python và kết luận.

Dataset chính để chấm bài là **Mall Customers** vì có đúng các feature trong đề: `Age`, `Gender`, `Annual Income (k$)`, `Spending Score (1-100)`. Dataset **UCI Online Retail** được dùng như track mở rộng để có thêm raw transaction data và tạo RFM segmentation, không merge trực tiếp với Mall Customers vì không cùng customer ID/schema.
"""
        ),
        _markdown(
            """
## 1. Define Problem

### Problem Statement

Ta có một tập dữ liệu khách hàng của cửa hàng bán lẻ gồm thông tin tuổi, giới tính, thu nhập hằng năm và spending score. Mục tiêu là **phân cụm khách hàng dựa trên mức độ tương đồng** để tìm ra các customer segments khác biệt.

### Machine Learning Type

- Đây là bài toán **Unsupervised Learning**.
- Không có target label.
- Không dùng classification hoặc regression.
- Model không dự đoán nhãn có sẵn; model tự tìm cấu trúc cụm dựa trên feature similarity.

### Input Features Chính

| Feature | Meaning | Usage |
| --- | --- | --- |
| `age` | Tuổi khách hàng | Numeric clustering feature |
| `gender` | Giới tính | One-hot optional feature để so sánh |
| `annual_income_k` | Thu nhập hằng năm, đơn vị nghìn USD | Numeric clustering feature |
| `spending_score` | Điểm chi tiêu 1-100 | Numeric clustering feature |

### Expected Output

Kết quả cuối cần có:

- Số lượng cụm hợp lý.
- Bảng profile từng cụm.
- Tên segment dễ hiểu.
- Biểu đồ scatter/dendrogram/heatmap để kiểm soát trực quan.
- Metric đánh giá clustering như silhouette, Davies-Bouldin và Calinski-Harabasz.
"""
        ),
        _code(
            """
import json
from pathlib import Path

import joblib
import pandas as pd

root = Path.cwd()
if not (root / "data").exists():
    root = Path("..").resolve()
pd.set_option("display.max_columns", 100)
"""
        ),
        _markdown(
            """
## 2. Raw Data Sources

Chương trình sử dụng hai nguồn raw data:

1. **Mall Customers**: nguồn chính đúng đề bài, có age/gender/income/spending score.
2. **UCI Online Retail**: nguồn mở rộng dạng transaction log, dùng để tạo RFM features.

Rule quan trọng: không merge hai nguồn này theo dòng vì customer ID không tương thích.
"""
        ),
        _code(
            """
data_sources = pd.read_csv(root / "data_sources" / "data_links.csv")
data_sources.tail(10)
"""
        ),
        _markdown(
            """
## 3. Data Cleaning and Validation

### Mall Customers Cleaning

- Kiểm tra đủ raw columns.
- Chuẩn hóa tên cột thành `customer_id`, `gender`, `age`, `annual_income_k`, `spending_score`.
- Kiểm tra missing values, duplicate rows, duplicate customer ID.
- Validate range: age > 0, annual income >= 0, spending score nằm trong 1-100.

### Online Retail Cleaning

- Loại invoice cancellation.
- Loại quantity <= 0.
- Loại unit price <= 0.
- Loại missing customer ID.
- Loại malformed invoice date.
- Tạo `line_total = quantity * unit_price`.
- Aggregate thành RFM theo customer.
"""
        ),
        _code(
            """
mall_quality = json.loads((root / "reports" / "data_quality_report.json").read_text(encoding="utf-8"))
online_quality = json.loads((root / "reports" / "online_retail" / "online_retail_quality_report.json").read_text(encoding="utf-8"))

mall_quality, online_quality
"""
        ),
        _markdown(
            """
## 4. Load Processed Data

Các file processed là dữ liệu đã được làm sạch và dùng cho EDA/training.
"""
        ),
        _code(
            """
mall_clean = pd.read_csv(root / "data" / "processed" / "mall_customers_clean.csv")
mall_clustered = pd.read_csv(root / "data" / "processed" / "mall_customers_clustered.csv")
online_rfm = pd.read_csv(root / "data" / "processed" / "online_retail_rfm.csv")
online_clustered = pd.read_csv(root / "data" / "processed" / "online_retail_rfm_clustered.csv")

print("Mall clean shape:", mall_clean.shape)
print("Mall clustered shape:", mall_clustered.shape)
print("Online Retail RFM shape:", online_rfm.shape)
print("Online Retail clustered shape:", online_clustered.shape)
mall_clean.head()
"""
        ),
        _markdown(
            """
## 5. Exploratory Data Analysis - Mall Customers

Phần này quan sát phân bố dữ liệu và quan hệ giữa các feature chính trước khi clustering.
"""
        ),
    ]
    cells.extend(_image_cell(*figure) for figure in MALL_FIGURES[:9])
    cells.extend(
        [
            _markdown(
                """
## 6. Feature Scaling

Clustering dựa trên khoảng cách, nên các feature numeric cần được scale trước khi train model. Nếu không scale, feature có thang đo lớn hơn như annual income có thể chi phối khoảng cách.

Chương trình dùng:

- `StandardScaler`
- Feature set 1: `numeric_only`
- Feature set 2: `numeric_plus_gender`
- `customer_id` luôn bị loại khỏi feature matrix.
"""
            ),
            _markdown(
                """
## 7. Train Clustering Models

Các thuật toán được thử nghiệm:

- K-Means với `k=2..10`.
- Agglomerative/Hierarchical Clustering với nhiều linkage.
- DBSCAN với grid `eps` và `min_samples`.

Metrics:

- `silhouette_score`: càng cao càng tốt.
- `davies_bouldin_score`: càng thấp càng tốt.
- `calinski_harabasz_score`: càng cao càng tốt.
- `inertia`: dùng cho K-Means elbow plot.
- `noise_ratio`: dùng cho DBSCAN.
"""
            ),
            _code(
                """
mall_metrics = pd.read_csv(root / "reports" / "metrics" / "clustering_metrics.csv")
online_metrics = pd.read_csv(root / "reports" / "online_retail" / "metrics" / "clustering_metrics.csv")

mall_metrics[mall_metrics["valid_candidate"]].sort_values("silhouette_score", ascending=False).head(10)
"""
            ),
            _markdown(
                """
## 8. Mall Customers Model Evaluation Figures

Các hình dưới đây dùng để kiểm soát quá trình chọn số cụm và kiểm tra kết quả phân cụm cuối.
"""
            ),
        ]
    )
    cells.extend(_image_cell(*figure) for figure in MALL_FIGURES[9:])
    cells.extend(
        [
            _markdown(
                """
## 9. Best Mall Customers Model and Segment Profile

Model cuối cho dataset chính được chọn theo metric và khả năng giải thích. Trong bài này, model chính là kết quả cần trình bày cho đề Practice Exercise 3.
"""
            ),
            _code(
                """
mall_best = json.loads((root / "reports" / "metrics" / "best_model.json").read_text(encoding="utf-8"))
mall_profile = pd.read_csv(root / "reports" / "tables" / "cluster_profile.csv")

mall_best_summary = {
    "algorithm": mall_best["algorithm"],
    "feature_set": mall_best["feature_set"],
    "params": mall_best["params"],
    "n_clusters": mall_best["n_clusters"],
    "silhouette_score": mall_best["silhouette_score"],
    "davies_bouldin_score": mall_best["davies_bouldin_score"],
    "calinski_harabasz_score": mall_best["calinski_harabasz_score"],
}
mall_best_summary
"""
            ),
            _code(
                """
mall_profile
"""
            ),
            _markdown(
                """
### Mall Segment Interpretation

- `Average Income - Average Spending`: nhóm khách hàng trung bình, phù hợp cho chiến lược duy trì.
- `High Income - High Spending`: nhóm giá trị cao, có thể ưu tiên chăm sóc hoặc loyalty program.
- `Young High Spenders`: nhóm trẻ có spending score cao, phù hợp chiến dịch sản phẩm mới hoặc promotion.
- `High Income - Low Spending`: nhóm có tiềm năng nhưng chưa chi tiêu nhiều, cần phân tích động lực mua hàng.
- `Low Income - Low Spending`: nhóm chi tiêu thấp, phù hợp ưu đãi nhỏ hoặc sản phẩm phổ thông.
"""
            ),
            _markdown(
                """
## 10. Extended EDA - Online Retail RFM

Track Online Retail dùng dữ liệu giao dịch thô để tạo RFM features:

- `recency_days`: số ngày từ lần mua gần nhất.
- `frequency`: số invoice duy nhất.
- `monetary_value`: tổng giá trị mua hàng.
- `average_order_value`: giá trị trung bình mỗi invoice.
"""
            ),
        ]
    )
    cells.extend(_image_cell(*figure) for figure in ONLINE_RETAIL_FIGURES)
    cells.extend(
        [
            _markdown(
                """
## 11. Online Retail RFM Model and Profile

Track này không thay thế bài Mall Customers, mà chứng minh chương trình có thể mở rộng sang raw transaction data và tạo segmentation bằng RFM.
"""
            ),
            _code(
                """
online_best = json.loads((root / "reports" / "online_retail" / "metrics" / "best_model.json").read_text(encoding="utf-8"))
online_profile = pd.read_csv(root / "reports" / "online_retail" / "tables" / "cluster_profile.csv")

online_best_summary = {
    "algorithm": online_best["algorithm"],
    "feature_set": online_best["feature_set"],
    "params": online_best["params"],
    "n_clusters": online_best["n_clusters"],
    "silhouette_score": online_best["silhouette_score"],
}
online_best_summary
"""
            ),
            _code(
                """
online_profile
"""
            ),
            _markdown(
                """
## 12. Saved Model Artifacts

Chương trình lưu model metadata, scaler, feature names, labels và đường dẫn profile/metrics bằng Joblib.
"""
            ),
            _code(
                """
mall_model = joblib.load(root / "models" / "customer_segmentation_pipeline.joblib")
online_model = joblib.load(root / "models" / "online_retail_segmentation_pipeline.joblib")

{
    "mall_model_keys": sorted(mall_model.keys()),
    "online_model_keys": sorted(online_model.keys()),
}
"""
            ),
            _markdown(
                """
## 13. Model Tests

Các phần test đã được viết trong thư mục `tests/`:

- Schema validation cho Mall Customers.
- Online Retail cleaning và RFM builder.
- Clustering edge cases.
- Model artifact metadata.
- App import không retrain.
- Notebook sections.

Lệnh kiểm tra:

```powershell
python -m compileall -q config.py main.py app.py src tests
python -m pytest -q
python -m json.tool notebooks/99_customer_segmentation_workflow.ipynb > $null
```
"""
            ),
            _markdown(
                """
## 14. Python UI

Giao diện được viết bằng Streamlit trong file `app.py`.

Chạy UI:

```powershell
streamlit run app.py
```

UI chỉ đọc artifact đã sinh sẵn:

- best model JSON
- metrics CSV
- cluster profile CSV
- clustered data preview
- final Markdown report

UI không retrain model khi load trang.
"""
            ),
            _markdown(
                """
## 15. Final Conclusion

Chương trình đã bám đúng trọng tâm đề bài:

- Có raw data chuẩn.
- Có bước làm sạch dữ liệu.
- Có EDA với hình ảnh đầy đủ.
- Có feature scaling.
- Có train nhiều clustering algorithms.
- Có evaluation metrics.
- Có model tests.
- Có Python UI.
- Có final Jupyter notebook tổng hợp toàn bộ workflow.

Kết quả chính để trình bày là Mall Customers segmentation. Online Retail RFM là phần mở rộng để tăng độ đầy đủ về raw data và workflow thực tế.
"""
            ),
        ]
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    FINAL_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    FINAL_NOTEBOOK.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    return FINAL_NOTEBOOK
