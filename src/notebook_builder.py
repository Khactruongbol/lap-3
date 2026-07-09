from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    BEST_MODEL_JSON,
    CLUSTER_PROFILE_CSV,
    DATA_BALANCE_REPORT,
    DATA_QUALITY_REPORT,
    DATA_SOURCE_LOG,
    FINAL_NOTEBOOK,
    ONLINE_RETAIL_BALANCE_REPORT,
    ONLINE_RETAIL_BEST_MODEL_JSON,
    ONLINE_RETAIL_CLUSTER_PROFILE_CSV,
    ONLINE_RETAIL_QUALITY_REPORT,
)


def _markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


def _table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No data available._"
    data = df.copy()
    if columns is not None:
        data = data[[column for column in columns if column in data.columns]]
    if max_rows is not None:
        data = data.head(max_rows)
    headers = data.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in data.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _best_model_summary(best: dict[str, Any]) -> str:
    if not best:
        return "_Best model metadata is not available._"
    rows = pd.DataFrame(
        [
            {"Metric": "Algorithm", "Value": best.get("algorithm", "N/A")},
            {"Metric": "Feature set", "Value": best.get("feature_set", "N/A")},
            {"Metric": "Parameters", "Value": best.get("params", "N/A")},
            {"Metric": "Number of clusters", "Value": best.get("n_clusters", "N/A")},
            {"Metric": "Silhouette score", "Value": best.get("silhouette_score", "N/A")},
            {"Metric": "Davies-Bouldin score", "Value": best.get("davies_bouldin_score", "N/A")},
            {"Metric": "Calinski-Harabasz score", "Value": best.get("calinski_harabasz_score", "N/A")},
        ]
    )
    return _table(rows)


def _balance_summary(mall_balance: dict[str, Any], online_balance: dict[str, Any]) -> str:
    rows = pd.DataFrame(
        [
            {
                "Dataset": "Mall Customers",
                "Original rows": mall_balance.get("original_rows", "N/A"),
                "Balanced rows": mall_balance.get("balanced_rows", "N/A"),
                "Removed rows": mall_balance.get("removed_rows", "N/A"),
                "Synthetic rows added": mall_balance.get("synthetic_rows_added", "N/A"),
                "Strategy": mall_balance.get("strategy", "N/A"),
            },
            {
                "Dataset": "Online Retail RFM",
                "Original rows": online_balance.get("original_rows", "N/A"),
                "Balanced rows": online_balance.get("balanced_rows", "N/A"),
                "Removed rows": online_balance.get("removed_rows", "N/A"),
                "Synthetic rows added": online_balance.get("synthetic_rows_added", "N/A"),
                "Strategy": online_balance.get("strategy", "N/A"),
            },
        ]
    )
    return _table(rows)


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
    mall_quality = _read_json(DATA_QUALITY_REPORT)
    online_quality = _read_json(ONLINE_RETAIL_QUALITY_REPORT)
    mall_balance = _read_json(DATA_BALANCE_REPORT)
    online_balance = _read_json(ONLINE_RETAIL_BALANCE_REPORT)
    mall_best = _read_json(BEST_MODEL_JSON)
    online_best = _read_json(ONLINE_RETAIL_BEST_MODEL_JSON)
    sources = _read_csv(DATA_SOURCE_LOG)
    mall_profile = _read_csv(CLUSTER_PROFILE_CSV)
    online_profile = _read_csv(ONLINE_RETAIL_CLUSTER_PROFILE_CSV)

    cells = [
        _markdown(
            """
# Final Notebook - Customer Segmentation Clustering

Notebook này chỉ dùng **hình ảnh, bảng tóm tắt và giải thích**, không chứa code cell. Mục tiêu là giúp kiểm soát toàn bộ chương trình và model một cách trực quan như một bản báo cáo Lab 3 hoàn chỉnh.

Dataset chính để chấm bài là **Mall Customers** vì có đúng các feature trong đề: `Age`, `Gender`, `Annual Income (k$)`, `Spending Score (1-100)`. Dataset **UCI Online Retail** là phần mở rộng raw transaction data để tạo RFM segmentation, không merge trực tiếp với Mall Customers vì không cùng customer ID/schema.
"""
        ),
        _markdown(
            """
## 1. Define Problem

### Problem Statement

Ta có dữ liệu khách hàng của một cửa hàng bán lẻ gồm tuổi, giới tính, thu nhập hằng năm và spending score. Nhiệm vụ là **phân cụm khách hàng theo mức độ tương đồng** để tìm ra các customer segments khác biệt.

### Machine Learning Type

- Đây là bài toán **Unsupervised Learning**.
- Không có target label.
- Không dùng classification hoặc regression.
- Model không dự đoán nhãn có sẵn; model tự tìm cấu trúc cụm dựa trên feature similarity.

### Expected Output

- Số cụm hợp lý.
- Bảng profile từng cụm.
- Tên segment dễ hiểu.
- Hình scatter, dendrogram, heatmap để kiểm soát trực quan.
- Metric đánh giá clustering: silhouette, Davies-Bouldin, Calinski-Harabasz.
"""
        ),
        _markdown(
            f"""
## 2. Data Sources

Chương trình sử dụng hai nguồn raw data:

{_table(sources, ["source_name", "url", "access_method", "download_date", "local_file"], max_rows=10)}

**Quy tắc quan trọng:** Mall Customers là dataset chính đúng đề bài. Online Retail chỉ là track mở rộng RFM, không merge theo dòng với Mall Customers.
"""
        ),
        _markdown(
            f"""
## 3. Data Cleaning Summary

### Mall Customers

- Raw shape: `{mall_quality.get("raw_shape", "N/A")}`
- Clean shape: `{mall_quality.get("clean_shape", "N/A")}`
- Missing values raw: `{mall_quality.get("missing_values_raw", "N/A")}`
- IQR outlier counts: `{mall_quality.get("outlier_counts_iqr", "N/A")}`

### Online Retail

- Raw shape: `{online_quality.get("raw_shape", "N/A")}`
- Clean shape: `{online_quality.get("clean_shape", "N/A")}`
- Removed rows: `{online_quality.get("removed_rows", "N/A")}`
- Issue counts: `{online_quality.get("issue_counts", "N/A")}`
- Customer count after cleaning: `{online_quality.get("customer_count", "N/A")}`
"""
        ),
        _markdown(
            f"""
## 4. Workflow and Data Balancing

Workflow của chương trình được kiểm soát theo đúng trọng tâm bài toán clustering:

1. **Data Acquisition:** lấy Mall Customers làm dataset chính và Online Retail làm raw transaction extension.
2. **Data Cleaning:** chuẩn hóa schema, loại missing/duplicate/invalid transaction và lưu quality reports.
3. **Data Balancing:** kiểm tra outlier/skew trước khi train. Chương trình không sinh synthetic customers; nếu feature lệch quá mạnh, row cực trị chỉ bị loại khỏi training dataset, còn raw/clean data vẫn được giữ lại.
4. **Feature Scaling:** scale numeric features bằng `StandardScaler`; Online Retail có thêm `rfm_log` để giảm ảnh hưởng độ lệch.
5. **Clustering:** train K-Means, Agglomerative/Hierarchical và DBSCAN trên các feature sets hợp lệ.
6. **Evaluation:** so sánh silhouette, Davies-Bouldin, Calinski-Harabasz, số cụm và khả năng giải thích.
7. **Artifacts:** xuất model `.joblib`, metrics CSV/JSON, cluster profile tables, figures, Streamlit UI và notebook báo cáo cuối.

### Balance Summary

{_balance_summary(mall_balance, online_balance)}
"""
        ),
        _image_cell(
            "Mall Customers Balance: Before vs After",
            "../reports/figures/data_balance_boxplots.png",
            "Hình này so sánh phân bố numeric features trước và sau bước cân bằng. Mall Customers chỉ loại các điểm IQR outlier khỏi training dataset, không sửa raw data.",
        ),
        _image_cell(
            "Online Retail RFM Balance: Before vs After",
            "../reports/online_retail/figures/rfm_balance_boxplots.png",
            "Hình này dùng trục log để thấy rõ RFM upper-tail outliers trước và sau cân bằng. Các điểm quá cực trị được loại khỏi training để model không bị chi phối bởi một vài khách hàng giao dịch bất thường.",
        ),
        _markdown(
            """
## 5. Mall Customers EDA

Các hình dưới đây giải thích dữ liệu chính trước khi train model. Đây là phần cần kiểm soát để biết dữ liệu có missing/outlier/phân bố bất thường hay không.
"""
        ),
    ]

    cells.extend(_image_cell(*figure) for figure in MALL_FIGURES[:9])
    cells.append(
        _markdown(
            """
## 6. Feature Scaling and Model Training

Clustering dựa trên khoảng cách, vì vậy các feature numeric được chuẩn hóa bằng `StandardScaler`.

Feature sets được so sánh:

- `income_spending`: `annual_income_k`, `spending_score`; đây là feature set lõi cho phân khúc hành vi mua sắm của Mall Customers.
- `numeric_only`: `age`, `annual_income_k`, `spending_score`.
- `numeric_plus_gender`: numeric features + one-hot encoded gender.
- `rfm`: `recency_days`, `frequency`, `monetary_value`, `average_order_value` cho Online Retail.
- `rfm_log`: log-transformed RFM features để giảm ảnh hưởng của outlier giao dịch.

Algorithms được train:

- K-Means.
- Agglomerative/Hierarchical Clustering.
- DBSCAN.

Model cuối không được chọn chỉ theo metric. Chương trình ưu tiên model có metric tốt, số cụm trong khoảng dễ giải thích, feature set rõ ý nghĩa và hạn chế chọn `single linkage` khi có ứng viên gần tương đương vì `single linkage` dễ tạo hiệu ứng chaining.
"""
        )
    )
    cells.append(
        _markdown(
            f"""
## 7. Best Mall Customers Model

{_best_model_summary(mall_best)}

### Mall Customers Cluster Profile

{_table(mall_profile)}
"""
        )
    )
    cells.append(
        _markdown(
            """
## 8. Mall Customers Model Evaluation Figures

Các hình dưới đây dùng để kiểm soát quá trình chọn số cụm, so sánh model và giải thích kết quả phân cụm cuối.
"""
        )
    )
    cells.extend(_image_cell(*figure) for figure in MALL_FIGURES[9:])
    cells.append(
        _markdown(
            """
## 9. Mall Segment Interpretation

- `Average Income - Average Spending`: nhóm khách hàng trung bình, phù hợp cho chiến lược duy trì.
- `High Income - High Spending`: nhóm giá trị cao, có thể ưu tiên chăm sóc hoặc loyalty program.
- `Young High Spenders`: nhóm trẻ có spending score cao, phù hợp chiến dịch sản phẩm mới hoặc promotion.
- `High Income - Low Spending`: nhóm có tiềm năng nhưng chưa chi tiêu nhiều, cần phân tích động lực mua hàng.
- `Low Income - Low Spending`: nhóm chi tiêu thấp, phù hợp ưu đãi nhỏ hoặc sản phẩm phổ thông.
"""
        )
    )
    cells.append(
        _markdown(
            """
## 10. Online Retail RFM Extension

Track Online Retail dùng raw transaction data để tạo RFM features:

- `recency_days`: số ngày từ lần mua gần nhất.
- `frequency`: số invoice duy nhất.
- `monetary_value`: tổng giá trị mua hàng.
- `average_order_value`: giá trị trung bình mỗi invoice.

Track này chứng minh chương trình có thể mở rộng sang dữ liệu giao dịch thật, nhưng kết quả chính của đề vẫn là Mall Customers.
"""
        )
    )
    cells.append(
        _markdown(
            f"""
## 11. Best Online Retail RFM Model

{_best_model_summary(online_best)}

### Online Retail RFM Cluster Profile

{_table(online_profile)}
"""
        )
    )
    cells.extend(_image_cell(*figure) for figure in ONLINE_RETAIL_FIGURES)
    cells.append(
        _markdown(
            """
## 12. Program and Model Summary

### Source Structure

- `src/data_acquisition.py`: tải raw data Mall Customers và UCI Online Retail.
- `src/data_quality.py`: làm sạch và validate Mall Customers.
- `src/online_retail.py`: làm sạch transaction và tạo RFM features.
- `src/preprocessing.py`: scale feature sets.
- `src/clustering.py`: train K-Means, Agglomerative, DBSCAN.
- `src/evaluation.py`: tính metric và chọn model cuối.
- `src/model_train.py`: lưu model artifact `.joblib`.
- `app.py`: Streamlit UI chỉ đọc artifact, không retrain.

### Model Artifacts

- `models/customer_segmentation_pipeline.joblib`
- `models/online_retail_segmentation_pipeline.joblib`

### Validation Commands

- `python -m compileall -q config.py main.py app.py src tests`
- `python -m pytest -q`
- `python -m json.tool notebooks/99_customer_segmentation_workflow.ipynb`
"""
        )
    )
    cells.append(
        _markdown(
            """
## 13. Final Conclusion

Chương trình đã đáp ứng các yêu cầu:

- Tìm kiếm thêm raw data.
- Lọc sạch data.
- Train clustering models.
- Viết tests cho data/model/UI/notebook.
- Có giao diện Python bằng Streamlit.
- Có notebook cuối chỉ chứa hình ảnh và giải thích để kiểm soát chương trình/model.

Kết quả chính để trình bày là **Mall Customers segmentation**. Online Retail RFM là phần mở rộng để tăng độ đầy đủ về raw data và workflow thực tế.
"""
        )
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
