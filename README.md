# Lap 3 - Customer Segmentation Clustering

This repository contains the Practice Exercise 3 customer segmentation project.

## Scope

- Primary task: cluster Mall Customers using age, gender, annual income, and spending score.
- Extended raw-data track: UCI Online Retail RFM segmentation.
- Learning type: unsupervised learning.
- No supervised target label is used.

## Main Commands

```powershell
python -m pip install -r requirements.txt
python main.py --run-all
python -m pytest -q
streamlit run app.py
```

## Key Outputs

- `notebooks/99_customer_segmentation_workflow.ipynb`
- `reports/final_customer_segments.md`
- `reports/online_retail/online_retail_customer_segments.md`
- `models/customer_segmentation_pipeline.joblib`
- `models/online_retail_segmentation_pipeline.joblib`

## Validation

Current validation passed with:

- `python -m compileall -q config.py main.py app.py src tests`
- `python -m pytest -q`
- `python -m json.tool notebooks/99_customer_segmentation_workflow.ipynb`
