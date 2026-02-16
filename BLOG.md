# Arvato Customer Acquisition: Data-Driven Marketing for Mail-Order Organic Products

*A technical walkthrough of unsupervised segmentation and supervised response prediction for the Udacity MLE Capstone*

---

## Problem and Business Context

A mail-order company selling organic products wants to acquire new customers more efficiently. Instead of targeting everyone in Germany with a broad marketing campaign, they want to identify and reach people most likely to become new customers. This is a classic **targeted marketing** problem: reduce cost, increase return on marketing spend, and move from intuition-based to data-driven decision making.

Arvato Financial Services (Bertelsmann) provides demographic and lifestyle data to support exactly this kind of analysis. The mission: *make decisions based on data instead of gut feel*.

---

## Data

Four datasets from AZ Direct GmbH / Arvato (Udacity Bertelsmann Capstone):

| Dataset | Rows | Role |
|---------|------|------|
| AZDIAS | ~891k | General population of Germany |
| CUSTOMERS | ~192k | Established customers of the mail-order company |
| MAILOUT_TRAIN | ~43k | Prior campaign targets with response labels (0/1) |
| MAILOUT_TEST | ~43k | Test campaign targets (for Kaggle submission) |

All share the same demographic attribute schema (366+ columns): demographics, financial behavior, lifestyle, household, and regional features. CUSTOMERS adds three metadata columns (CUSTOMER_GROUP, ONLINE_PURCHASE, PRODUCT_GROUP) that are dropped for alignment.

---

## Approach

### 1. Preprocessing

- **Missing codes:** Replace DIAS-style missing/unknown codes (-1, 0, -2, -9, 9) with NaN
- **High-missing features:** Drop columns with >40% missing values
- **High-missing rows:** Drop rows with >30% missing
- **Categoricals:** Label-encode object columns, mapping unseen values to -1
- **Imputation:** Median imputation (fit on AZDIAS, apply to all datasets)
- **Standardization:** StandardScaler for clustering and models

### 2. Customer Segmentation (Unsupervised)

- **PCA:** Reduce dimensionality (e.g., 50 components) to capture most variance and reduce noise
- **K-means (MiniBatchKMeans):** Cluster the general population; map customers to the same cluster space
- **Over/under-representation:** Compare cluster proportions in customers vs. general population
  - Clusters with higher customer proportion = “customer-like” segments
  - Clusters with lower customer proportion = segments to deprioritize

**Evaluation:** Silhouette score for cluster quality; proportion ratios for business interpretation.

### 3. Campaign Response Prediction (Supervised)

- **Target:** Predict probability of response (1) vs. non-response (0) for campaign targets
- **Models:** Logistic regression, Random Forest, XGBoost (when available)
- **Class imbalance:** Use `class_weight='balanced'` or equivalent

**Benchmarks:**
- Majority class: always predict 0 → AUC-ROC = 0.5
- Random predictor → AUC-ROC ≈ 0.5
- A useful model should clearly exceed these baselines

**Evaluation:** AUC-ROC (ranking quality), AUC-PR (important for imbalanced data); Kaggle competition metric for the test set.

---

## Results and Takeaways

- **Segmentation:** Clusters reveal which demographic profiles are over- or under-represented among customers, informing targeting and messaging.
- **Response model:** A trained classifier (e.g., XGBoost or Random Forest) produces response probabilities for each individual. Marketing can focus on the top-ranked subset.
- **Deliverable:** `kaggle_submission.csv` with LNR and RESPONSE (probability) for each test individual, ready for competition upload.

---

## Reproducibility

- Python 3.10, pandas, scikit-learn, (optional) XGBoost
- Run the Jupyter notebook `Arvato_Customer_Acquisition.ipynb` or `python run_pipeline.py`
- Use `nrows_azdias` / `nrows_customers` or `--quick` for faster iteration on subsets
