# Arvato Customer Acquisition – MLE Capstone

Udacity Machine Learning Engineer capstone: customer segmentation and campaign response prediction for a mail-order organic products company.

## Problem

**How can the company acquire new customers more efficiently?**

Use demographic data from existing customers and the general population to identify who is most likely to respond to marketing campaigns, enabling targeted advertising instead of broad outreach.

## Deliverables

- **Proposal** – `proposal.md` (convert to PDF for submission)
- **Notebook** – `Arvato_Customer_Acquisition.ipynb` (EDA, segmentation, supervised model)
- **Code** – `src/preprocessing.py`, `run_pipeline.py`
- **Blog post** – `BLOG.md`

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (generates kaggle_submission.csv)
python run_pipeline.py

# Faster run with smaller subsets
python run_pipeline.py --quick
```

Or open `Arvato_Customer_Acquisition.ipynb` and run all cells.

## Data

Place these files in `data/` (from Udacity Bertelsmann Capstone):

- `Udacity_AZDIAS_052018.csv` – General population (~891k rows)
- `Udacity_CUSTOMERS_052018.csv` – Established customers (~192k)
- `Udacity_MAILOUT_052018_TRAIN.csv` – Campaign targets with labels (~43k)
- `Udacity_MAILOUT_052018_TEST.csv` – Test set for Kaggle (~43k)

Data use is restricted to this project; deletion required within 2 weeks of completion (AZ Direct GmbH terms).

## Project Structure

```
├── Arvato_Customer_Acquisition.ipynb  # Main notebook
├── run_pipeline.py                    # CLI pipeline
├── src/
│   ├── __init__.py
│   └── preprocessing.py               # Load, clean, encode, impute
├── proposal.md                        # Capstone proposal
├── BLOG.md                            # Technical blog post
├── requirements.txt
└── README.md
```

## Approach

1. **Preprocessing** – Replace missing codes, drop high-missing cols/rows, encode categoricals, impute, standardize.
2. **Segmentation** – PCA + MiniBatchKMeans on general population; compare cluster proportions for customers vs population.
3. **Response prediction** – Train classifiers (Logistic Regression, Random Forest, XGBoost) on MAILOUT_TRAIN; predict probabilities for MAILOUT_TEST.
4. **Benchmarks** – Compare against majority-class and random baselines (AUC-ROC, AUC-PR).

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- xgboost (optional, for best model)
