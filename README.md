# Arvato Customer Acquisition – MLE Capstone

Udacity Machine Learning Engineer capstone: customer segmentation and campaign response prediction for a mail-order organic products company.

## Problem

**How can the company acquire new customers more efficiently?**

Use demographic data from existing customers and the general population to identify who is most likely to respond to marketing campaigns, enabling targeted advertising instead of broad outreach.

---

## Submission Checklist

Before submitting, ensure you have:

| Required | File | Notes |
|----------|------|--------|
| ✓ | **proposal.pdf** | From `proposal.md`. Generate: `pandoc proposal.md -o proposal.pdf` |
| ✓ | **report.pdf** | Project report (5 stages, 9–15 pages). From `report.md`: `pandoc report.md -o report.pdf` |
| ✓ | **Python code** | `Arvato_Customer_Acquisition.ipynb`, `src/preprocessing.py`, `run_pipeline.py` |
| ✓ | **README** | This file (software, libraries, setup instructions) |
| ✓ | **Data** | Too large to include; obtain from Udacity Bertelsmann Capstone (see Data section below) |

Include your proposal review link in the student submission notes if applicable.

---

## Deliverables

- **Proposal** – `proposal.md` → convert to **proposal.pdf** for submission
- **Project report** – `report.md` → convert to **report.pdf** (five stages: Definition, Analysis, Design, Implementation, Results)
- **Notebook** – `Arvato_Customer_Acquisition.ipynb` (EDA, segmentation, supervised model)
- **Code** – `src/preprocessing.py`, `run_pipeline.py`
- **Blog post** – `BLOG.md` (optional supporting material)

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
├── proposal.md                        # Capstone proposal → proposal.pdf
├── report.md                          # Project report (5 stages) → report.pdf
├── BLOG.md                            # Technical blog post
├── requirements.txt
└── README.md
```

## Approach

1. **Preprocessing** – Replace missing codes, drop high-missing cols/rows, encode categoricals, impute, standardize.
2. **Segmentation** – PCA + MiniBatchKMeans on general population; compare cluster proportions for customers vs population.
3. **Response prediction** – Train classifiers (Logistic Regression, Random Forest, XGBoost) on MAILOUT_TRAIN; predict probabilities for MAILOUT_TEST.
4. **Benchmarks** – Compare against majority-class and random baselines (AUC-ROC, AUC-PR).

## Software and Libraries

- **Python:** 3.8 or 3.10+
- **Core:** pandas, numpy, scikit-learn (preprocessing, PCA, clustering, classification, metrics)
- **Visualization:** matplotlib, seaborn
- **Optional:** xgboost (for XGBoost model comparison)
- **Development:** Jupyter (for the notebook)

All dependencies are listed in `requirements.txt`. Install with: `pip install -r requirements.txt`.

## Generating PDFs for Submission

- **proposal.pdf:** `pandoc proposal.md -o proposal.pdf`
- **report.pdf:** `pandoc report.md -o report.pdf`

If pandoc fails (e.g., missing LaTeX), install `texlive-latex-extra` or export from a Markdown editor / browser (e.g., open the HTML export and Print → Save as PDF).
