#!/usr/bin/env python3
"""
Run the full Arvato customer acquisition pipeline and generate Kaggle submission.
Usage: python run_pipeline.py [--quick]
  --quick: use smaller subsets for faster execution
"""
import argparse
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, silhouette_score

from src.preprocessing import load_data, preprocess_pipeline

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def main(quick=False):
    print("Loading data...")
    if quick:
        azdias, customers, mailout_train, mailout_test = load_data(
            'data', nrows_azdias=50000, nrows_customers=25000
        )
        sample_azdias, sample_customers = 20000, 10000
    else:
        azdias, customers, mailout_train, mailout_test = load_data('data')
        sample_azdias, sample_customers = 100000, 50000

    print("Preprocessing...")
    proc = preprocess_pipeline(
        azdias, customers, mailout_train, mailout_test,
        sample_azdias=sample_azdias, sample_customers=sample_customers
    )
    X_azdias = proc['azdias']
    X_customers = proc['customers']
    X_train = proc['mailout_train']
    X_test = proc['mailout_test']
    y_train = proc['y_train']

    scaler = StandardScaler()
    X_azdias_scaled = scaler.fit_transform(X_azdias)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Segmentation
    print("Running PCA and K-means...")
    pca = PCA(n_components=50, random_state=42)
    X_azdias_pca = pca.fit_transform(X_azdias_scaled)
    X_customers_pca = pca.transform(scaler.transform(X_customers))

    kmeans = MiniBatchKMeans(n_clusters=8, random_state=42, batch_size=1000, n_init=3)
    azdias_labels = kmeans.fit_predict(X_azdias_pca)
    customer_labels = kmeans.predict(X_customers_pca)

    sil = silhouette_score(X_azdias_pca, azdias_labels)
    print(f"  Silhouette score: {sil:.3f}")

    azdias_props = np.bincount(azdias_labels, minlength=8) / len(azdias_labels)
    customer_props = np.bincount(customer_labels, minlength=8) / len(customer_labels)
    ratio = customer_props / (azdias_props + 1e-8)
    print(f"  Over-represented clusters: {np.where(ratio > 1.2)[0].tolist()}")

    # Benchmarks
    print("\nBenchmarks:")
    print(f"  Majority class AUC: {roc_auc_score(y_train, np.zeros_like(y_train)):.3f}")
    np.random.seed(42)
    print(f"  Random AUC: {roc_auc_score(y_train, np.random.rand(len(y_train))):.3f}")

    # Models - compare, use Logistic Regression for submission (generalizes better than RF)
    models = [
        ('Logistic Regression', LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
    ]
    if HAS_XGB:
        models.append(('XGBoost', xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='auc', random_state=42)))

    for name, model in models:
        model.fit(X_train_scaled, y_train)
        pred = model.predict_proba(X_train_scaled)[:, 1]
        auc = roc_auc_score(y_train, pred)
        ap = average_precision_score(y_train, pred)
        print(f"  {name}: AUC-ROC={auc:.3f}, AUC-PR={ap:.3f}")

    best_model = models[0][1]  # Logistic Regression
    best_name = 'Logistic Regression'

    # Kaggle submission
    test_pred = best_model.predict_proba(X_test_scaled)[:, 1]
    submission = pd.DataFrame({'LNR': mailout_test['LNR'], 'RESPONSE': test_pred})
    submission.to_csv('kaggle_submission.csv', index=False)
    print(f"\nSaved kaggle_submission.csv using {best_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Use smaller data subsets')
    args = parser.parse_args()
    main(quick=args.quick)
