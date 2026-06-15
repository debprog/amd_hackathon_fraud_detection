import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from feature_engineering import load_data, engineer_features, get_feature_columns


def main():
    df = load_data()
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)

    X = df[feature_cols]
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.1f}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # --- evaluation ---
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # --- save model ---
    joblib.dump(model, "fraud_model.pkl")
    joblib.dump(feature_cols, "feature_cols.pkl")
    print("\nSaved fraud_model.pkl and feature_cols.pkl")

    # --- build SHAP explainer (saved for reuse at inference) ---
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, "shap_explainer.pkl")
    print("Saved shap_explainer.pkl")

    # quick sanity check: explain first 3 test rows
    sample = X_test.iloc[:3]
    shap_values = explainer.shap_values(sample)
    for i in range(3):
        contrib = pd.Series(shap_values[i], index=feature_cols).sort_values(
            key=np.abs, ascending=False
        )
        print(f"\nTop reasons for row {i} (prob={y_prob[i]:.2f}, true={y_test.iloc[i]}):")
        print(contrib.head(3))


if __name__ == "__main__":
    main()
