import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import joblib


def train_logistic_model(df: pd.DataFrame):
    """Train and evaluate a Logistic Regression model on insurance fraud data."""

    y = df['fraud_reported']
    X = df.drop(columns=['fraud_reported'])

    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    num_cols = X.select_dtypes(include=['number', 'bool']).columns

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ])

    X_scaled = preprocessor.fit_transform(X)

    base_model = LogisticRegression(
        C=0.1, class_weight={0: 1, 1: 4},
        solver='liblinear', max_iter=1000, random_state=42
    )

    selector = RFE(base_model, n_features_to_select=20)
    X_reduced = selector.fit_transform(X_scaled, y)

    model = LogisticRegression(
        C=0.1, class_weight={0: 1, 1: 4},
        solver='liblinear', max_iter=1000, random_state=42
    )

    # K-fold validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_reduced, y, cv=kf, scoring='roc_auc')

    print(f"Mean CV ROC AUC: {cv_scores.mean():.4f}")

    model.fit(X_reduced, y)

    y_pred_prob = model.predict_proba(X_reduced)[:, 1]
    print(f"Final ROC AUC: {roc_auc_score(y, y_pred_prob):.4f}")
    print(f"Average Precision: {average_precision_score(y, y_pred_prob):.4f}")
    print("\nClassification Report:\n", classification_report(y, (y_pred_prob > 0.5).astype(int)))

    # Save model
    joblib.dump(model, "logistic_model.joblib")
    joblib.dump(preprocessor, "preprocessor.joblib")
    joblib.dump(selector, "feature_selector.joblib")

    return model, preprocessor, selector
