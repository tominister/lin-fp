from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from models.metrics import compute_metrics

SEED = 42

def run_xgboost(X_train, y_train, X_val, y_val, X_recovery, y_recovery, args):
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=5,
        max_df=0.9,
        stop_words="english",
        sublinear_tf=True,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_recovery_vec = vectorizer.transform(X_recovery)

    model = XGBClassifier(
        n_estimators=180,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=5,
        gamma=1.0,
        subsample=0.65,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=4,
    )
    model.fit(X_train_vec, y_train)

    val_pred = model.predict(X_val_vec)
    recovery_pred = model.predict(X_recovery_vec)

    return {
        "welfake_validation_20pct": compute_metrics(y_val, val_pred),
        "recovery_external_test": compute_metrics(y_recovery, recovery_pred),
        "config": {
            "vectorizer": {
                "max_features": args.max_features,
                "ngram_range": [1, args.ngram_max],
                "min_df": 5,
                "max_df": 0.9,
                "stop_words": "english",
                "sublinear_tf": True,
            },
            "model": {
                "type": "XGBClassifier",
                "n_estimators": 180,
                "learning_rate": 0.05,
                "max_depth": 3,
                "min_child_weight": 5,
                "gamma": 1.0,
                "subsample": 0.65,
                "colsample_bytree": 0.6,
                "reg_alpha": 0.5,
                "reg_lambda": 2.0,
            },
        },
    }
