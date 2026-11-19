import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def get_xgboost_classifier(config, early_stopping: bool = True):
    """Instantiate regularized multi-class XGBoost classifier."""
    xgb_cfg = config["training"]["xgboost"]
    early_stop = xgb_cfg["early_stopping_rounds"] if early_stopping else None
    return xgb.XGBClassifier(
        objective=xgb_cfg["objective"],
        num_class=xgb_cfg["num_class"],
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        reg_lambda=xgb_cfg["reg_lambda"],
        eval_metric=xgb_cfg["eval_metric"],
        random_state=config["training"]["random_seed"],
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=early_stop,
    )


def get_baseline_classifiers(random_state: int = 42):
    """Instantiate benchmark ML heads for internal ablation study."""
    return {
        "SVM_RBF": SVC(
            C=10.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state,
        ),
        "LightGBM": lgb.LGBMClassifier(
            objective="multiclass",
            num_class=4,
            n_estimators=400,
            learning_rate=0.05,
            n_jobs=-1,
            verbosity=-1,
            random_state=random_state,
        ),
    }
