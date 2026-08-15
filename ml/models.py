from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_models(random_state=42):
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=random_state, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", random_state=random_state,
            n_jobs=-1),
        "LightGBM": LGBMClassifier(
            n_estimators=250, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=random_state, verbosity=-1),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64,32,16), max_iter=500,
                          early_stopping=True, random_state=random_state)),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=random_state)),
        "One-Class SVM": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced",
                                  random_state=random_state)),
        "Isolation Forest": None,
    }
