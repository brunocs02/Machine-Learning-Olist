from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

NUMERIC_FEATURES = ["promised_days", "item_count", "seller_count", "total_price", "total_freight",]
CATEGORICAL_FEATURES = ["purchase_month", "purchase_weekday", "customer_state", "purchase_hour"]

def create_preprocessor() -> ColumnTransformer:
    """
    Creates a preprocessor for the data.
    Transforms categorical features 
    using One Hot Econding.
    """
    return ColumnTransformer(
        transformers=[
            #transformação em colunas numéricas
            ("numeric", "passthrough", NUMERIC_FEATURES),
            #transformações em colunas categóricas
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )

def create_gradient_boosting_pipeline() -> Pipeline:
    """
    Creates pipeline using the preprocessor
    and Gradient Boosting model.
    """
    preprocessor = create_preprocessor()

    model = GradientBoostingClassifier(
        n_estimators=100, #número de árvores
        learning_rate=0.1,
        max_depth=3, #profundidade das árvores
        random_state=42
    )

    return Pipeline(
        steps=[("preprocessor", preprocessor), 
               ("model", model)]
    )

def create_xgboost_pipeline() -> Pipeline:
    """
    Creates pipeline using the preprocessor
        and XGBoost model.
    """

    preprocessor = create_preprocessor()

    model = XGBClassifier(n_estimators=100, #número de árvores
            learning_rate=0.1,
            max_depth=3, #profundidade das árvores
            random_state=42)
    
    return Pipeline(
        steps=[("preprocessor", preprocessor), 
               ("model", model)])

def create_lightgbm_pipeline() -> Pipeline:
    """
    Creates pipeline using the preprocessor
        and LightGBM model.
    """

    preprocessor = create_preprocessor()

    model = LGBMClassifier(n_estimators=100, #número de árvores
            learning_rate=0.1,
            max_depth=3, #profundidade das árvores
            random_state=42)
    
    return Pipeline(
        steps=[("preprocessor", preprocessor), 
               ("model", model)])