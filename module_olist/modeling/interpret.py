import pandas as pd
from loguru import logger
import shap as sh
from scipy import sparse

def prepare_data_for_shap(pipeline, X):
    """
    Prepares the data
    for the explainer

    Args:
        pipeline: pipeline from which the preprocessing transformations are retrieved
        X: data to be transformed
    Return:
        X_transformed: data transformed by the preprocessor
    """
    preprocessor = pipeline.named_steps['preprocessor']

    #Applies pipeline transformations to the data
    X_transformed = preprocessor.transform(X)

    #Checks for sparse data caused by One Hot Encoding and reduces the matrix to an array
    if sparse.issparse(X_transformed):
        X_transformed - X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    X_transformed = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

    return X_transformed

def create_explainer(pipeline):

    """
    Creates an explainer to 
    understand (and explain) the model's
    predictions

    Args:
        pipeline: the pipeline from which the model is retrieved
    """

    model = pipeline.named_steps['model']

    explainer = sh.TreeExplainer(model)

    return explainer

def calculate_data_shap_values(explainer, X_transformed):
    """
    Calculates how much 
    each feature contributes
    towards predictions
    """

    shap_values = explainer(X_transformed)

    return shap_values