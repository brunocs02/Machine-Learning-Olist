import numpy as np
from loguru import logger
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def evaluate_model(model, name, threshold, X_test, y_test):

    """
    Final model evaluation
    using test sets
    """

    y_proba = model.predict_proba(X_test)[:,1]

    best_threshold = threshold
    best_f1 = -1
    best_precision = None
    best_recall = None

    y_pred = (y_proba>=threshold).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    best_f1 = f1
    best_threshold = threshold
    best_precision = precision
    best_recall = recall

    

    logger.info(f"Model: {name}")
    logger.info(f"Best threshold {best_threshold:.2f}")
    logger.info(f"Precision {best_precision:.3f}")
    logger.info(f"Recall {best_recall:.3f}")
    logger.info(f"F1: {best_f1:.3f}")
    logger.info(f"ROC-AUC: {roc_auc:.3f}")
    logger.info(f"PR-AUC: {pr_auc:.3f}")