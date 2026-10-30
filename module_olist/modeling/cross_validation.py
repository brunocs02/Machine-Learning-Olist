from sklearn.model_selection import KFold, cross_val_score
from loguru import logger

def cross_validate(models, X, y, splits):

    folds = KFold(n_splits=splits)

    for name, model in models.items():

        score = cross_val_score(model, X, y, cv=folds)

        logger.info(f"Model: {name}")
        logger.info(f"CV scores: {score}")
        logger.info(f"Mean score {score.mean()}")