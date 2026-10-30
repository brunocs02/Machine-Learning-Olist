from module_olist.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
)
import dataset as ds
from features import create_features
from modeling.split import split_data
from modeling.train import train_model
from modeling.evaluate import evaluate_model
from modeling.cross_validation_ans import cross_validate_models
from loguru import logger

def main():

    #Dataset prep

    [orders, items, customers] = ds.load_data(orders_path=(RAW_DATA_DIR/"olist_orders_dataset.csv"), 
                                                items_path=(RAW_DATA_DIR/"olist_order_items_dataset.csv"), 
                                                customers_path=(RAW_DATA_DIR/"olist_customers_dataset.csv"))

    dataset = ds.create_dataset(orders, items, customers)

    featured_dataset = create_features(dataset)

    ds.save_dataset(featured_dataset, INTERIM_DATA_DIR/"dataset.csv")

    #Spliting train/test sets

    X_train, X_test, y_train, y_test = split_data(featured_dataset)

    #Cross validating models
    #Chooses best performing model

    name, threshold = cross_validate_models(X_train, y_train)

    logger.info(f"Chosen model: {name}")
    logger.info(f"Chosen threshold: {threshold}")

    trained_model = train_model(name, threshold, X_train, y_train, MODELS_DIR/"trained_model.pkl", MODELS_DIR/"metadata.json")

    #Final evaluation

    evaluate_model(trained_model, name, threshold, X_test, y_test)
    logger.success(f"Pipeline finished")
    
if __name__ == "__main__":
    main()