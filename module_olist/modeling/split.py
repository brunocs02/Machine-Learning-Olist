from sklearn.model_selection import train_test_split
import pandas as pd

FEATURES = ["purchase_hour", "purchase_month",  "purchase_weekday", "item_count", "promised_days", "seller_count", "total_price", "total_freight", "customer_state"]

TARGET = "is_late"

def split_data(data: pd.DataFrame):
    """
    Splits dataset in training and testing sets

    Args:
        data: dataset to be split
    """
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test