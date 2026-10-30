import dataset as ds
import features as ft

def main():

    [orders, items, customers] = ds.load_data("data\\raw\\olist_orders_dataset.csv", "data\\raw\\olist_order_items_dataset.csv", "data\\raw\\olist_customers_dataset.csv")

    dataset = ds.create_dataset(orders, items, customers)

    featured_dataset = ft.create_features(dataset)

    ds.save_dataset(featured_dataset, "data\\interim\\dataset.csv")
    
if __name__ == "__main__":
    main()