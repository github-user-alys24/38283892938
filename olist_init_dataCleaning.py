"""
init_dataCleaning.py
Alyssa Ang alysappleseed@gmail.com

This script contains the Initial Data Cleaning Step

Initial data cleaning has been carried out to ensure data quality and a workable dataset before moving onto the EDA

By performing data cleaning and merging of tables as a separate step ensures 
1. Modularity
2. Focus - focus of the process is solely on achieving data quality and not analysis
3. Clearer pipeline and increased efficiency

In this process, there will be sections for
1. Understanding data quality - inspecting the raw data tables
2. Handling missing values, and inconsistencies
3. Data transformation
4. Merging tables to form one cohesive, and unified dataset

Throughout the code, there will comments to explain respective choices and functions.
"""

# Import libraries
import numpy as np
import pandas as pd
import math

from config import (
    CUSTOMERS_DATASET_PATH, GEOLOCATION_DATASET_PATH, ORDER_ITEMS_DATASET_PATH, 
    ORDER_PAYMENTS_DATASET_PATH, ORDER_REVIEWS_DATASET_PATH, ORDER_ORDERS_DATASET_PATH, 
    ORDER_PRODUCTS_DATASET_PATH, ORDER_SELLERS_DATASET_PATH, 
    PRODUCT_CATEGORY_NAME_TRANSLATION_DATASET_PATH
)

class dataCleaning: # Performs dataCleaning
    def __init__(self):
        self.customer_data = pd.read_csv(CUSTOMERS_DATASET_PATH)
        self.geolocation_data = pd.read_csv(GEOLOCATION_DATASET_PATH)
        self.order_items_data = pd.read_csv(ORDER_ITEMS_DATASET_PATH)
        self.order_payments_data = pd.read_csv(ORDER_PAYMENTS_DATASET_PATH)
        self.order_reviews_data = pd.read_csv(ORDER_REVIEWS_DATASET_PATH)
        self.orders_data = pd.read_csv(ORDER_ORDERS_DATASET_PATH)
        self.products_data = pd.read_csv(ORDER_PRODUCTS_DATASET_PATH)
        self.sellers_data = pd.read_csv(ORDER_SELLERS_DATASET_PATH)
        self.prod_cat_name_translation = pd.read_csv(PRODUCT_CATEGORY_NAME_TRANSLATION_DATASET_PATH)
        self.cleaned_data = None
        self.geo_data = None
        
    """
    Declare functions to be used in the data cleaning steps
    Comments have been made for better understanding of each function
    """
    # Check for duplicated values in specified dataset, by column
    def checkDuplicated(self, df, col):
        for name, dataframe in df.items():
            if col in dataframe.columns:
                duplicated_col = dataframe[col].value_counts()
                duplicated_col = duplicated_col[duplicated_col > 1]
                if not duplicated_col.empty:
                    print(f"There are duplicated values in '{col}' of {name} dataset")
                else:
                    return
                
    # Get the number of unique values in each dataset
    def getNunique(self, df):
        for col in df.columns:
            print(f"{col}{df[col].nunique()}")

    # Convert values in columns into pd.datetime
    def convertDatetime(self, df, col_list):
        for col in col_list:
            try:
                df = df.copy()
                df[col] = pd.to_datetime(df[col])
            except (KeyError, ValueError) as error:
                print(f"Error converting column '{col}' to datetime: {error}")
        return df

    # Categorize types by grouping by specified cols and aggregate information
    def unique_types(self, x):
        return sorted(pd.Series(x).unique())

    # Fill missing values for target column with specified fillwith value
    def fill_missing_values(self, df, df2, target, fillwith):
        fill_missing = df.loc[df2, target].fillna(fillwith)
        df.loc[df2, target] = fill_missing

    # Function to remove zip_code and map states to relevant datasets
    def addGeolocation(self, df, name):
        df.drop(columns=[name+'_zip_code_prefix', name+'_city'], inplace=True)
        df.rename(columns={name+'_state': 'geolocation_state'}, inplace=True)
        df = pd.merge(df, self.geolocation_data, on='geolocation_state', how='left')
        df.rename(columns={'geolocation_state':name+'_state', 
                           'geolocation_lat': name+'_lat',
                           'geolocation_lng':name+'_lng'}, inplace=True)
        return df
    
    # Haversine to calculate distance
    def haversine(self, lon1, lat1, lon2, lat2):
        # Radius of the Earth in kilometers
        R = 6371.0

        # Convert coordinates from degrees to radians
        lon1 = math.radians(lon1)
        lat1 = math.radians(lat1)
        lon2 = math.radians(lon2)
        lat2 = math.radians(lat2)

        # Difference in coordinates
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        # Haversine formula
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Distance in kilometers
        distance = R * c

        return distance

    """
    Declared specialized functions here to clean each individual dataset
    """
    # Clean geolocation data first as it has the least columns
    def clean_geolocation_data(self):
        print("cleaning geolocation data..")
        # Filter values that lie outside of Brazil
        self.geolocation_data = self.geolocation_data[(self.geolocation_data["geolocation_lat"].between(-33.75, 5.27)) & 
                                            (self.geolocation_data["geolocation_lng"].between(-73.98, -34.79))]

        # Get the avg lat and lng
        self.geolocation_data = self.geolocation_data.groupby('geolocation_state').agg({
            'geolocation_lat': 'mean',                  
            'geolocation_lng': 'mean'
        }).reset_index()
        print(">> completed! :)")
        return self.geolocation_data
    
    # Customer data contains important cols such as
    # customer_unique_id, customer_id, customer_state, lat and lng
    def clean_customer_data(self):
        print("cleaning customer data..")
        self.customer_data = self.addGeolocation(self.customer_data,"customer")
        print(">> completed! :)")
        return self.customer_data
    
    # Sellers data is similar to customer data, which contains important cols such as
    # seller_id, seller_state, lat and lng
    def clean_sellers_data(self):
        print("cleaning sellers data..")
        self.sellers_data = self.addGeolocation(self.sellers_data,"seller")
        print(">> completed! :)")
        return self.sellers_data
    
    # There are missing values
    def clean_prod_cat_name_translation(self):
        global cat_dict
        print("cleaning product category name translation data..")
        # Check if categories are present between both datasets
        self.products_data['product_category_name'].loc[
            ~self.products_data['product_category_name'].isin(self.prod_cat_name_translation['product_category_name'])].unique()

        # Map missing translations for categories
        missing_translations = {
                'pc_gamer': 'pc_gamer',
                'portateis_cozinha_e_preparadores_de_alimentos': 'portable_kitchen_food_preparers'
            }

        # Create a dataframe using Pandas for missing values
        missing_df = pd.DataFrame({
            'product_category_name': list(missing_translations.keys()),
            'product_category_name_english': list(missing_translations.values())
        })

        # Concat to the original dataset
        self.prod_cat_name_translation = pd.concat([self.prod_cat_name_translation, 
                                               missing_df], ignore_index=True)

        # Create a dictionary for mapping
        cat_dict = dict(zip(self.prod_cat_name_translation['product_category_name'], 
                            self.prod_cat_name_translation['product_category_name_english']))
        print(">> completed! :)")
        return self.prod_cat_name_translation
    
    def clean_products_data(self):
        global products_data_2
        print("cleaning products data..")
        # Fill missing categories with 'unknown' 
        prodcat = 'product_category_name'
        self.products_data[prodcat] = self.products_data[prodcat].map(cat_dict).fillna('unknown')

        # List of columns to fill null values with median
        columns_to_fill = ['product_name_lenght', 'product_description_lenght',
                           'product_photos_qty', 'product_weight_g', 
                           'product_length_cm', 'product_height_cm', 
                           'product_width_cm']

        # Fill missing values with the median
        for column in columns_to_fill:
            median_value = self.products_data[column].median()
            self.products_data[column].fillna(median_value, inplace=True)
        
        # Create a new column 'product_size'
        self.products_data['product_size'] = (
            self.products_data['product_length_cm'] *
            self.products_data['product_height_cm'] *
            self.products_data['product_width_cm']
        )
        
        # Create a new dataset for products_data with relevant columns to reduce dimensionality
        products_data_2 = self.products_data[['product_id', 'product_photos_qty', 'product_name_lenght',
                                              'product_description_lenght','product_size']]
        
        print(">> completed! :)")
        return self.products_data
    
    def clean_order_items_data(self):
        print("cleaning order items data..")

        # Create a dictionary for mapping
        prodID_dict = dict(zip(self.products_data['product_id'], 
                               self.products_data['product_category_name']))
        
        # Drop this order_id that is hard to impute for
        self.order_items_data = self.order_items_data[self.order_items_data.order_id!="bfbd0f9bdef84302105ad712db648a6c"]
        
        # Map product_id to category names
        self.order_items_data['product_category_name'] = self.order_items_data['product_id'].map(prodID_dict)

        broader_mapping = {
            'home and furnishing': [
                'bed_bath_table', 'furniture_decor', 'housewares', 'home_appliances',
                'furniture_living_room', 'furniture_bedroom', 'furniture_mattress_and_upholstery',
                'home_confort', 'home_construction', 'kitchen_dining_laundry_garden_furniture',
                'home_appliances_2', 'office_furniture', 'portable_kitchen_food_preparers', 'air_conditioning'
            ],
            'electronics and accessories': [
                'computers_accessories', 'telephony', 'electronics', 'small_appliances',
                'fixed_telephony', 'tablets_printing_image', 'pc_gamer'
            ],
            'fashion and accessories': [
                'watches_gifts', 'fashion_bags_accessories', 'fashion_shoes', 'fashion_male_clothing',
                'fashion_underwear_beach', 'fashion_sport', 'fashio_female_clothing', 'fashion_childrens_clothes',
                'luggage_accessories'
            ],
            'health and beauty': [
                'health_beauty', 'perfumery', 'diapers_and_hygiene'
            ],
            'toys and entertainment': [
                'toys', 'cool_stuff', 'consoles_games', 'musical_instruments', 'party_supplies'
            ],
            'sports, pets and outdoors': [
                'sports_leisure', 'garden_tools', 'agro_industry_and_commerce', 'pet_shop'
            ],
            'books, media and stationery': [
                'books_general_interest', 'books_technical', 'books_imported', 'cds_dvds_musicals',
                'dvds_blu_ray', 'music', 'stationery'
            ],
            'auto and tools': [
                'auto', 'construction_tools_construction', 'construction_tools_safety',
                'costruction_tools_garden', 'costruction_tools_tools', 'construction_tools_lights'
            ],
            'fnb': [
                'food_drink', 'food', 'drinks','la_cuisine'
            ],
            'miscellaneous': [
                'unknown', 'market_place', 'signaling_and_security', 'industry_commerce_and_business',
                'christmas_supplies', 'audio', 'art', 'cine_photo', 'arts_and_craftmanship', 'flowers',
                'security_and_services'
            ]
        }

        # Define a function to map product_category_name to broader categories
        def map_to_broad_category(product_category_name):
            for category, products in broader_mapping.items():
                if product_category_name in products:
                    return category
            return 'miscellaneous'  # Default to 'miscellaneous' if product_category_name is not found in any category

        # Apply the mapping function to create a new column 'broad_category'
        self.order_items_data['product_category_name'] = self.order_items_data['product_category_name'].apply(map_to_broad_category)
        
        print(">> completed! :)")
        return self.order_items_data
    
    def clean_order_payments_data(self):
        global order_payments_agg
        print("cleaning order payments data..")
        
        # Change payment_installments and payment_sequential accordingly
        self.order_payments_data.loc[self.order_payments_data['payment_installments'] == 0, ['payment_sequential', "payment_installments"]] = 1

        # Group by 'order_id' to aggregate payment information
        order_payments_agg = self.order_payments_data.groupby('order_id').agg({
            'payment_type': lambda x: self.unique_types(x),         # Aggregated payment_types
            'payment_installments': 'max',        # Total number of installments
            'payment_value': 'sum',               # Total cost of order   
            'payment_sequential': 'max'           # Number of sequences
        }).reset_index()

        # Rename columns for clarity
        order_payments_agg.columns = ['order_id', 'payment_types', 'total_installments', 'total_payment_value', 'total_sequences']       

        print(">> completed! :)")
        return self.order_payments_data
        
    def clean_orders_data(self):
        global orders_data_copy
        print("cleaning orders data..")
        # Remove order_id that is hard to impute missing data for
        self.orders_data = self.orders_data[self.orders_data['order_id'] != "bfbd0f9bdef84302105ad712db648a6c"]

        # Filter the DataFrame to include only delivered orders
        delivered_orders = self.orders_data[self.orders_data["order_status"] == "delivered"]

        # List of columns to convert, excluding 'order_id', 'customer_id', and 'order_status'
        col_list = list(self.orders_data.drop(columns=['order_id', 'customer_id', 'order_status'], axis=1).columns)
        self.orders_data = self.convertDatetime(self.orders_data, col_list)

        # Calculate the difference between the dates
        difference = self.orders_data['order_delivered_customer_date'] - self.orders_data['order_estimated_delivery_date']

        # Calculate the mean of the differences
        mean_difference = difference.mean().days

        # Fill in values for columns with missing values
        dlr = 'order_delivered_customer_date'
        est = 'order_estimated_delivery_date'
        app = 'order_approved_at'
        carrier = 'order_delivered_carrier_date'

        # Only concerning orders under delivered status, as it should have all dates present
        status_dlr = self.orders_data['order_status'] == 'delivered'

        # Fill missing values 
        self.fill_missing_values(self.orders_data,status_dlr,dlr,
                                 self.orders_data[est]+pd.Timedelta(days=mean_difference))
        self.fill_missing_values(self.orders_data,status_dlr,app,
                                 self.orders_data["order_purchase_timestamp"])
        self.fill_missing_values(self.orders_data,status_dlr,carrier,
                                 self.order_items_data["shipping_limit_date"])

        # Create a copy to avoid FutureWarning
        orders_data_copy = self.orders_data[status_dlr].copy()   
        
        # Month the customer purchased in
        orders_data_copy['customer_purchase_month'] = orders_data_copy['order_purchase_timestamp'].dt.month
        # Year the customer purchased in
        orders_data_copy['customer_purchase_year'] = orders_data_copy['order_purchase_timestamp'].dt.year
        # Total days starting from day of purchase to day of delivery
        orders_data_copy['order_lifecycle'] = (orders_data_copy['order_delivered_customer_date'] - orders_data_copy['order_purchase_timestamp']).dt.days
        # Estimated lifecycle provided by olist
        orders_data_copy['order_estimated_lifecycle'] = (orders_data_copy['order_estimated_delivery_date'] - orders_data_copy['order_purchase_timestamp']).dt.days
        # Days taken for purchase to be approved from the day of purchase
        orders_data_copy['order_approved_days'] = (orders_data_copy['order_approved_at'] - orders_data_copy['order_purchase_timestamp']).dt.days
        # Days taken for purchase to be handed to logistics partner
        orders_data_copy['order_to_logs_days'] = (orders_data_copy['order_delivered_carrier_date'] - orders_data_copy['order_purchase_timestamp']).dt.days
        
        # Drop cols that were already used to calculate
        orders_data_copy.drop(columns=['order_purchase_timestamp','order_approved_at',
                          'order_delivered_carrier_date', 'order_delivered_customer_date',
                          'order_estimated_delivery_date'], inplace=True)
        
        print(">> completed! :)")
        return self.orders_data
    
    def clean_order_reviews_data(self):
        global order_reviews_data_agg
        
        print("cleaning order reviews data..")
        # List of columns to convert, excluding 'order_id', 'customer_id', and 'order_status'
        col_list = list(self.order_reviews_data.drop(columns=['review_id', 'order_id', 'review_score',
                                                              'review_comment_title','review_comment_message'], axis=1).columns)
        self.order_reviews_data = self.convertDatetime(self.order_reviews_data, col_list)

        # Number of days taken for the seller to answer the review
        self.order_reviews_data['answer_days_taken'] = (self.order_reviews_data['review_answer_timestamp'] 
                                                        - self.order_reviews_data['review_creation_date']).dt.days
        
        # Categorize where 0 denotes no review title given, and 1 for title given
        self.order_reviews_data['review_comment_title'] = self.order_reviews_data['review_comment_title'].apply(lambda x: 1 if pd.notnull(x) else 0)

        # Categorize accordingly where 0 for no message, 1 for short, 2 for medium, 3 for long
        def categorize_message_length(message):
            if pd.isnull(message):
                return 0
            elif len(message) <= 50:
                return 1
            elif len(message) <= 100:
                return 2
            else:
                return 3
        
        # Apply categorization
        self.order_reviews_data['review_comment_message'] = self.order_reviews_data['review_comment_message'].apply(categorize_message_length)
        
        # Drop columns that were used, reduce redundancy
        self.order_reviews_data.drop(columns=['review_creation_date', 'review_answer_timestamp'], inplace=True)
        
        # Aggregrate data accordingly, use median to avoid skewness, review_id counts is total reviews given
        order_reviews_data_agg = self.order_reviews_data.groupby('order_id').agg({
            'review_id': 'count',
            'review_score':'median',
            'review_comment_title':'median',
            'review_comment_message':'median',
            'answer_days_taken':'median',
        }).reset_index()
        order_reviews_data_agg.rename(columns={'review_id': 'total_reviews'}, inplace=True)
        
        # Change to float for consistency with other dtypes
        order_reviews_data_agg['review_comment_title'] = order_reviews_data_agg['review_comment_title'].astype('int64')
        order_reviews_data_agg['review_comment_message'] = order_reviews_data_agg['review_comment_message'].astype('int64')
        order_reviews_data_agg['answer_days_taken'] = order_reviews_data_agg['answer_days_taken'].astype('int64')
        order_reviews_data_agg['review_score'] = order_reviews_data_agg['review_score'].astype('int64')
        print(">> completed! :)")
        return self.order_reviews_data
        
    def merging(self):
        print("Merging datasets..")
        merged_customer_data = pd.merge(orders_data_copy, self.customer_data, on='customer_id', how='left')
        
        order_and_customer_state = merged_customer_data[['order_id', 'customer_lat', 'customer_lng']]
        merged_order_items = pd.merge(self.order_items_data, self.sellers_data, on='seller_id', how='left')
        merged_order_items = pd.merge(merged_order_items, order_and_customer_state, on='order_id', how='left')
        merged_order_items = pd.merge(merged_order_items, products_data_2, on='product_id', how='left')
        merged_order_items['distance_km'] = merged_order_items.apply(lambda row: 
                                                             round(self.haversine(row['seller_lng'], row['seller_lat'], 
                                                                                  row['customer_lng'], row['customer_lat']), 0), 
                                                             axis=1)
        
        order_items_agg = merged_order_items.groupby('order_id').agg({
            'order_item_id': 'max',                    # Total number of items
            'seller_id': 'count',                      # Count number of sellers bought from
            'product_category_name':  lambda x: self.unique_types(x),   # Group product categories
            'seller_state': lambda x: self.unique_types(x),   # Group sellers states
            'distance_km': 'median',                   # Median distance between seller and customer
            'price': 'sum',                            # Total cost of order   
            'freight_value': 'sum',                    # Total freight cost
            'product_photos_qty': 'median',            # Median number of photos
            'product_name_lenght': 'median',           # Median length of name
            'product_description_lenght': 'median',    # Median length of description
            'product_size':'median',                   # Median size of all products bought
            'product_size_binned': lambda x: self.unique_types(x) # Group product size
        }).reset_index()
        
        merged_order_payments = pd.merge(order_items_agg, order_payments_agg, on='order_id', how='left')
        merged_order_payments.rename(columns={'order_item_id': 'total_items',
                                              'seller_id': 'total_sellers_in_order',
                                              'price': 'total_price',
                                              'freight_value': 'total_freight_cost'
                                             }, inplace=True)
        merged_order_reviews = pd.merge(merged_order_payments, order_reviews_data_agg, on='order_id', how='left')
        # Fill missing values with 0 if missing
        merged_order_reviews['review_comment_title'].fillna(0, inplace=True)
        merged_order_reviews['review_comment_title'].fillna(0, inplace=True)
        merged_order_reviews['total_reviews'].fillna(0, inplace=True)
        
        merged_order_data = pd.merge(orders_data_copy, merged_order_reviews, on='order_id', how='left')
        
        customer_data_2 = self.customer_data[['customer_id','customer_unique_id', 'customer_state']]
        main = pd.merge(merged_order_data, customer_data_2, on='customer_id', how='left')
        
        # Calculate the total value counts for customer_unique_id
        customer_counts = main['customer_unique_id'].value_counts()

        # Map purchase counts to each customer_unique_id
        main['customer_purchase_count'] = main['customer_unique_id'].map(customer_counts)
        
        # Create a separate geodata
        orders_customer_id = orders_data_copy[['order_id','customer_id']]
        orders_customer_id = pd.merge(orders_customer_id, self.customer_data, on='customer_id', how='left')
        
        order_product_id = self.order_items_data[['order_id','seller_id']]
        orders_seller_id = pd.merge(orders_customer_id, order_product_id, on='order_id', how='left')
        customer_seller_geodata = pd.merge(orders_seller_id, self.sellers_data, on='seller_id', how='left')
        customer_seller_geodata.drop_duplicates(keep='first', inplace=True)
        
        self.cleaned_data = main
        self.geo_data = customer_seller_geodata
        
        print(">> completed! :)")
        return self.cleaned_data, self.geo_data 
    
    """
    Main cleaning function
    """
    def clean_data(self):
        print("data cleaning step")
        # Individual dataset cleaning functions here
        self.clean_geolocation_data()
        self.clean_customer_data()
        self.clean_sellers_data()
        self.clean_prod_cat_name_translation()
        self.clean_products_data()
        self.clean_order_items_data()
        self.clean_order_payments_data()
        self.clean_orders_data()
        self.clean_order_reviews_data()
        self.merging()
        # Convert to csv file for EDA portion
        self.cleaned_data.to_csv("main",index=False)
        self.geo_data.to_csv("geodata",index=False)
        print()
        print("Cleaning pipeline completed!")
        
if __name__ == "__main__":
    # Initialize the dataCleaning class
    data_cleaner = dataCleaning()
    
    # Perform cleaning
    data_cleaner.clean_data()