import pandas as pd
from sklearn.model_selection import train_test_split


from scripts.data_fetching import DBHandler
from scripts.data_preprocessing import Preprocess


# NOTE:: the commented out part should only run once
# database_path = './data/dialects_database.db'
# db_handler = DBHandler(database_path)

# db_table_names = db_handler.list_tables()

# if db_table_names is not None:
#     df_list = db_handler.read_tables_to_dataframes(db_table_names)
# df = pd.merge(df_list[0], df_list[1], on='id')
# db_handler.close_connection()

# print(df.shape)
# df.head()
# df.to_csv('./data/dialects_data.csv', index=False)


df = pd.read_csv('./data/dialects_data.csv')
df_copy = df.copy()
df_preprocessed = Preprocess(df_copy)
df_preprocessed = df_preprocessed.preprocess()
df_preprocessed.head()

X_train, X_test, y_train, y_test = train_test_split(
    df_preprocessed['text'], df_preprocessed['dialect'], test_size=0.2, random_state=42)
