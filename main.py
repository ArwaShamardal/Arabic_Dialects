import pandas as pd
from sklearn.model_selection import train_test_split


from scripts.data_fetching import DBHandler
from scripts.data_preprocessing import Preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression

from scripts.ml_model import MLModel

from scripts.dl_model import NLPModel


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

############################################

count_vect = CountVectorizer()
model = LogisticRegression()

my_model = MLModel(model, count_vect)
my_model.train(X_train, y_train)
predictions = my_model.predict(X_test)
my_model.evaluate(y_test, predictions)

my_model.save_model('./models/lg_model.pkl')


model = MLModel.load_model('./models/lg_model.pkl')
predictions = model.predict(X_test)
model.evaluate(y_test, predictions)

############################################

nlp_model = NLPModel()  # embedding_dim=32, max_sequence_length=40

nlp_model.build_model(num_classes=5, X_train=X_train)

class_weights = {0: 0.5126136442501215, 1: 1.0698120722743238,
                 2: 0.8094742321707444, 3: 2.5604471791316405, 4: 2.046903145351254}

nlp_model.train(X_train, y_train, class_weights=class_weights,
                epochs=10, batch_size=1024)

y_pred = nlp_model.predict(X_test)

nlp_model.evaluate(y_test, y_pred)

nlp_model.save_model('./models/dl_model.keras')
nlp_model.save_tokenizer('./models/tokenizer.pkl')
nlp_model.save_label_encoder('./models/label_encoder.pkl')


# loaded_model = NLPModel.load_model('path_to_save_model.h5', 'path_to_save_tokenizer.pkl')
# if loaded_model:
#   y_pred_loaded = loaded_model.predict(X_test)
#  loaded_model.evaluate(y_test, y_pred_loaded)
