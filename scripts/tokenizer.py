from keras_preprocessing.text import Tokenizer
import pandas as pd
import pickle



def save_tokenizer(tokenizer, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

if __name__ == '__main__':
    data = pd.read_csv('../data/dialects_data.csv')
    X = data['text']
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(X)
    save_tokenizer(tokenizer, '../models/tokenizer.pkl')
    print("Done....")