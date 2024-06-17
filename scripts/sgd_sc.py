import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np
from data_preprocessing import Preprocess
from gensim.models import Word2Vec
from gensim.models import KeyedVectors



df=pd.read_csv('../../data/dialects_data.csv')
#preprocess
preprocess = Preprocess(df)
cleaned_df=preprocess.preprocess()


def load_embedding_model(path):
    return KeyedVectors.load_word2vec_format(path, binary=False)


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = 300

    def fit(self, X, y):
        return self

    def transform(self, X):
        return pd.DataFrame([np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                                     or [np.zeros(self.dim)], axis=0) for words in X])
        
        
class tfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = 300

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return pd.DataFrame([np.mean([self.word2vec[w] * self.word2weight[w]
                                      for w in words if w in self.word2vec] or
                                     [np.zeros(self.dim)], axis=0) for words in X])