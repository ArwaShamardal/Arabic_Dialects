import keras
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout, SimpleRNN
from keras.models import Sequential
import os
import pickle
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print(keras.__version__)


class NLPModel:
    def __init__(self, model=None, tokenizer=None, label_encoder=None, max_sequence_length=27,
                 embedding_dim=100, vocab_size=20000):

        self.model = model if model is not None else None
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer()
        self.label_encoder = label_encoder if label_encoder is not None else LabelEncoder()
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim

    def build_model(self, X_train, num_classes=5, learning_rate=0.001):
        self.tokenizer.fit_on_texts(X_train)
        vocab_size = len(self.tokenizer.word_index) + 1
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=self.embedding_dim,
                            input_length=self.max_sequence_length))
        model.add(SimpleRNN(100, return_sequences=False))
        # model.add(SimpleRNN(100))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate),
                      metrics=['accuracy'])  # , loss_weights=
        self.model = model

    def train(self, X_train, y_train, class_weights=None, epochs=20, batch_size=64, validation_data=None):
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_train_pad = pad_sequences(
            X_train_seq, maxlen=self.max_sequence_length)
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_train_cat = to_categorical(y_train_enc)

        self.model.fit(X_train_pad, y_train_cat, epochs=epochs, batch_size=batch_size,
                       validation_data=validation_data, class_weight=class_weights)

    def tokenize(self, X):
        X_seq = self.tokenizer.texts_to_sequences(X)
        return pad_sequences(X_seq, maxlen=self.max_sequence_length)

    def predict(self, X_test):
        X_test_pad = self.tokenize(X_test)
        y_pred_prob = self.model.predict(X_test_pad)
        y_pred_int = np.argmax(y_pred_prob, axis=1)
        return self.label_encoder.inverse_transform(y_pred_int)

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        percision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='micro')
        f1 = f1_score(y_test, y_pred, average='micro')

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Percision: {percision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(
            f"F1 Score (micro): {f1_score(y_test, y_pred, average='micro'):.2f}")
        print(classification_report(y_test, y_pred))

    def save_model(self, path):
        self.model.save(path)

    def save_tokenizer(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

    def save_label_encoder(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

    @classmethod
    def load_model(cls, model_path, tokenizer_path, label_encoder_path, max_sequence_length=27, vocab_size=20000, embedding_dim=100):
        from keras.models import load_model
        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path) or not os.path.exists(label_encoder_path):
            print(
                f"File {model_path}, {tokenizer_path}, or {label_encoder_path} does not exist")
            return None
        model = load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)

        return cls(model=model, tokenizer=tokenizer, label_encoder=label_encoder, max_sequence_length=max_sequence_length, vocab_size=vocab_size, embedding_dim=embedding_dim)


if __name__ == 'main':
    nlp_model = NLPModel()
    nlp_model.build_model()
    print("Done....")
