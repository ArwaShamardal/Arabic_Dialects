import pickle
import os

from sklearn.metrics import accuracy_score, classification_report, f1_score


class MLModel:
    def __init__(self, model: object, vectorizer: object):
        self.model = model if model is not None else None
        self.vectorizer = vectorizer if vectorizer is not None else None

    def train(self, X_train, y_train):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vec, y_train)

    def predict(self, X_test):
        X_test_vec = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_vec)

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
        print(classification_report(y_test, y_pred))

    def save_model(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if os.path.exists(path):
            os.remove(path)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, path):
        if not os.path.exists(path):
            print(f"File {path} does not exist")
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)
