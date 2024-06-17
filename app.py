from flask import Flask, request, jsonify, render_template

from scripts.data_fetching import DBHandler
from scripts.data_preprocessing import Preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from scripts.ml_model import MLModel
from scripts.dl_model import NLPModel
from keras_preprocessing.sequence import pad_sequences

import os

import joblib
# TODO:: Add models here and add them to the models dictionary

model_logistic = MLModel.load_model('./models/lg_model.pkl')
model_obj = NLPModel.load_model(
    './models/dl_model.keras', './models/tokenizer.pkl', './models/label_encoder.pkl')
model_dl = model_obj.model
tokenizer = model_obj.tokenizer


models_dict = {
    "Logistic Regression": model_logistic,
    "Deep Learning": model_obj
}

dialects = {
    "EG": "Egyptian Arabic",
    "LY": "Libyan Arabic",
    "LB": "Lebanese Arabic",
    "SD": "Sudanese Arabic",
    "MA": "Moroccan Arabic"
}


app = Flask(__name__)


@app.route("/", methods=['GET'])
def home():
    f_model_names = list(models_dict.keys())
    return render_template("home.html", model_names=f_model_names, result="")


@app.route("/predict", methods=['POST'])
def predict():
    f_text = request.form['text']
    if f_text == "":
        return jsonify({'result': "", 'text': "", 'model': ""})

    f_text = Preprocess(f_text).preprocess()
    f_text = f_text['text'][0]
    if f_text == "":
        return jsonify({'result': "", 'text': "", 'model': ""})

    f_model_name = request.form['model']
    f_model = models_dict[f_model_name]
    prediction = f_model.predict([f_text])

    if f_model_name == "Logistic Regression":
        prediction_prob = f_model.model.predict_proba(
            f_model.vectorizer.transform([f_text]))
        # class_indices = f_model.get_class_indices()
    else:
        X_test_pad = f_model.tokenize([f_text])
        prediction_prob = f_model.model.predict(X_test_pad)
        # class_indices = f_model.label_encoder.classes_

    prediction_prob = prediction_prob[0].tolist()
    for i in range(len(prediction_prob)):
        prediction_prob[i] = prediction_prob[i]*100
    print(prediction_prob)
    # print(class_indices)

    f_result = dialects[prediction[0]]

    return jsonify({'result': f_result, 'text': f_text, 'model': f_model_name, 'prediction_percentages': prediction_prob})


if __name__ == '__main__':
    app.run(debug=False)
