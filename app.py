from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
app = Flask(__name__)
@app.route("/", methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        df = pd.read_csv('scraping-dataset/news.csv', encoding="UTF-8")
        # Обробка відсутніх значень у колонці 'Text'
        df['Text'] = df['Text'].fillna('')
        X = df['Text']
        y = df['Label']
        cv = CountVectorizer()
        X = cv.fit_transform(X)  # Побудова векторів ознак
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        # Класифікатор "Наївний баєс"
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print("Accuracy:", accuracy)

        keyword = request.form['keyword']
        data = [keyword]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        return render_template('index.html', prediction=my_prediction)
    else:
        return "Unsupported Request Method"
@app.route("/index", methods=['GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
@app.route("/tips", methods=['GET'])
def tips():
    if request.method == 'GET':
        return render_template('tips.html')
@app.route("/information", methods=['GET'])
def information():
    if request.method == 'GET':
        return render_template('information.html')
if __name__ == '__main__':
    app.run(port=5000, debug=True)
