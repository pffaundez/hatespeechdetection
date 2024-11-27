from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Cargar el modelo y el vectorizador
model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Asegúrate de crear un archivo 'index.html'

@app.route('/predict', methods=['POST'])
def predict():
    tweet = [request.form['tweet']]
    vect_tweet = vectorizer.transform(tweet)
    prediction = model.predict_proba(vect_tweet)[0]
    labels = ['Hate Speech', 'Offensive Language', 'None of the Above']
    results = {label: round(prob * 100, 2) for label, prob in zip(labels, prediction)}
    return render_template('result.html', prediction=results)

if __name__ == '__main__':
    app.run(debug=True)
