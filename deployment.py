from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "ayln/distilbert_finetuned_hatespeech"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

import torch
from torch.nn.functional import softmax

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = softmax(logits, dim=1)

    predicted_class_id = logits.argmax().item()
    predicted_probability = probabilities[0, predicted_class_id].item()

    labels = {
        "LABEL_0": "Hate Speech",
        "LABEL_1": "Offensive Language",
        "LABEL_2": "That is ok"
    }

    formatted_result = f'{labels[model.config.id2label[predicted_class_id]]} (Probability = {predicted_probability * 100:.2f}%)'
    return formatted_result



from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        classification = classify_text(text)
        return render_template('result.html', classification=classification)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
