import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "bert_multilabel"

tokenizer = AutoTokenizer.from_pretrained(model_path)
bert_model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

from flask import Flask, request, jsonify

app = Flask(__name__)

MAX_LEN = 512
BERT_THRESHOLD = 0.5

KNOWN_LABELS = list(bert_model.config.id2label.values())

# load model + tokenizer here (same as your notebook)
def predict_bert_multilabel(title, abstract, threshold=None):
    if threshold is None:
        threshold = BERT_THRESHOLD

    text = title.strip() + ' ' + abstract.strip()

    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=MAX_LEN,
        truncation=True,
        padding=True   # better than 'max_length'
    )

    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device)
        )

    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    predicted = [
        {"label": KNOWN_LABELS[i], "score": round(float(p), 4)}
        for i, p in enumerate(probs) if p >= threshold
    ]

    all_scores = {
        KNOWN_LABELS[i]: round(float(p), 4)
        for i, p in enumerate(probs)
    }

    return predicted if predicted else [{"label": "(no label predicted)", "score": 0.0}], all_scores


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    title = data.get("title", "")
    abstract = data.get("abstract", "")
    threshold = data.get("threshold", None)

    predicted, all_scores = predict_bert_multilabel(
        title, abstract, threshold
    )

    return jsonify({
        "predicted": predicted,
        "all_scores": all_scores
    })


@app.route("/", methods=["GET"])
def home():
    return """
    <!doctype html>
    <html>
    <head>
        <title>BERT Paper Classifier</title>
        <style>
            textarea { width: 100%; }
            #results { margin-top: 20px; }
        </style>
    </head>
    <body>
        <h2>Paper Category Predictor</h2>

        <label>Title:</label><br>
        <input type="text" id="title" size="100"><br><br>

        <label>Abstract:</label><br>
        <textarea id="abstract" rows="10"></textarea><br><br>

        <button onclick="predict()">Predict</button>

        <div id="results"></div>

        <script>
            function predict() {
                const title = document.getElementById("title").value;
                const abstract = document.getElementById("abstract").value;

                fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({title: title, abstract: abstract})
                })
                .then(response => response.json())
                .then(data => {
                    let html = "<h3>Predicted Labels:</h3><ul>";
                    data.predicted.forEach(item => {
                        html += `<li>${item.label}: ${item.score}</li>`;
                    });
                    html += "</ul>";

                    html += "<h3>All Scores:</h3><ul>";
                    for (let label in data.all_scores) {
                        html += `<li>${label}: ${data.all_scores[label]}</li>`;
                    }
                    html += "</ul>";

                    document.getElementById("results").innerHTML = html;
                })
                .catch(error => {
                    document.getElementById("results").innerHTML = "<p style='color:red;'>Error: " + error + "</p>";
                });
            }
        </script>
    </body>
    </html>
    """


@app.route("/predict_form", methods=["POST"])
def predict_form():
    title = request.form.get("title", "")
    abstract = request.form.get("abstract", "")

    predicted, all_scores = predict_bert_multilabel(title, abstract)

    # Render results in simple HTML
    results_html = "<h3>Predicted Labels:</h3><ul>"
    for item in predicted:
        results_html += f"<li>{item['label']}: {item['score']}</li>"
    results_html += "</ul>"

    results_html += "<h3>All Scores:</h3><ul>"
    for label, score in all_scores.items():
        results_html += f"<li>{label}: {score}</li>"
    results_html += "</ul>"

    return results_html + '<br><a href="/">Back</a>'

if __name__ == "__main__":
    app.run(debug=True)