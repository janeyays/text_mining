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
BERT_THRESHOLD = 0.025

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
        <title>Paper Classifier</title>
        <link href="https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
        <style>
            *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

            body {
                font-family: 'DM Sans', sans-serif;
                background: #f7f5f0;
                color: #1a1a18;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 3rem 1.5rem;
            }

            header {
                text-align: center;
                margin-bottom: 2.5rem;
            }

            header h1 {
                font-family: 'Lora', serif;
                font-size: 2rem;
                font-weight: 600;
                letter-spacing: -0.02em;
                color: #1a1a18;
            }

            header p {
                margin-top: 0.4rem;
                font-size: 0.9rem;
                color: #6b6b64;
                font-weight: 300;
            }

            .card {
                background: #fff;
                border-radius: 16px;
                border: 0.5px solid #ddddd5;
                padding: 2rem;
                width: 100%;
                max-width: 680px;
                margin-bottom: 1.5rem;
            }

            label {
                display: block;
                font-size: 0.75rem;
                font-weight: 500;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #6b6b64;
                margin-bottom: 0.5rem;
            }

            input[type="text"], textarea {
                width: 100%;
                border: 0.5px solid #ddddd5;
                border-radius: 8px;
                padding: 0.75rem 1rem;
                font-family: 'DM Sans', sans-serif;
                font-size: 0.95rem;
                background: #fafaf8;
                color: #1a1a18;
                outline: none;
                transition: border-color 0.2s;
                resize: vertical;
            }

            input[type="text"]:focus, textarea:focus {
                border-color: #888880;
                background: #fff;
            }

            .field + .field { margin-top: 1.25rem; }

            .predict-btn {
                margin-top: 1.5rem;
                width: 100%;
                padding: 0.85rem;
                background: #1a1a18;
                color: #f7f5f0;
                border: none;
                border-radius: 8px;
                font-family: 'DM Sans', sans-serif;
                font-size: 0.95rem;
                font-weight: 500;
                cursor: pointer;
                transition: background 0.2s, transform 0.1s;
            }

            .predict-btn:hover { background: #333330; }
            .predict-btn:active { transform: scale(0.99); }
            .predict-btn:disabled { background: #b0b0a8; cursor: not-allowed; }

            /* Modal overlay */
            .overlay {
                display: none;
                position: fixed;
                inset: 0;
                background: rgba(26,26,24,0.45);
                z-index: 100;
                align-items: center;
                justify-content: center;
                padding: 1.5rem;
            }

            .overlay.active { display: flex; }

            .modal {
                background: #fff;
                border-radius: 16px;
                border: 0.5px solid #ddddd5;
                padding: 2rem;
                width: 100%;
                max-width: 600px;
                max-height: 85vh;
                overflow-y: auto;
                position: relative;
            }

            .modal-title {
                font-family: 'Lora', serif;
                font-size: 1.2rem;
                font-weight: 600;
                margin-bottom: 0.3rem;
            }

            .modal-sub {
                font-size: 0.83rem;
                color: #6b6b64;
                margin-bottom: 1.5rem;
            }

            .section-label {
                font-size: 0.7rem;
                font-weight: 500;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #6b6b64;
                margin-bottom: 0.6rem;
            }

            .tags {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 1.25rem;
            }

            .tag {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 5px 12px;
                border-radius: 999px;
                font-size: 0.82rem;
                font-weight: 400;
                cursor: pointer;
                border: 0.5px solid;
                transition: all 0.15s;
                user-select: none;
            }

            .tag.selected {
                background: #1a1a18;
                color: #f7f5f0;
                border-color: #1a1a18;
            }

            .tag.unselected {
                background: #fafaf8;
                color: #6b6b64;
                border-color: #ddddd5;
            }

            .tag.unselected:hover {
                border-color: #888880;
                color: #1a1a18;
            }

            .tag .remove {
                font-size: 1rem;
                line-height: 1;
                opacity: 0.6;
            }

            .tag.unselected .remove { display: none; }

            .add-row {
                display: flex;
                gap: 8px;
                margin-top: 0.5rem;
                margin-bottom: 1.5rem;
            }

            .add-row input {
                flex: 1;
                border: 0.5px solid #ddddd5;
                border-radius: 8px;
                padding: 0.6rem 0.9rem;
                font-family: 'DM Sans', sans-serif;
                font-size: 0.88rem;
                background: #fafaf8;
                outline: none;
            }

            .add-row input:focus { border-color: #888880; background: #fff; }

            .add-btn {
                padding: 0.6rem 1rem;
                background: transparent;
                border: 0.5px solid #ddddd5;
                border-radius: 8px;
                font-family: 'DM Sans', sans-serif;
                font-size: 0.88rem;
                cursor: pointer;
                color: #1a1a18;
                transition: background 0.15s;
            }

            .add-btn:hover { background: #f0efe9; }

            .modal-divider {
                height: 0.5px;
                background: #ddddd5;
                margin: 1.25rem 0;
            }

            .modal-actions {
                display: flex;
                gap: 10px;
                justify-content: flex-end;
            }

            .btn-secondary {
                padding: 0.7rem 1.25rem;
                background: transparent;
                border: 0.5px solid #ddddd5;
                border-radius: 8px;
                font-family: 'DM Sans', sans-serif;
                font-size: 0.88rem;
                cursor: pointer;
                color: #6b6b64;
                transition: background 0.15s;
            }

            .btn-secondary:hover { background: #f0efe9; color: #1a1a18; }

            .btn-publish {
                padding: 0.7rem 1.5rem;
                background: #1a1a18;
                color: #f7f5f0;
                border: none;
                border-radius: 8px;
                font-family: 'DM Sans', sans-serif;
                font-size: 0.88rem;
                font-weight: 500;
                cursor: pointer;
                transition: background 0.15s;
            }

            .btn-publish:hover { background: #333330; }

            /* Published paper card */
            .published-card {
                background: #fff;
                border-radius: 16px;
                border: 0.5px solid #ddddd5;
                padding: 2rem;
                width: 100%;
                max-width: 680px;
                display: none;
            }

            .published-card.active { display: block; }

            .pub-eyebrow {
                font-size: 0.7rem;
                font-weight: 500;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                color: #888880;
                margin-bottom: 0.75rem;
            }

            .pub-title {
                font-family: 'Lora', serif;
                font-size: 1.3rem;
                font-weight: 600;
                line-height: 1.4;
                margin-bottom: 1rem;
            }

            .pub-abstract {
                font-size: 0.9rem;
                color: #444440;
                line-height: 1.75;
                margin-bottom: 1.25rem;
            }

            .pub-tags { display: flex; flex-wrap: wrap; gap: 8px; }

            .pub-tag {
                padding: 4px 12px;
                border-radius: 999px;
                background: #f0efe9;
                color: #444440;
                font-size: 0.78rem;
                font-weight: 400;
                border: 0.5px solid #ddddd5;
            }

            .pub-actions {
                margin-top: 1.25rem;
                padding-top: 1.25rem;
                border-top: 0.5px solid #ddddd5;
            }

            .btn-reset {
                font-family: 'DM Sans', sans-serif;
                font-size: 0.83rem;
                color: #6b6b64;
                background: none;
                border: none;
                cursor: pointer;
                text-decoration: underline;
                text-underline-offset: 2px;
            }

            .btn-reset:hover { color: #1a1a18; }

            .spinner {
                display: inline-block;
                width: 14px; height: 14px;
                border: 2px solid #f7f5f0;
                border-top-color: transparent;
                border-radius: 50%;
                animation: spin 0.7s linear infinite;
                vertical-align: middle;
                margin-right: 6px;
            }

            @keyframes spin { to { transform: rotate(360deg); } }
        </style>
    </head>
    <body>

        <header>
            <h1>Paper Classifier</h1>
            <p>Paste your title and abstract here and we'll help you suggest categories automatically.</p>
        </header>

        <!-- Input card -->
        <div class="card" id="input-card">
            <div class="field">
                <label for="title">Title</label>
                <input type="text" id="title" placeholder="e.g. Attention Is All You Need">
            </div>
            <div class="field">
                <label for="abstract">Abstract</label>
                <textarea id="abstract" rows="8" placeholder="Paste your abstract here..."></textarea>
            </div>
            <button class="predict-btn" id="predict-btn" onclick="runPredict()">Predict categories</button>
        </div>

        <!-- Published result card -->
        <div class="published-card" id="published-card">
            <div class="pub-eyebrow">Published</div>
            <div class="pub-title" id="pub-title"></div>
            <div class="pub-abstract" id="pub-abstract"></div>
            <div class="pub-tags" id="pub-tags"></div>
            <div class="pub-actions">
                <button class="btn-reset" onclick="resetAll()">Classify another paper</button>
            </div>
        </div>

        <!-- Modal -->
        <div class="overlay" id="overlay">
            <div class="modal">
                <div class="modal-title">Review categories</div>
                <div class="modal-sub">Toggle to keep or remove, or add your own below.</div>

                <div class="section-label">Predicted</div>
                <div class="tags" id="predicted-tags"></div>

                <div class="section-label">Other categories</div>
                <div class="tags" id="other-tags"></div>


                <div class="modal-divider"></div>

                <div class="modal-actions">
                    <button class="btn-secondary" onclick="closeModal()">Cancel</button>
                    <button class="btn-publish" onclick="publish()">Publish</button>
                </div>
            </div>
        </div>

        <script>
            let allData = {};
            let tagState = {};  // label -> true (selected) / false (unselected)

            async function runPredict() {
                const title = document.getElementById('title').value.trim();
                const abstract = document.getElementById('abstract').value.trim();
                if (!title && !abstract) return;

                const btn = document.getElementById('predict-btn');
                btn.disabled = true;
                btn.innerHTML = '<span class="spinner"></span>Predicting...';

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({title, abstract})
                    });
                    allData = await res.json();
                    openModal(allData);
                } catch(e) {
                    alert('Prediction failed: ' + e);
                } finally {
                    btn.disabled = false;
                    btn.textContent = 'Predict categories';
                }
            }

            function openModal(data) {
                allData = data;
                tagState = {};
                const predicted = data.predicted.map(p => p.label).filter(l => l !== '(no label predicted)');
                const allLabels = Object.keys(data.all_scores);

                // Set initial state: predicted = selected, rest = unselected
                allLabels.forEach(l => { tagState[l] = predicted.includes(l); });

                renderPredictedTags(predicted);
                renderOtherTags(allLabels.filter(l => !predicted.includes(l)));

                document.getElementById('overlay').classList.add('active');
            }

            function closeModal() {
                document.getElementById('overlay').classList.remove('active');
            }

            function renderPredictedTags(labels) {
                const container = document.getElementById('predicted-tags');
                container.innerHTML = '';
                labels.forEach(l => container.appendChild(makeTag(l)));
            }

            function renderOtherTags(labels) {
                const container = document.getElementById('other-tags');
                container.innerHTML = '';
                labels.forEach(l => container.appendChild(makeTag(l)));
            }

            function makeTag(label) {
                const el = document.createElement('span');
                el.className = 'tag ' + (tagState[label] ? 'selected' : 'unselected');
                el.dataset.label = label;
                const score = allData.all_scores[label] || 0;
                el.innerHTML = `${label} <span style="opacity: 0.6; font-size: 0.75em;">(${score})</span><span class="remove">&#x2715;</span>`;
                el.onclick = () => toggleTag(el, label);
                return el;
            }

            function toggleTag(el, label) {
                tagState[label] = !tagState[label];
                el.className = 'tag ' + (tagState[label] ? 'selected' : 'unselected');
            }

            function addCustom() {
                const input = document.getElementById('custom-input');
                const val = input.value.trim();
                if (!val || tagState.hasOwnProperty(val)) { input.value = ''; return; }
                tagState[val] = true;
                const container = document.getElementById('other-tags');
                container.appendChild(makeTag(val));
                input.value = '';
            }

            function publish() {
                const title = document.getElementById('title').value.trim();
                const abstract = document.getElementById('abstract').value.trim();
                const chosen = Object.keys(tagState).filter(l => tagState[l]);

                closeModal();

                // Populate published card
                document.getElementById('pub-title').textContent = title;
                document.getElementById('pub-abstract').textContent = abstract;
                const tagsEl = document.getElementById('pub-tags');
                tagsEl.innerHTML = '';
                chosen.forEach(l => {
                    const t = document.createElement('span');
                    t.className = 'pub-tag';
                    t.textContent = l;
                    tagsEl.appendChild(t);
                });

                document.getElementById('input-card').style.display = 'none';
                document.getElementById('published-card').classList.add('active');
            }

            function resetAll() {
                document.getElementById('title').value = '';
                document.getElementById('abstract').value = '';
                document.getElementById('published-card').classList.remove('active');
                document.getElementById('input-card').style.display = '';
            }

            // Close overlay on background click
            document.getElementById('overlay').addEventListener('click', function(e) {
                if (e.target === this) closeModal();
            });
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