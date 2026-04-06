from bertopic_inference import BERTopicPredictor

p = BERTopicPredictor(model_dir="topic_analysis/BERTopic/bertopic_models")
res = p.predict(title="Test title", abstract="Short test abstract")

if isinstance(res, dict):
    for sc, pred in res.items():
        print(sc, pred.topic_id, pred.topic_label, pred.top_keywords[:8], round(pred.probability,4), pred.is_outlier)
else:
    pred = res
    print(pred.topic_id, pred.topic_label, pred.top_keywords, pred.probability, pred.is_outlier)