import requests
r = requests.post("http://127.0.0.1:5000/predict_topics", json={"title":"Test","abstract":"Short test", "subcategory":"eess.SY"})
print(r.json())