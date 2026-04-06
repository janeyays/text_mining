import traceback
from bertopic import BERTopic
p = "topic_analysis/BERTopic/bertopic_models/bertopic_econ_EM"
try:
    BERTopic.load(p)
    print("Loaded OK")
except Exception:
    traceback.print_exc()