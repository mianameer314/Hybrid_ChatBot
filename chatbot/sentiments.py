from transformers import pipeline
import warnings

# Silence noisy warnings about torch.classes
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

# Explicit model (small & fast). 
# For multilingual, try "cardiffnlp/twitter-xlm-roberta-base-sentiment".
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1              # force CPU (avoid GPU/extension errors)
)

LABEL_MAP = {
    "POSITIVE": "Positive",
    "NEGATIVE": "Negative",
    "NEUTRAL": "Neutral",  # some models return NEUTRAL
}

def analyze(text: str) -> dict:
    res = sentiment_model(text)[0]
    raw = res["label"].upper()
    label = LABEL_MAP.get(
        raw,
        "Positive" if "POS" in raw else "Negative" if "NEG" in raw else "Neutral"
    )
    # simple tone heuristic
    lower = text.lower()
    tone = (
        "Enthusiastic" if any(w in lower for w in ["amazing", "awesome", "great", "!"]) else
        "Polite" if any(p in lower for p in ["please", "thank you"]) else
        "Urgent" if any(u in lower for u in ["urgent", "asap", "immediately"]) else
        "Casual"
    )
    return {
        "label": label,
        "score": round(float(res.get("score", 0.0)), 3),
        "tone": tone
    }
