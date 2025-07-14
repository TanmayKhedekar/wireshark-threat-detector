# wireshark-threat-detector


# 🧠 results – DistilBERT for Malicious Traffic Classification

This model is a fine-tuned version of [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) for **binary classification of network traffic**, especially useful for distinguishing **malicious vs. benign** packets based on preprocessed Wireshark-style logs.

---

## 📊 Evaluation Results

| Metric      | Value |
|-------------|-------|
| Accuracy    | 1.0   |
| Precision   | 1.0   |
| Recall      | 1.0   |
| F1 Score    | 1.0   |
| Eval Loss   | 0.0000 |

> ⚠️ These perfect results are on the validation set and may not generalize to unseen or noisy real-world data. Be sure to test on diverse inputs.

---

## 🧩 Model Description

This model uses the lightweight and efficient **DistilBERT** transformer, fine-tuned for binary classification. Input data should be short text sequences (e.g., protocol descriptions, IP headers, or Wireshark logs). 

---

## 💡 Intended Use & Limitations

### ✅ Intended Uses

- **Malicious traffic detection** (from packet text)
- **Intrusion detection system (IDS)** aid
- Sentiment analysis or spam detection (if retrained)

### ❌ Limitations

- English and network-related text only
- Binary classification (0 = benign, 1 = malicious)
- Not trained on raw PCAPs — requires preprocessing

---

## 🧪 Example Usage

### 🔌 Hugging Face Pipeline (Single Prediction)

```python
from transformers import pipeline

# Load from Hugging Face Hub
classifier = pipeline("text-classification", model="TanmaySK/results")

# Predict
text = "SrcIP:10.0.0.1 DstIP:192.168.1.1 Protocol:TCP Flags:SYN"
result = classifier(text)

# Interpret label
label_map = {"LABEL_0": "Benign", "LABEL_1": "Malicious"}
print(f"Prediction: {label_map[result[0]['label']]} (Confidence: {result[0]['score']:.4f})")
```

---

### 📁 CSV Batch Prediction (Local Wireshark Data)

```python
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained("TanmaySK/results")
tokenizer = AutoTokenizer.from_pretrained("TanmaySK/results")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load CSV
df = pd.read_csv("wireshark_unlabeled.csv")  # Must have 'input' column
label_map = {0: "Benign", 1: "Malicious"}
predictions = []

# Predict each row
for text in df["input"]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        predictions.append(pred)

# Save results
df["PredictedLabel"] = predictions
df["PredictionText"] = [label_map[p] for p in predictions]
df.to_csv("wireshark_predictions.csv", index=False)
print("✅ Saved to wireshark_predictions.csv")
```
