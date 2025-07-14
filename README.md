# wireshark-threat-detector


# 🚨 Wireshark Threat Detector using DistilBERT

This project fine-tunes a `distilbert-base-uncased` model for binary classification of **network traffic**, predicting whether a packet is **benign** or **malicious**, using Wireshark-style data.

✅ Live on Hugging Face: [https://huggingface.co/TanmaySK/results](https://huggingface.co/TanmaySK/results)

---

## 🧠 Model Overview

- **Base Model**: [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)
- **Task**: Binary text classification (`0` = Benign, `1` = Malicious)
- **Input Format**: Packet-level metadata (e.g. `SrcIP`, `DstIP`, `Protocol`, `Flags`)
- **Use Case**: Network threat detection / intrusion detection systems (IDS)

---

## 📊 Evaluation Results

| Metric      | Value |
|-------------|-------|
| Accuracy    | 1.0   |
| Precision   | 1.0   |
| Recall      | 1.0   |
| F1 Score    | 1.0   |
| Eval Loss   | 0.0000 |

---

## 📁 Project Structure

```bash
wireshark-threat-detector/
├── README.md
├── predict.py
├── batch_predict.py
├── wireshark_unlabeled.csv
└── requirements.txt
```

---

## 🛠 Setup

```bash
pip install -r requirements.txt
```

---

## 🔮 Single Prediction

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="TanmaySK/results")
text = "SrcIP:10.0.0.1 DstIP:192.168.1.1 Protocol:TCP Flags:SYN"
result = classifier(text)

label_map = {{"LABEL_0": "Benign", "LABEL_1": "Malicious"}}
print(f"Prediction: {{label_map[result[0]['label']]}} (Confidence: {{result[0]['score']:.4f}})")
```

---

## 📁 CSV Batch Prediction

```python
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("TanmaySK/results")
tokenizer = AutoTokenizer.from_pretrained("TanmaySK/results")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

df = pd.read_csv("wireshark_unlabeled.csv")
label_map = {{0: "Benign", 1: "Malicious"}}
predictions = []

for text in df["input"]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {{k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}}
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        predictions.append(pred)

df["PredictedLabel"] = predictions
df["PredictionText"] = [label_map[p] for p in predictions]
df.to_csv("wireshark_predictions.csv", index=False)
print("✅ Saved to wireshark_predictions.csv")
```

---

## 🔗 Hugging Face Model Card

Live model link: [TanmaySK/results](https://huggingface.co/TanmaySK/results)

---

## 🔐 License

Apache License 2.0 — free to use and modify.

---

## 👤 Author

Made with ❤️ by **Tanmay Khadikar**

GitHub: [TanmaySK](https://github.com/TanmaySK)
