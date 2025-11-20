# 🌿 AfroDB Drug Discovery Platform

AI-powered chatbot for discovering drug candidates from African medicinal plants.

## Features
- 🔍 Natural language plant search
- 🧬 15+ ML models for bioactivity prediction
- ☠️ Toxicity screening (Tox21)
- 🎯 Molecular docking validation
- 💊 Drug-likeness filtering

## Usage
```bash
streamlit run app/chatbot_app.py
```

Or use Python:
```python
from main_pipeline import ChatbotAgent
chatbot = ChatbotAgent(df)
print(chatbot.chat("Screen bitter leaf for cancer"))
```

## Results
- 47K drug-like compounds screened
- 6 diseases covered
- Validated with ChEMBL data
