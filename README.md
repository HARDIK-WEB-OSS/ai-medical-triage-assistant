# 🏥 AI Medical Triage Assistant
> Low-Cost Edge AI for Rural Healthcare — Complete Working Prototype

---

## File Structure

```
triage/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app entry point
│   │   ├── api/
│   │   │   ├── triage.py            # POST /triage/assess, /triage/sync
│   │   │   └── cases.py             # GET/PATCH /cases/*
│   │   ├── core/
│   │   │   ├── config.py            # Settings (env vars)
│   │   │   ├── database.py          # SQLAlchemy PostgreSQL
│   │   │   └── security.py          # JWT utilities
│   │   ├── models/
│   │   │   ├── case.py              # TriageCase SQLAlchemy model
│   │   │   └── schemas.py           # Pydantic request/response schemas
│   │   └── services/
│   │       ├── triage_engine.py     # Orchestrates full pipeline
│   │       ├── ml_classifier.py     # XGBoost + red flag rules
│   │       ├── symptom_extractor.py # NLP: text → symptom tokens
│   │       └── rag_pipeline.py      # LangChain + ChromaDB + LLM
│   ├── ml/
│   │   └── train_model.py           # Train XGBoost classifier
│   ├── data/
│   │   └── seed_guidelines.py       # Seed ChromaDB with WHO protocols
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   └── services/
│   │       └── api.js               # Axios + offline queue + JS classifier
│   └── package.json
│
├── web-demo/
│   └── index.html                   # COMPLETE STANDALONE DEMO — open in browser
│
└── docs/
    └── ARCHITECTURE.md              # Full system architecture
```

---

## Quick Start (3 options)

### Option 1: Web Demo — Zero Setup (Hackathon)
```bash
# Just open this file in any browser:
open triage/web-demo/index.html
# Works offline. No server. No dependencies.
```

### Option 2: Full Backend
```bash
cd triage/backend

# 1. Install Python deps
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env — set your DATABASE_URL and optionally OPENAI_API_KEY

# 3. Start PostgreSQL (or use Docker)
docker run -d -p 5432:5432 -e POSTGRES_DB=triage_db \
  -e POSTGRES_USER=triage_user -e POSTGRES_PASSWORD=triage_pass postgres:15

# 4. Train the ML model
python ml/train_model.py

# 5. Seed RAG knowledge base
python data/seed_guidelines.py

# 6. Start API
uvicorn app.main:app --reload --port 8000

# API docs: http://localhost:8000/docs
```

### Option 3: Docker Full Stack
```bash
cd triage/backend
OPENAI_API_KEY=sk-your-key docker-compose up --build
# API at http://localhost:8000
```

---

## The AI Pipeline — How It Works

```
Input: "45yo male, chest pain, left arm pain, sweating"
   ↓
Symptom Extractor   → [chest_pain, arm_pain_left]
   ↓
Red Flag Rules      → chest_pain + arm_pain_left = MATCH
   ↓ (bypasses ML model for safety)
Output: EMERGENCY, confidence=0.97
   ↓
RAG Pipeline        → Retrieves ACS protocol from ChromaDB
   ↓
LLM Explanation     → "Symptoms consistent with possible MI. Transfer within 90min..."
   ↓
Response:
{
  urgency_level: "EMERGENCY",
  confidence_score: 0.97,
  recommended_action: "🔴 Transfer IMMEDIATELY...",
  clinical_explanation: "...",
  requires_human_review: false
}
```

**Critical design rule:** The ML model classifies. The LLM only explains. LLMs hallucinate — they never make the urgency decision.

---

## Testing the API

```bash
# Test triage assessment
curl -X POST http://localhost:8000/api/v1/triage/assess \
  -H "Content-Type: application/json" \
  -d '{
    "patient_age": 45,
    "patient_sex": "male",
    "chief_complaint": "severe chest pain radiating to left arm with sweating",
    "nurse_id": "nurse_001"
  }'

# Expected: urgency_level = "EMERGENCY"

# Test non-urgent case
curl -X POST http://localhost:8000/api/v1/triage/assess \
  -H "Content-Type: application/json" \
  -d '{
    "patient_age": 28,
    "patient_sex": "female",
    "chief_complaint": "mild headache and sore throat since morning"
  }'

# Expected: urgency_level = "NON_URGENT"

# View dashboard stats
curl http://localhost:8000/api/v1/cases/stats/summary
```

---

## What To Build Next (v2 Roadmap)

1. **Real clinical data** — DDXPlus dataset + 500 validated Indian cases → retrain model
2. **Agentic symptom interview** — LangGraph state machine for dynamic follow-up questions
3. **Hindi UI** — Complete i18n with react-i18next
4. **Doctor dashboard** — React web app for case review and model validation
5. **CDSCO SaMD filing** — Regulatory approval required for formal clinic deployment
6. **Vital signs integration** — Bluetooth BP monitor / pulse oximeter pairing

---

## Important Warnings

**This is a prototype.** Before clinical deployment:
- Get real clinical data labeled by qualified physicians
- Achieve >80% accuracy on real (not synthetic) data  
- Obtain CDSCO SaMD regulatory approval
- Have a licensed physician validate outputs
- Run a supervised pilot at one clinic before scaling
- The synthetic training data included is for demo only
