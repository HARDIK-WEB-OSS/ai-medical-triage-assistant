# Architecture Documentation — AI Medical Triage Assistant

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                                  │
│                                                                 │
│   React Native App (Android/iOS)    Web Demo (index.html)       │
│   ┌────────────────────────┐        ┌──────────────────────┐    │
│   │ UI Screens             │        │ Standalone demo      │    │
│   │ - PatientInput         │        │ Full JS classifier   │    │
│   │ - TriageResult         │        │ No server needed     │    │
│   │ - CaseHistory          │        └──────────────────────┘    │
│   │ - DoctorDashboard      │                                    │
│   └────────┬───────────────┘                                    │
│            │ Online/Offline detection                           │
│   ┌────────▼───────────────┐                                    │
│   │ Offline Classifier (JS)│  ← Mirrors Python ML logic        │
│   │ - Symptom extraction   │  ← Red flag rules always run      │
│   │ - Red flag detection   │  ← Works with ZERO internet       │
│   │ - Local SQLite storage │                                    │
│   │ - Offline sync queue   │                                    │
│   └────────┬───────────────┘                                    │
└────────────┼────────────────────────────────────────────────────┘
             │ When online: REST API calls
             │ When offline: queue for later sync
┌────────────▼────────────────────────────────────────────────────┐
│                    BACKEND LAYER (FastAPI)                      │
│                                                                 │
│  POST /api/v1/triage/assess                                     │
│       │                                                         │
│  ┌────▼──────────────────────────────────────────────────────┐  │
│  │                  TRIAGE ENGINE                            │  │
│  │                                                           │  │
│  │  1. Symptom Extractor (NLP)                              │  │
│  │     └─ Keyword + regex matching against 60+ symptom      │  │
│  │        vocabulary → standardized symptom token list      │  │
│  │                                                           │  │
│  │  2. ML Classifier (XGBoost) ← OFFLINE CAPABLE            │  │
│  │     ├─ Layer 1: Red Flag Rules (safety override)          │  │
│  │     │   └─ Pattern match on critical symptom combos      │  │
│  │     │      → Always forces EMERGENCY classification      │  │
│  │     └─ Layer 2: XGBoost Model                            │  │
│  │         ├─ Input: 83-dim feature vector                   │  │
│  │         │   (80 symptoms + age_norm + sex + age_extreme)  │  │
│  │         ├─ Output: P(NON_URGENT), P(URGENT), P(EMERGENCY) │  │
│  │         └─ Confidence < 0.70 → flag for human review     │  │
│  │                                                           │  │
│  │  3. RAG Pipeline (LangChain + ChromaDB) ← ONLINE ONLY    │  │
│  │     ├─ Query: symptom tokens → ChromaDB similarity search │  │
│  │     ├─ Retrieve: top-3 relevant clinical guidelines       │  │
│  │     ├─ LLM: GPT-4o generates plain-language explanation   │  │
│  │     └─ Fallback: rule-based explanation (always works)    │  │
│  │                                                           │  │
│  │  4. Response Assembly                                     │  │
│  │     └─ urgency + confidence + action + explanation        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ PostgreSQL  │  │   Redis     │  │     ChromaDB            │ │
│  │ Case logs   │  │ Cache layer │  │ Clinical guidelines     │ │
│  │ Audit trail │  │ Rate limits │  │ WHO + India protocols   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow — Single Assessment

```
Nurse types: "45yo male, chest pain, left arm pain, sweating"
                    │
                    ▼
         Symptom Extractor
         Detects: [chest_pain, arm_pain_left, shortness_of_breath implied by sweating context]
                    │
                    ▼
         Red Flag Checker
         Pattern match: [chest_pain + arm_pain_left] → RED FLAG TRIGGERED
                    │
                    ▼
         Returns: EMERGENCY, confidence=0.97
         (Red flags bypass ML model entirely — safety critical)
                    │
                    ▼
         RAG Pipeline (if online)
         Query: "chest pain left arm pain EMERGENCY cardiac"
         ChromaDB returns: ACS Recognition guideline, WHO ETAT guideline
         GPT-4o generates: "Symptoms consistent with possible MI. Transfer within 90min..."
                    │
                    ▼
         Response:
         {
           urgency: "EMERGENCY",
           confidence: 0.97,
           action: "🔴 Transfer IMMEDIATELY...",
           explanation: "Symptoms consistent with possible acute cardiac event...",
           symptoms: ["chest_pain", "arm_pain_left"],
           requires_review: false
         }
```

## Offline Architecture

```
Device has no internet:
         │
         ▼
  JS Red Flag Rules → Immediate classification in <100ms
         │
         ▼
  Result displayed to nurse
         │
         ▼
  Case stored in SQLite (device)
         │
  When internet restored:
         ▼
  Sync queue → POST /api/v1/triage/sync (bulk upload)
  Server re-classifies with full ML + RAG
  Server returns corrected results if different
  Local records updated
```

## Model Architecture

```
Input: 83-dimensional feature vector
  ├── 80 binary symptom features (0 or 1)
  ├── 1 normalized age feature (age/100)
  ├── 1 binary sex feature (male=1, female=0)
  └── 1 age extremes flag (age<5 or age>70 = 1)

XGBoost Classifier:
  ├── n_estimators: 200
  ├── max_depth: 6
  ├── learning_rate: 0.1
  └── Output: P(class=0), P(class=1), P(class=2)
             NON_URGENT    URGENT      EMERGENCY

Class weights: NON_URGENT=1.0, URGENT=1.2, EMERGENCY=3.0
(Emergency weighted 3x — we prefer false positives over false negatives)
```

## RAG Knowledge Base

```
ChromaDB Collection: "clinical_guidelines"
  ├── WHO ETAT — Emergency Signs
  ├── Acute Coronary Syndrome Recognition
  ├── Stroke Recognition (FAST Protocol)
  ├── Dengue — Indian Context Triage
  ├── Fever with Rash — Differential Triage
  ├── Pediatric Emergency Signs
  ├── Obstetric Emergencies
  ├── Non-Urgent Fever Management
  ├── Anaphylaxis Recognition and Response
  └── Abdominal Pain — Triage Guide

Embedding model: all-MiniLM-L6-v2 (local, no API key needed)
LLM for explanation: GPT-4o (optional — falls back to rule-based)
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/v1/triage/assess | Run triage assessment |
| POST | /api/v1/triage/sync | Bulk sync offline cases |
| GET | /api/v1/cases/ | List cases with filters |
| GET | /api/v1/cases/{id} | Get case details |
| PATCH | /api/v1/cases/{id}/override | Doctor override |
| GET | /api/v1/cases/stats/summary | Dashboard stats |
| GET | /health | Health check |
| GET | /docs | Swagger UI |

## Security Boundaries

- Patient data encrypted AES-256 on device
- JWT auth on all API endpoints
- No patient PII in ML training pipeline
- Every AI decision logged with full audit trail
- Doctor override recorded for model improvement
- Confidence threshold (0.70) as safety gate
