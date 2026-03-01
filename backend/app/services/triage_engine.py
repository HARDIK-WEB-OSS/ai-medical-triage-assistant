"""
Triage Engine: Orchestrates the full assessment pipeline.

Input: TriageRequest (symptoms, age, sex, vitals)
Output: TriageResponse (urgency, confidence, explanation, actions)

Pipeline:
1. Extract symptoms from free text (NLP)
2. Classify urgency (XGBoost ML)
3. Generate explanation (RAG + LLM or rule-based)
4. Apply safety overrides
5. Return structured response
"""

import time
import uuid
import logging
from typing import Dict, Any
from app.services.symptom_extractor import extract_symptoms
from app.services.ml_classifier import get_classifier
from app.services.rag_pipeline import get_rag_pipeline, get_recommended_action
from app.models.schemas import TriageRequest, TriageResponse, ConditionPrediction
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def run_triage(request: TriageRequest) -> Dict[str, Any]:
    """
    Full triage pipeline. Returns dict matching TriageResponse schema.
    """
    start_time = time.time()

    # ── Step 1: Extract Symptoms ──
    symptoms, extraction_meta = extract_symptoms(
        text=request.chief_complaint,
        age=request.patient_age,
        sex=request.patient_sex.value
    )

    logger.info(f"Extracted {len(symptoms)} symptoms from complaint: {symptoms}")

    # ── Step 2: Classify Urgency ──
    classifier = get_classifier()
    vital_signs_dict = request.vital_signs.dict() if request.vital_signs else None

    urgency_level, confidence, raw_predictions = classifier.classify(
        symptoms=symptoms,
        age=request.patient_age,
        sex=request.patient_sex.value,
        vital_signs=vital_signs_dict
    )

    # ── Step 3: Safety Flag ──
    requires_review = confidence < settings.confidence_threshold
    if requires_review:
        logger.warning(f"Low confidence ({confidence:.2f}) — flagging for human review")

    # ── Step 4: Generate Explanation (RAG) ──
    rag = get_rag_pipeline()
    explanation, rag_used = rag.generate_explanation(
        urgency=urgency_level,
        symptoms=symptoms,
        age=request.patient_age,
        sex=request.patient_sex.value,
        chief_complaint=request.chief_complaint
    )

    # ── Step 5: Recommended Action ──
    action = get_recommended_action(urgency_level, symptoms)

    # ── Step 6: Format Predictions ──
    conditions = [
        ConditionPrediction(
            condition=p["condition"],
            probability=round(p["probability"], 3)
        )
        for p in raw_predictions[:3]
    ]

    processing_time = int((time.time() - start_time) * 1000)

    return {
        "urgency_level": urgency_level,
        "confidence_score": round(confidence, 3),
        "requires_human_review": requires_review,
        "recommended_action": action,
        "clinical_explanation": explanation,
        "predicted_conditions": conditions,
        "extracted_symptoms": symptoms,
        "processing_time_ms": processing_time,
        "rag_used": rag_used,
        "case_uid": request.case_uid or str(uuid.uuid4()),
    }
