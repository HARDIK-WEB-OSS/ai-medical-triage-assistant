from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import uuid
import logging

from app.core.database import get_db
from app.models.schemas import TriageRequest, TriageResponse, OfflineCaseSync
from app.models.case import TriageCase
from app.services.triage_engine import run_triage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/triage", tags=["triage"])


@router.post("/assess", response_model=TriageResponse, summary="Run AI triage assessment")
async def assess_patient(
    request: TriageRequest,
    db: Session = Depends(get_db)
):
    """
    Main triage endpoint.
    Accepts patient symptoms → returns urgency level + explanation.
    Works fully offline on client side; this endpoint syncs results to DB.
    """
    try:
        # Run the full triage pipeline
        result = await run_triage(request)

        # Persist to database
        case = TriageCase(
            case_uid=result["case_uid"],
            patient_age=request.patient_age,
            patient_sex=request.patient_sex.value,
            chief_complaint=request.chief_complaint,
            symptoms_raw=result["extracted_symptoms"],
            symptom_duration_hours=request.symptom_duration_hours,
            vital_signs=request.vital_signs.dict() if request.vital_signs else None,
            urgency_level=result["urgency_level"],
            confidence_score=result["confidence_score"],
            predicted_conditions=[c.dict() for c in result["predicted_conditions"]],
            clinical_explanation=result["clinical_explanation"],
            recommended_action=result["recommended_action"],
            requires_human_review=result["requires_human_review"],
            nurse_id=request.nurse_id,
            clinic_id=request.clinic_id,
            device_id=request.device_id,
        )
        db.add(case)
        db.commit()
        db.refresh(case)

        return TriageResponse(
            case_id=case.id,
            **result
        )

    except Exception as e:
        logger.error(f"Triage assessment failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Triage assessment failed: {str(e)}"
        )


@router.post("/sync", summary="Sync offline cases from device")
async def sync_offline_cases(
    payload: OfflineCaseSync,
    db: Session = Depends(get_db)
):
    """
    Bulk sync endpoint for offline cases collected when device had no internet.
    Processes each case through the triage pipeline and stores results.
    """
    results = []
    errors = []

    for case_request in payload.cases:
        try:
            # Check if already synced (idempotent)
            if case_request.case_uid:
                existing = db.query(TriageCase).filter(
                    TriageCase.case_uid == case_request.case_uid
                ).first()
                if existing:
                    results.append({"case_uid": case_request.case_uid, "status": "already_exists"})
                    continue

            result = await run_triage(case_request)
            case = TriageCase(
                case_uid=result["case_uid"],
                patient_age=case_request.patient_age,
                patient_sex=case_request.patient_sex.value,
                chief_complaint=case_request.chief_complaint,
                symptoms_raw=result["extracted_symptoms"],
                urgency_level=result["urgency_level"],
                confidence_score=result["confidence_score"],
                clinical_explanation=result["clinical_explanation"],
                recommended_action=result["recommended_action"],
                requires_human_review=result["requires_human_review"],
                device_id=payload.device_id,
                is_synced=True,
            )
            db.add(case)
            results.append({"case_uid": result["case_uid"], "status": "synced"})

        except Exception as e:
            errors.append({"case_uid": getattr(case_request, 'case_uid', 'unknown'), "error": str(e)})

    db.commit()
    return {"synced": len(results), "errors": len(errors), "details": results}
