from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from datetime import datetime

from app.core.database import get_db
from app.models.schemas import CaseListItem, DoctorOverride
from app.models.case import TriageCase

router = APIRouter(prefix="/cases", tags=["cases"])


@router.get("/", response_model=List[CaseListItem], summary="List triage cases")
def list_cases(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    urgency: Optional[str] = Query(None, description="Filter by urgency level"),
    requires_review: Optional[bool] = Query(None),
    clinic_id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(TriageCase)
    if urgency:
        query = query.filter(TriageCase.urgency_level == urgency.upper())
    if requires_review is not None:
        query = query.filter(TriageCase.requires_human_review == requires_review)
    if clinic_id:
        query = query.filter(TriageCase.clinic_id == clinic_id)

    cases = query.order_by(desc(TriageCase.created_at)).offset(skip).limit(limit).all()
    return cases


@router.get("/{case_id}", summary="Get case details")
def get_case(case_id: int, db: Session = Depends(get_db)):
    case = db.query(TriageCase).filter(TriageCase.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@router.patch("/{case_id}/override", summary="Doctor override — correct urgency level")
def doctor_override(
    case_id: int,
    override: DoctorOverride,
    db: Session = Depends(get_db)
):
    """
    Allows doctor to correct an AI triage decision.
    Override data feeds back into model improvement pipeline.
    """
    case = db.query(TriageCase).filter(TriageCase.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    case.doctor_override = override.corrected_urgency.value
    case.doctor_notes = override.doctor_notes
    case.is_validated = True
    db.commit()

    return {
        "message": "Override recorded. Thank you — this improves the model.",
        "original_urgency": case.urgency_level,
        "corrected_urgency": override.corrected_urgency.value,
        "case_id": case_id
    }


@router.get("/stats/summary", summary="Clinic dashboard stats")
def get_stats(clinic_id: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(TriageCase)
    if clinic_id:
        query = query.filter(TriageCase.clinic_id == clinic_id)

    total = query.count()
    emergency = query.filter(TriageCase.urgency_level == "EMERGENCY").count()
    urgent = query.filter(TriageCase.urgency_level == "URGENT").count()
    non_urgent = query.filter(TriageCase.urgency_level == "NON_URGENT").count()
    needs_review = query.filter(TriageCase.requires_human_review == True).count()
    validated = query.filter(TriageCase.is_validated == True).count()

    return {
        "total_cases": total,
        "by_urgency": {
            "EMERGENCY": emergency,
            "URGENT": urgent,
            "NON_URGENT": non_urgent,
        },
        "needs_human_review": needs_review,
        "validated_by_doctor": validated,
        "validation_rate": round(validated / total * 100, 1) if total > 0 else 0
    }
