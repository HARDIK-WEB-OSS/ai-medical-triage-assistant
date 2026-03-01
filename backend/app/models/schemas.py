from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class UrgencyLevel(str, Enum):
    EMERGENCY = "EMERGENCY"
    URGENT = "URGENT"
    NON_URGENT = "NON_URGENT"
    UNKNOWN = "UNKNOWN"

class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"

# ── TRIAGE REQUEST ──
class VitalSigns(BaseModel):
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None
    heart_rate: Optional[int] = None
    temperature_celsius: Optional[float] = None
    spo2_percent: Optional[int] = None
    respiratory_rate: Optional[int] = None

class TriageRequest(BaseModel):
    patient_age: int = Field(..., ge=0, le=120, description="Patient age in years")
    patient_sex: Sex
    chief_complaint: str = Field(..., min_length=3, max_length=2000, description="Primary symptom description")
    symptom_duration_hours: Optional[int] = Field(None, ge=0, description="Hours since symptoms started")
    vital_signs: Optional[VitalSigns] = None
    nurse_id: Optional[str] = None
    clinic_id: Optional[str] = None
    device_id: Optional[str] = None
    case_uid: Optional[str] = None  # Client-generated UUID for offline sync

# ── TRIAGE RESPONSE ──
class ConditionPrediction(BaseModel):
    condition: str
    probability: float
    icd10_code: Optional[str] = None

class TriageResponse(BaseModel):
    case_id: int
    case_uid: str
    urgency_level: UrgencyLevel
    confidence_score: float
    requires_human_review: bool
    recommended_action: str
    clinical_explanation: str
    predicted_conditions: List[ConditionPrediction]
    extracted_symptoms: List[str]
    processing_time_ms: int
    rag_used: bool  # Whether online RAG was available

# ── CASE MANAGEMENT ──
class CaseListItem(BaseModel):
    id: int
    case_uid: str
    patient_age: int
    patient_sex: str
    chief_complaint: str
    urgency_level: str
    confidence_score: Optional[float]
    requires_human_review: bool
    is_validated: bool
    created_at: datetime

    class Config:
        from_attributes = True

class DoctorOverride(BaseModel):
    corrected_urgency: UrgencyLevel
    doctor_notes: Optional[str] = None
    doctor_id: str

# ── AUTH ──
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

# ── OFFLINE SYNC ──
class OfflineCaseSync(BaseModel):
    cases: List[TriageRequest]
    device_id: str
