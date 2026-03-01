from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import enum

class UrgencyLevel(str, enum.Enum):
    EMERGENCY = "EMERGENCY"
    URGENT = "URGENT"
    NON_URGENT = "NON_URGENT"
    UNKNOWN = "UNKNOWN"

class TriageCase(Base):
    __tablename__ = "triage_cases"

    id = Column(Integer, primary_key=True, index=True)
    case_uid = Column(String(36), unique=True, index=True)  # UUID for offline sync

    # Patient info
    patient_age = Column(Integer, nullable=False)
    patient_sex = Column(String(10), nullable=False)
    patient_weight_kg = Column(Float, nullable=True)

    # Symptoms
    chief_complaint = Column(Text, nullable=False)        # Free text input
    symptoms_raw = Column(JSON, nullable=True)            # Extracted symptom list
    symptom_duration_hours = Column(Integer, nullable=True)
    vital_signs = Column(JSON, nullable=True)             # BP, SpO2, temp if available

    # AI Output
    urgency_level = Column(String(20), nullable=False, default="UNKNOWN")
    confidence_score = Column(Float, nullable=True)
    predicted_conditions = Column(JSON, nullable=True)    # Top N possible conditions
    clinical_explanation = Column(Text, nullable=True)    # RAG-generated explanation
    recommended_action = Column(Text, nullable=True)
    requires_human_review = Column(Boolean, default=False) # Confidence < threshold

    # Audit
    nurse_id = Column(String(100), nullable=True)
    clinic_id = Column(String(100), nullable=True)
    doctor_override = Column(String(20), nullable=True)   # Doctor's corrected urgency
    doctor_notes = Column(Text, nullable=True)
    is_validated = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    synced_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Offline sync
    device_id = Column(String(100), nullable=True)
    is_synced = Column(Boolean, default=True)
