"""
XGBoost-based urgency classifier.
Trained on symptom features → outputs urgency level + confidence.
Runs fully offline. No internet required.
"""

import numpy as np
import joblib
import os
import logging
from typing import Tuple, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# ── SYMPTOM FEATURE VOCABULARY ──
# These 80 symptoms form the feature vector for the classifier.
# Each input maps to a binary presence vector across these features.
SYMPTOM_VOCAB = [
    # Cardiac/Respiratory - HIGH urgency indicators
    "chest_pain", "chest_tightness", "shortness_of_breath", "difficulty_breathing",
    "palpitations", "irregular_heartbeat", "arm_pain_left", "jaw_pain",
    "coughing_blood", "severe_cough", "wheezing", "cyanosis",

    # Neurological
    "sudden_headache", "confusion", "loss_of_consciousness", "seizure",
    "facial_drooping", "arm_weakness_sudden", "speech_difficulty", "vision_changes",
    "dizziness", "vertigo", "headache_mild", "numbness_tingling",

    # Gastrointestinal
    "severe_abdominal_pain", "vomiting_blood", "black_stool", "rectal_bleeding",
    "abdominal_pain_mild", "nausea", "vomiting", "diarrhea", "constipation",

    # Fever / Infectious
    "high_fever", "fever_mild", "chills", "rigors", "night_sweats",
    "rash_with_fever", "rash_petechial", "rash_mild", "swollen_lymph_nodes",

    # Trauma / Injury
    "head_injury", "loss_of_consciousness_trauma", "deep_laceration", "fracture_suspected",
    "eye_injury", "burn_severe", "burn_mild", "spinal_injury_suspected",

    # Pediatric specific
    "infant_not_feeding", "bulging_fontanelle", "child_seizure", "child_high_fever",
    "child_rash", "child_difficulty_breathing",

    # Obstetric
    "pregnancy_bleeding", "pregnancy_pain", "reduced_fetal_movement",

    # Allergic / Toxic
    "anaphylaxis_signs", "throat_swelling", "hives_widespread", "toxic_ingestion",

    # Musculoskeletal / Other
    "back_pain_severe", "back_pain_mild", "joint_pain", "muscle_pain",
    "urinary_symptoms", "sore_throat", "ear_pain", "eye_redness",
    "fatigue", "weight_loss", "decreased_urine", "edema_legs",

    # Vital sign abnormalities
    "very_low_bp", "very_high_bp", "very_high_hr", "very_low_spo2",
    "very_high_temp", "altered_mental_status",
]

URGENCY_LABELS = {0: "NON_URGENT", 1: "URGENT", 2: "EMERGENCY"}
URGENCY_REVERSE = {"NON_URGENT": 0, "URGENT": 1, "EMERGENCY": 2}

# Rule-based RED FLAG patterns (override model for critical safety)
RED_FLAG_PATTERNS = [
    ["chest_pain", "arm_pain_left"],
    ["chest_pain", "jaw_pain"],
    ["chest_pain", "shortness_of_breath"],
    ["facial_drooping", "arm_weakness_sudden"],
    ["facial_drooping", "speech_difficulty"],
    ["seizure"],
    ["loss_of_consciousness"],
    ["coughing_blood"],
    ["vomiting_blood"],
    ["anaphylaxis_signs"],
    ["throat_swelling"],
    ["very_low_spo2"],
    ["altered_mental_status"],
    ["infant_not_feeding", "child_high_fever"],
    ["pregnancy_bleeding"],
    ["head_injury", "loss_of_consciousness_trauma"],
]


class TriageClassifier:
    """
    Two-layer classifier:
    1. Rule-based red flag detection (safety override — always runs)
    2. XGBoost ML model (probabilistic urgency classification)

    The rule-based layer ensures critical presentations are NEVER
    misclassified by the ML model, even at low confidence.
    """

    def __init__(self, model_path: str):
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"Classifier loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}. Using rule-based fallback only.")
        else:
            logger.warning(f"Model not found at {self.model_path}. Using rule-based fallback.")

    def symptoms_to_features(self, symptoms: List[str], age: int, sex: str,
                              vital_signs: Dict = None) -> np.ndarray:
        """Convert symptom list to feature vector."""
        features = np.zeros(len(SYMPTOM_VOCAB) + 3)  # +3 for age, sex, vital flags

        # Symptom binary features
        for i, symptom in enumerate(SYMPTOM_VOCAB):
            if symptom in symptoms:
                features[i] = 1.0

        # Demographic features
        features[len(SYMPTOM_VOCAB)] = age / 100.0  # Normalize
        features[len(SYMPTOM_VOCAB) + 1] = 1.0 if sex == "male" else 0.0
        features[len(SYMPTOM_VOCAB) + 2] = 1.0 if age < 5 or age > 70 else 0.0  # Extremes flag

        # Vital sign flags
        if vital_signs:
            if vital_signs.get("spo2_percent") and vital_signs["spo2_percent"] < 90:
                features[SYMPTOM_VOCAB.index("very_low_spo2")] = 1.0
            if vital_signs.get("systolic_bp") and vital_signs["systolic_bp"] < 80:
                features[SYMPTOM_VOCAB.index("very_low_bp")] = 1.0
            if vital_signs.get("temperature_celsius") and vital_signs["temperature_celsius"] > 39.5:
                features[SYMPTOM_VOCAB.index("very_high_temp")] = 1.0
            if vital_signs.get("heart_rate") and vital_signs["heart_rate"] > 130:
                features[SYMPTOM_VOCAB.index("very_high_hr")] = 1.0

        return features.reshape(1, -1)

    def check_red_flags(self, symptoms: List[str]) -> bool:
        """
        Rule-based safety check.
        Returns True if any red flag pattern is detected.
        These ALWAYS override the ML model → forces EMERGENCY classification.
        """
        for pattern in RED_FLAG_PATTERNS:
            if all(flag in symptoms for flag in pattern):
                return True
        return False

    def classify(self, symptoms: List[str], age: int, sex: str,
                 vital_signs: Dict = None) -> Tuple[str, float, List[Dict]]:
        """
        Main classification method.

        Returns:
            urgency_level: "EMERGENCY" | "URGENT" | "NON_URGENT"
            confidence: float 0-1
            top_predictions: list of condition predictions
        """
        # Layer 1: Red flag override (safety critical)
        if self.check_red_flags(symptoms):
            return "EMERGENCY", 0.97, [{"condition": "Critical presentation — immediate evaluation required", "probability": 0.97}]

        # Layer 2: ML model
        if self.model is not None:
            try:
                features = self.symptoms_to_features(symptoms, age, sex, vital_signs)
                proba = self.model.predict_proba(features)[0]
                predicted_class = int(np.argmax(proba))
                confidence = float(np.max(proba))
                urgency = URGENCY_LABELS[predicted_class]

                top_predictions = [
                    {"condition": f"Urgency class: {URGENCY_LABELS[i]}", "probability": float(p)}
                    for i, p in sorted(enumerate(proba), key=lambda x: -x[1])
                ]
                return urgency, confidence, top_predictions

            except Exception as e:
                logger.error(f"ML classification failed: {e}. Falling back to rule-based.")

        # Layer 3: Rule-based fallback (when no model available)
        return self._rule_based_classify(symptoms, age, vital_signs)

    def _rule_based_classify(self, symptoms: List[str], age: int,
                              vital_signs: Dict = None) -> Tuple[str, float, List[Dict]]:
        """Simple rule-based fallback when ML model unavailable."""
        urgent_symptoms = {
            "high_fever", "severe_abdominal_pain", "rash_with_fever",
            "child_high_fever", "very_high_bp", "back_pain_severe",
            "pregnancy_pain", "reduced_fetal_movement"
        }
        if any(s in symptoms for s in urgent_symptoms):
            return "URGENT", 0.75, [{"condition": "Rule-based urgent flag", "probability": 0.75}]

        if age < 2 or age > 80:
            if len(symptoms) >= 2:
                return "URGENT", 0.65, [{"condition": "Vulnerable age group with multiple symptoms", "probability": 0.65}]

        return "NON_URGENT", 0.70, [{"condition": "No urgent/emergency flags detected", "probability": 0.70}]


# Singleton instance
_classifier_instance = None

def get_classifier(model_path: str = None) -> TriageClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        from app.core.config import get_settings
        path = model_path or get_settings().model_path
        _classifier_instance = TriageClassifier(path)
    return _classifier_instance
