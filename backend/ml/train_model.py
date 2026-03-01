"""
Train the XGBoost urgency classifier.

Run this BEFORE starting the API:
    python ml/train_model.py

Data sources (in order of preference):
1. Your own labeled clinical data (best — use data/label_tool.py)
2. data/sample_dataset.csv (synthetic — included in repo for demo)
3. DDXPlus dataset (download separately — best open source option)

Target: 80%+ accuracy on validation set before deployment.
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.services.ml_classifier import SYMPTOM_VOCAB, URGENCY_REVERSE
from app.services.symptom_extractor import extract_symptoms

MODEL_OUTPUT_PATH = "ml/models/triage_classifier.joblib"


def load_or_generate_training_data():
    """
    Load training data or generate synthetic data for demo purposes.
    Replace this with real clinical data for production.
    """
    data_path = "data/sample_dataset.csv"

    if os.path.exists(data_path):
        logger.info(f"Loading training data from {data_path}")
        df = pd.read_csv(data_path)
        return df

    logger.warning("No training data found. Generating synthetic demo dataset.")
    logger.warning("REPLACE WITH REAL CLINICAL DATA FOR PRODUCTION USE.")
    return generate_synthetic_dataset()


def generate_synthetic_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generates a synthetic training dataset.
    Symptom→urgency mappings based on clinical triage guidelines.

    THIS IS NOT A SUBSTITUTE FOR REAL CLINICAL DATA.
    Use this only for demo/development purposes.
    """
    np.random.seed(42)
    records = []

    # Emergency cases (~25%)
    emergency_patterns = [
        (["chest_pain", "arm_pain_left", "shortness_of_breath"], "EMERGENCY"),
        (["chest_pain", "jaw_pain", "palpitations"], "EMERGENCY"),
        (["facial_drooping", "arm_weakness_sudden", "speech_difficulty"], "EMERGENCY"),
        (["seizure"], "EMERGENCY"),
        (["loss_of_consciousness"], "EMERGENCY"),
        (["coughing_blood"], "EMERGENCY"),
        (["vomiting_blood"], "EMERGENCY"),
        (["anaphylaxis_signs", "throat_swelling"], "EMERGENCY"),
        (["very_low_spo2", "shortness_of_breath"], "EMERGENCY"),
        (["altered_mental_status", "high_fever"], "EMERGENCY"),
        (["head_injury", "loss_of_consciousness_trauma"], "EMERGENCY"),
        (["pregnancy_bleeding"], "EMERGENCY"),
    ]

    # Urgent cases (~35%)
    urgent_patterns = [
        (["high_fever", "rash_with_fever"], "URGENT"),
        (["high_fever", "chills", "rigors"], "URGENT"),
        (["severe_abdominal_pain", "vomiting"], "URGENT"),
        (["child_high_fever", "rash_mild"], "URGENT"),
        (["rash_petechial", "fever_mild"], "URGENT"),
        (["very_high_bp", "headache_mild"], "URGENT"),
        (["pregnancy_pain", "reduced_fetal_movement"], "URGENT"),
        (["dizziness", "very_high_hr"], "URGENT"),
        (["back_pain_severe", "decreased_urine"], "URGENT"),
        (["confusion", "fever_mild"], "URGENT"),
    ]

    # Non-urgent cases (~40%)
    non_urgent_patterns = [
        (["fever_mild", "sore_throat"], "NON_URGENT"),
        (["headache_mild", "fatigue"], "NON_URGENT"),
        (["nausea", "vomiting"], "NON_URGENT"),
        (["back_pain_mild", "muscle_pain"], "NON_URGENT"),
        (["diarrhea", "abdominal_pain_mild"], "NON_URGENT"),
        (["rash_mild"], "NON_URGENT"),
        (["sore_throat", "fever_mild"], "NON_URGENT"),
        (["joint_pain", "fatigue"], "NON_URGENT"),
        (["urinary_symptoms"], "NON_URGENT"),
        (["ear_pain", "fever_mild"], "NON_URGENT"),
        (["dizziness", "headache_mild"], "NON_URGENT"),
    ]

    all_patterns = (
        emergency_patterns * 10 +  # Oversample emergency
        urgent_patterns * 15 +
        non_urgent_patterns * 20
    )

    for symptoms, urgency in all_patterns:
        for _ in range(3):
            age = np.random.randint(1, 85)
            sex = np.random.choice(["male", "female"])
            # Add random additional symptoms (noise)
            extra = np.random.choice(SYMPTOM_VOCAB, size=np.random.randint(0, 3), replace=False).tolist()
            all_syms = list(set(symptoms + extra))

            # Build feature vector
            features = np.zeros(len(SYMPTOM_VOCAB) + 3)
            for sym in all_syms:
                if sym in SYMPTOM_VOCAB:
                    features[SYMPTOM_VOCAB.index(sym)] = 1.0
            features[len(SYMPTOM_VOCAB)] = age / 100.0
            features[len(SYMPTOM_VOCAB) + 1] = 1.0 if sex == "male" else 0.0
            features[len(SYMPTOM_VOCAB) + 2] = 1.0 if age < 5 or age > 70 else 0.0

            records.append({
                **{f"sym_{i}": features[i] for i in range(len(SYMPTOM_VOCAB))},
                "age_norm": features[len(SYMPTOM_VOCAB)],
                "sex_male": features[len(SYMPTOM_VOCAB) + 1],
                "age_extreme": features[len(SYMPTOM_VOCAB) + 2],
                "urgency": urgency
            })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Generated {len(df)} synthetic training samples")
    logger.info(df["urgency"].value_counts().to_string())
    return df


def train():
    os.makedirs("ml/models", exist_ok=True)

    # Load data
    df = load_or_generate_training_data()
    logger.info(f"Dataset: {len(df)} samples, {df['urgency'].value_counts().to_dict()}")

    # Features / Labels
    feature_cols = [c for c in df.columns if c != "urgency"]
    X = df[feature_cols].values
    y = df["urgency"].map(URGENCY_REVERSE).values

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class weights (emergency is rare but critical)
    class_weights = {0: 1.0, 1: 1.2, 2: 3.0}  # Weight emergency 3x
    sample_weights = np.array([class_weights[label] for label in y_train])

    # XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    logger.info(f"\n{'='*50}")
    logger.info(f"VALIDATION ACCURACY: {acc:.3f} ({acc*100:.1f}%)")
    logger.info(f"{'='*50}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_val, y_pred, target_names=["NON_URGENT", "URGENT", "EMERGENCY"]))

    if acc < 0.80:
        logger.warning("⚠️  Accuracy below 80%. DO NOT deploy. Get better training data.")
    else:
        logger.info("✅ Accuracy acceptable for demo. Get real clinical data before production.")

    # Save
    joblib.dump(model, MODEL_OUTPUT_PATH)
    logger.info(f"Model saved to {MODEL_OUTPUT_PATH}")
    return model, acc


if __name__ == "__main__":
    train()
