"""
Seeds the ChromaDB vector store with clinical triage guidelines.

Run once before starting the server:
    python data/seed_guidelines.py

Sources used:
- WHO Emergency Triage Assessment and Treatment (ETAT) guidelines
- Indian Academy of Pediatrics emergency guidelines
- National Health Mission India clinical protocols
- Manchester Triage System principles
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clinical guidelines text — manually curated from public WHO/Indian health docs
CLINICAL_GUIDELINES = [
    {
        "title": "WHO ETAT — Emergency Signs",
        "content": """Emergency signs requiring IMMEDIATE action:
Airway: Obstructed or absent breathing. Action: Open airway, position patient, give oxygen.
Breathing: Severe respiratory distress, central cyanosis, SpO2 < 90%. Action: Oxygen, emergency referral.
Circulation: Cold extremities with capillary refill > 3 seconds, weak/fast pulse, very low BP. Action: IV access, fluid if indicated, emergency transfer.
Consciousness: Unconscious, not responding, AVPU score P or U. Action: Protect airway, emergency referral immediately.
Convulsions (active): Give diazepam rectally, protect airway, check glucose. Emergency referral.
Dehydration (severe): Sunken eyes, skin pinch very slow, unable to drink. IV rehydration required.""",
        "urgency": "EMERGENCY",
        "source": "WHO ETAT Guidelines"
    },
    {
        "title": "Acute Coronary Syndrome Recognition",
        "content": """Suspected heart attack (myocardial infarction) — EMERGENCY:
Classic presentation: Central crushing chest pain, radiating to left arm or jaw, sweating, shortness of breath, nausea.
Atypical presentations (common in women, elderly, diabetics): Jaw/neck/back pain only, breathlessness only, sudden fatigue, indigestion-like discomfort.
Action: DO NOT give food or water. Keep patient still and calm. Transfer IMMEDIATELY to hospital with ECG capability within 90 minutes of symptom onset (golden window). Aspirin 300mg chewable if not allergic and no contraindications.
DO NOT delay transfer to perform investigations at clinic.""",
        "urgency": "EMERGENCY",
        "source": "Indian ACS Management Guidelines"
    },
    {
        "title": "Stroke Recognition — FAST Protocol",
        "content": """Stroke — EMERGENCY. Time = Brain. Every minute counts.
FAST criteria:
F — Face drooping (ask to smile — is it uneven?)
A — Arm weakness (ask to raise both arms — does one drift down?)
S — Speech difficulty (ask to repeat a phrase — is it slurred or strange?)
T — Time to call emergency services IMMEDIATELY
Additional signs: Sudden vision change, sudden severe headache, sudden loss of balance.
Action: Emergency transfer within 60 minutes of onset. Record exact time symptoms started. Do NOT give food, water, or medications unless directed by physician.
Clot-busting treatment (thrombolysis) can only be given within 4.5 hours — every minute of delay worsens outcome.""",
        "urgency": "EMERGENCY",
        "source": "Stroke Foundation Guidelines"
    },
    {
        "title": "Dengue — Indian Context Triage",
        "content": """Dengue Fever — common in India, monsoon season peak:
Phase 1 — Febrile (Days 1-3): High fever (39-40°C), severe headache, eye pain, muscle/joint pain. Manage with paracetamol ONLY. NO aspirin, NO ibuprofen, NO NSAIDs.
Phase 2 — Critical (Days 4-5): Fever may drop, but this is DANGEROUS. Watch for: warning signs below.
WARNING SIGNS requiring URGENT/EMERGENCY escalation:
- Abdominal pain or tenderness
- Persistent vomiting
- Bleeding from any site (gum, nose, petechiae/purple spots on skin)
- Lethargy, restlessness, altered consciousness
- Rapid breathing, cold clammy skin
Non-blanching rash (petechiae/purpura): EMERGENCY — may indicate dengue hemorrhagic fever.
Platelet count monitoring required from Day 3.""",
        "urgency": "URGENT",
        "source": "NVBDCP India — Dengue Clinical Management"
    },
    {
        "title": "Fever with Rash — Differential Triage",
        "content": """Fever with rash — urgency depends on rash type:
EMERGENCY — Non-blanching purple/red spots (petechiae/purpura): May indicate meningococcemia or dengue hemorrhagic fever. Transfer immediately.
EMERGENCY — Widespread blistering rash with fever: Stevens-Johnson syndrome possible.
URGENT — Maculopapular rash with high fever in child: Consider measles, rubella, dengue. Evaluate promptly.
URGENT — Diffuse rash after insect bite with fever: Consider chikungunya, Rickettsial disease in India.
NON-URGENT — Mild rash without fever, good general condition: Likely viral exanthem. Monitor.
Blanch test: Press rash firmly — if it disappears under pressure (blanches), less urgent. If it stays (non-blanching), EMERGENCY.""",
        "urgency": "URGENT",
        "source": "Pediatric Infectious Disease Guidelines India"
    },
    {
        "title": "Pediatric Emergency Signs — Under 5 Years",
        "content": """Children under 5 deteriorate FASTER than adults. Lower threshold for escalation.
IMMEDIATE danger signs (EMERGENCY):
- Cannot drink or breastfeed
- Vomits everything
- Has had convulsions in this illness
- Lethargic or unconscious
- Breathing very fast or difficulty breathing
- Fever in infant under 3 months (any fever = emergency in <3 months)
- Bulging fontanelle (soft spot on top of head)
URGENT signs:
- High fever (>38.5°C in under 5 without source)
- Fast breathing without severe distress
- Moderate dehydration
- Unable to eat/drink adequately
DO NOT wait to observe a sick child. Err on the side of escalation.""",
        "urgency": "EMERGENCY",
        "source": "IMNCI India — Integrated Management of Neonatal and Childhood Illness"
    },
    {
        "title": "Obstetric Emergencies",
        "content": """Pregnancy-related emergencies — all require URGENT or EMERGENCY evaluation:
EMERGENCY:
- Vaginal bleeding at any stage of pregnancy
- Severe abdominal pain during pregnancy
- Seizures in pregnancy (eclampsia)
- Signs of shock: cold pale skin, rapid weak pulse, reduced consciousness
- No fetal movement after 28 weeks (reduced fetal movement)
- Cord prolapse (cord visible at vagina)
URGENT:
- Headache with visual changes in pregnancy (pre-eclampsia warning)
- Reduced fetal movement (less than 10 kicks in 12 hours)
- Preterm labor signs before 37 weeks
- Premature rupture of membranes (water breaking before labor)
Never delay transfer for an obstetric emergency. Maternal and fetal lives both at risk.""",
        "urgency": "EMERGENCY",
        "source": "MoHFW India — Emergency Obstetric Care Guidelines"
    },
    {
        "title": "Non-Urgent Fever Management",
        "content": """Mild fever (37.5-38.5°C) without danger signs — NON-URGENT management:
For children: Paracetamol 15mg/kg every 6 hours if uncomfortable. Do NOT routinely combine paracetamol + ibuprofen. Ensure adequate fluids. Sponge bath for comfort — tepid water only.
For adults: Paracetamol 500-1000mg every 6 hours. Rest, oral fluids. NSAIDs if no contraindications.
Advise RETURN immediately if:
- Fever rises above 39°C
- Fever persists beyond 3 days
- New symptoms develop (rash, confusion, severe pain)
- Patient appears worse or stops eating/drinking
- Child becomes very lethargic or inconsolable
Do not prescribe antibiotics for viral fever without evidence of bacterial infection.""",
        "urgency": "NON_URGENT",
        "source": "WHO Essential Medicines Guidelines"
    },
    {
        "title": "Anaphylaxis Recognition and Response",
        "content": """Anaphylaxis — LIFE-THREATENING EMERGENCY. Act within minutes.
Recognition: Sudden onset of TWO OR MORE:
- Skin/mucosal: hives, flushing, swollen lips/tongue/face
- Respiratory: wheeze, stridor, difficulty breathing, reduced SpO2
- Cardiovascular: rapid weak pulse, low BP, collapse
- GI: sudden vomiting/diarrhea after exposure to potential allergen
Common triggers in India: Medications (penicillin, NSAIDs), food (nuts, shellfish), insect stings.
Action: Position patient flat (legs elevated). Epinephrine (adrenaline) IM 0.3mg adult / 0.15mg child into outer thigh — first-line treatment. Transfer IMMEDIATELY. Maintain airway. IV access en route.
Do NOT give antihistamine as first treatment — it is NOT adequate for anaphylaxis.""",
        "urgency": "EMERGENCY",
        "source": "World Allergy Organization Anaphylaxis Guidelines"
    },
    {
        "title": "Abdominal Pain — Triage Guide",
        "content": """Abdominal pain urgency classification:
EMERGENCY:
- Severe sudden-onset abdominal pain (especially with rigid abdomen)
- Abdominal pain with signs of shock
- Suspected aortic aneurysm (pulsating abdominal mass, tearing pain)
- Appendicitis signs with peritonism (rebound tenderness, guarding)
URGENT:
- Right lower quadrant pain, especially with fever and loss of appetite (appendicitis possible)
- Severe pain with fever
- Abdominal pain in diabetic patient (may be silent MI or DKA)
- Any abdominal pain in pregnancy
NON-URGENT:
- Mild cramping without fever or systemic signs
- Constipation with mild discomfort
- Gastroenteritis with mild pain and hydrated patient
Beware: Elderly and diabetic patients may have serious pathology with mild pain.""",
        "urgency": "URGENT",
        "source": "Surgical Emergency Triage Guidelines"
    },
]


def seed_knowledge_base():
    """Seed all guidelines into ChromaDB."""
    logger.info("Seeding clinical knowledge base...")

    try:
        from app.services.rag_pipeline import get_rag_pipeline
        rag = get_rag_pipeline()

        if not rag.vectorstore:
            logger.error("Vector store not initialized. Ensure ChromaDB is installed.")
            return

        texts = [g["content"] for g in CLINICAL_GUIDELINES]
        metadatas = [
            {"title": g["title"], "urgency": g["urgency"], "source": g["source"]}
            for g in CLINICAL_GUIDELINES
        ]

        rag.add_documents(texts, metadatas)
        logger.info(f"✅ Seeded {len(CLINICAL_GUIDELINES)} clinical guidelines into knowledge base.")
        logger.info("Knowledge base ready for RAG queries.")

    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        raise


if __name__ == "__main__":
    seed_knowledge_base()
