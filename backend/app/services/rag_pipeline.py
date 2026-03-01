"""
RAG Pipeline: Retrieves relevant clinical protocols and generates explanations.

Flow:
1. Symptom list + urgency → query vector DB (ChromaDB)
2. Retrieve top-k relevant clinical guidelines
3. Feed to LLM → generate plain-language clinical explanation for nurse
4. Fallback: if no LLM available → return rule-based explanation

IMPORTANT: RAG is ONLY for explanation generation.
The urgency classification decision comes from the ML classifier, NOT the LLM.
"""

import logging
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# ── RULE-BASED EXPLANATION FALLBACK ──
# Used when offline or no LLM API key available.
# Covers the most common triage presentations.

RULE_BASED_EXPLANATIONS = {
    "EMERGENCY": {
        "default": "This symptom combination indicates a potentially life-threatening emergency. The patient requires IMMEDIATE medical evaluation. Do not delay.",
        "chest_pain+arm_pain_left": "Symptoms consistent with possible acute cardiac event (myocardial infarction). Immediate transfer to emergency facility required. Keep patient calm and still. Do not give food or water.",
        "facial_drooping+arm_weakness_sudden": "Symptom pattern matches acute stroke (FAST criteria positive). Time-critical: brain tissue loss begins immediately. Transfer within 60 minutes for best outcomes.",
        "seizure": "Active or recent seizure requires immediate evaluation to identify cause (febrile, epileptic, metabolic, structural). Protect patient from injury. Document seizure duration.",
        "loss_of_consciousness": "Loss of consciousness requires immediate assessment of airway, breathing, circulation. Do not leave patient unattended.",
        "anaphylaxis_signs": "Signs of anaphylaxis. Life-threatening allergic reaction. Epinephrine if available. Immediate transfer.",
        "very_low_spo2": "Oxygen saturation critically low. Patient at immediate risk of hypoxic organ damage. Ensure airway patency and arrange emergency oxygen.",
    },
    "URGENT": {
        "default": "This patient requires medical evaluation within 2-4 hours. Condition is not immediately life-threatening but delays may worsen outcomes.",
        "high_fever": "High fever requires evaluation to identify source (bacterial infection, malaria, dengue). Check for danger signs (stiff neck, rash, confusion). Blood tests if available.",
        "rash_with_fever": "Fever with rash requires urgent evaluation. In Indian context, consider dengue (check for petechiae), chikungunya, typhoid. Avoid NSAIDs if dengue suspected.",
        "severe_abdominal_pain": "Severe abdominal pain may indicate surgical emergency (appendicitis, perforation) or other serious condition. Needs physician evaluation and potentially imaging.",
        "pregnancy": "Symptoms in pregnancy require prompt obstetric evaluation. Any bleeding, pain, or reduced fetal movement is urgent.",
    },
    "NON_URGENT": {
        "default": "Current symptoms do not indicate an emergency. Patient can be managed at the clinic or at home with appropriate guidance. Re-evaluate if symptoms worsen.",
        "fever_mild": "Mild fever: Ensure adequate hydration. Paracetamol for comfort. Return if fever persists >3 days, rises above 39°C, or new symptoms develop.",
        "sore_throat": "Sore throat: Usually viral. Supportive care. Consider strep if severe. Return if unable to swallow or fever develops.",
        "back_pain_mild": "Mild back pain: Rest, analgesia. Return if pain radiates to legs, bowel/bladder symptoms develop, or no improvement in 1 week.",
    }
}

def get_rule_based_explanation(urgency: str, symptoms: List[str], age: int, sex: str) -> str:
    """Generate rule-based explanation without LLM."""
    explanations = RULE_BASED_EXPLANATIONS.get(urgency, {})

    # Find most specific matching explanation
    for key, explanation in explanations.items():
        if key == "default":
            continue
        key_symptoms = key.split("+")
        if all(s in symptoms for s in key_symptoms):
            base = explanation
            break
    else:
        base = explanations.get("default", "Assessment complete. Follow urgency level guidance.")

    # Add age-specific notes
    if age < 5:
        base += f" Note: Patient is a young child ({age} years). Lower threshold for escalation in pediatric patients."
    elif age > 70:
        base += f" Note: Patient is elderly ({age} years). Elderly patients may present atypically — consider escalation if in doubt."

    return base

def get_recommended_action(urgency: str, symptoms: List[str]) -> str:
    """Return concrete nurse action based on urgency."""
    actions = {
        "EMERGENCY": "🔴 EMERGENCY: Transfer to hospital IMMEDIATELY. Do not wait. Alert receiving facility if possible. Monitor vitals continuously during transport.",
        "URGENT": "🟡 URGENT: Patient must see a doctor within 2-4 hours. Do not send home. Arrange referral or keep under observation. Recheck vitals every 30 minutes.",
        "NON_URGENT": "🟢 NON-URGENT: Manage at clinic or provide home care instructions. Schedule follow-up within 24-48 hours. Advise patient to return immediately if symptoms worsen.",
    }
    return actions.get(urgency, "Assessment complete. Use clinical judgment.")


class RAGPipeline:
    """
    RAG pipeline using ChromaDB + LangChain.
    Gracefully degrades to rule-based when offline or no API key.
    """

    def __init__(self, persist_dir: str, openai_api_key: str = ""):
        self.persist_dir = persist_dir
        self.openai_api_key = openai_api_key
        self.vectorstore = None
        self.llm = None
        self.chain = None
        self.is_available = False
        self._initialize()

    def _initialize(self):
        """Initialize RAG components. Silently fails if dependencies unavailable."""
        try:
            import chromadb
            from langchain_community.vectorstores import Chroma
            from langchain_community.embeddings import HuggingFaceEmbeddings

            # Use lightweight local embeddings (no API key needed)
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self.vectorstore = Chroma(
                collection_name="clinical_guidelines",
                embedding_function=embeddings,
                persist_directory=self.persist_dir
            )

            # Try to set up LLM (optional — needs API key)
            if self.openai_api_key and self.openai_api_key.startswith("sk-"):
                from langchain_openai import ChatOpenAI
                from langchain.chains import RetrievalQA
                from langchain.prompts import PromptTemplate

                self.llm = ChatOpenAI(
                    model="gpt-4o",
                    api_key=self.openai_api_key,
                    temperature=0.1,
                    max_tokens=500
                )

                prompt = PromptTemplate(
                    template="""You are a clinical decision support system for rural Indian healthcare workers.
Based on the retrieved clinical guidelines below, provide a clear, concise explanation for a nurse with basic training.

Clinical Context:
{context}

Patient Assessment:
Urgency Level: {urgency}
Symptoms Identified: {symptoms}
Patient: {age} year old {sex}

Provide:
1. Why this urgency level was assigned (1-2 sentences)
2. What the nurse should do right now (specific actions)
3. Warning signs to watch for

Keep your response under 150 words. Use simple language. No medical jargon without explanation.

Response:""",
                    input_variables=["context", "urgency", "symptoms", "age", "sex"]
                )
                self.is_available = True
                logger.info("RAG pipeline initialized with LLM support.")
            else:
                self.is_available = True  # Vector search available, no LLM
                logger.info("RAG pipeline initialized in retrieval-only mode (no LLM key).")

        except ImportError as e:
            logger.warning(f"RAG dependencies not available: {e}. Using rule-based fallback.")
        except Exception as e:
            logger.error(f"RAG initialization failed: {e}. Using rule-based fallback.")

    def generate_explanation(
        self,
        urgency: str,
        symptoms: List[str],
        age: int,
        sex: str,
        chief_complaint: str
    ) -> Tuple[str, bool]:
        """
        Generate clinical explanation.
        Returns: (explanation_text, rag_was_used)
        """
        # Always try RAG first
        if self.is_available and self.vectorstore:
            try:
                # Retrieve relevant clinical guidelines
                query = f"{' '.join(symptoms[:5])} {urgency} triage protocol"
                docs = self.vectorstore.similarity_search(query, k=3)

                if docs and self.llm:
                    context = "\n\n".join([d.page_content for d in docs])
                    prompt_text = f"""Clinical Context:\n{context}\n\nPatient: {age}yo {sex}
Urgency: {urgency}\nSymptoms: {', '.join(symptoms)}\nComplaint: {chief_complaint}
Explain urgency and nurse actions in 100 words max, simple language:"""

                    response = self.llm.invoke(prompt_text)
                    return response.content, True

                elif docs:
                    # Retrieval without LLM — return top guideline snippet
                    snippet = docs[0].page_content[:400] if docs else ""
                    base = get_rule_based_explanation(urgency, symptoms, age, sex)
                    return f"{base}\n\nRelevant guideline: {snippet}", True

            except Exception as e:
                logger.error(f"RAG generation failed: {e}. Falling back to rules.")

        # Rule-based fallback
        explanation = get_rule_based_explanation(urgency, symptoms, age, sex)
        return explanation, False

    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Add clinical guidelines to the vector store."""
        if not self.vectorstore:
            logger.error("Vector store not initialized. Cannot add documents.")
            return

        from langchain.schema import Document
        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(texts, metadatas or [{}] * len(texts))
        ]
        self.vectorstore.add_documents(docs)
        logger.info(f"Added {len(docs)} documents to knowledge base.")


# Singleton
_rag_instance = None

def get_rag_pipeline() -> RAGPipeline:
    global _rag_instance
    if _rag_instance is None:
        from app.core.config import get_settings
        settings = get_settings()
        _rag_instance = RAGPipeline(
            persist_dir=settings.chroma_persist_dir,
            openai_api_key=settings.openai_api_key
        )
    return _rag_instance
