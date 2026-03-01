import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';

// Change this to your backend URL
const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
});

// ── OFFLINE QUEUE ──
const QUEUE_KEY = 'offline_triage_queue';

export async function getOfflineQueue() {
  const raw = await AsyncStorage.getItem(QUEUE_KEY);
  return raw ? JSON.parse(raw) : [];
}

export async function addToOfflineQueue(triageRequest) {
  const queue = await getOfflineQueue();
  queue.push({ ...triageRequest, queued_at: new Date().toISOString() });
  await AsyncStorage.setItem(QUEUE_KEY, JSON.stringify(queue));
}

export async function clearOfflineQueue() {
  await AsyncStorage.removeItem(QUEUE_KEY);
}

// ── LOCAL OFFLINE CLASSIFIER ──
// Runs on device when no internet — mirrors the Python logic
const RED_FLAGS = [
  ['chest_pain', 'arm_pain_left'],
  ['chest_pain', 'jaw_pain'],
  ['chest_pain', 'shortness_of_breath'],
  ['facial_drooping', 'arm_weakness_sudden'],
  ['seizure'],
  ['loss_of_consciousness'],
  ['coughing_blood'],
  ['vomiting_blood'],
  ['anaphylaxis_signs'],
  ['throat_swelling'],
  ['very_low_spo2'],
  ['altered_mental_status'],
  ['pregnancy_bleeding'],
];

const SYMPTOM_KEYWORDS = {
  chest_pain: ['chest pain', 'chest ache', 'chest pressure', 'seene mein dard'],
  arm_pain_left: ['left arm pain', 'left arm numbness', 'baaye haath'],
  jaw_pain: ['jaw pain', 'jaw ache'],
  shortness_of_breath: ['shortness of breath', 'difficulty breathing', 'breathless', 'saans'],
  seizure: ['seizure', 'convulsion', 'fit', 'dauraa'],
  loss_of_consciousness: ['unconscious', 'fainted', 'unresponsive', 'behosh'],
  facial_drooping: ['face drooping', 'facial droop'],
  arm_weakness_sudden: ['arm weakness', 'sudden weakness'],
  speech_difficulty: ["can't speak", 'slurred speech'],
  high_fever: ['high fever', 'very high temperature', '103', '104', '105', 'tez bukhar'],
  fever_mild: ['fever', 'temperature', 'bukhar'],
  rash_with_fever: ['rash with fever', 'fever and rash'],
  severe_abdominal_pain: ['severe abdominal pain', 'severe stomach pain', 'tez pet dard'],
  abdominal_pain_mild: ['stomach pain', 'abdominal pain', 'pet dard'],
  vomiting_blood: ['vomiting blood', 'blood in vomit'],
  coughing_blood: ['coughing blood', 'blood in cough'],
  anaphylaxis_signs: ['anaphylaxis', 'allergic reaction', 'tongue swelling'],
  throat_swelling: ["throat swelling", "can't swallow"],
  very_low_spo2: ['spo2 low', 'oxygen low'],
  altered_mental_status: ['not responding', 'barely conscious', 'drowsy', 'lethargi'],
  pregnancy_bleeding: ['bleeding during pregnancy', 'vaginal bleeding'],
  nausea: ['nausea', 'nauseous', 'ji machlana'],
  vomiting: ['vomiting', 'vomit', 'ulti'],
  headache_mild: ['headache', 'head pain', 'sir dard'],
  dizziness: ['dizzy', 'dizziness', 'chakkar'],
  sore_throat: ['sore throat', 'throat pain', 'gala dard'],
  back_pain_mild: ['back pain', 'backache', 'kamar dard'],
  fatigue: ['fatigue', 'weakness', 'tired', 'kamzori'],
};

export function extractSymptomsOffline(text) {
  const lower = text.toLowerCase();
  const found = [];
  for (const [token, keywords] of Object.entries(SYMPTOM_KEYWORDS)) {
    if (keywords.some(kw => lower.includes(kw))) {
      found.push(token);
    }
  }
  return found;
}

export function classifyOffline(symptoms, age, sex) {
  // Red flag check
  for (const pattern of RED_FLAGS) {
    if (pattern.every(s => symptoms.includes(s))) {
      return {
        urgency_level: 'EMERGENCY',
        confidence_score: 0.97,
        requires_human_review: false,
        recommended_action: '🔴 EMERGENCY: Transfer to hospital IMMEDIATELY. Do not wait.',
        clinical_explanation: 'Critical symptom pattern detected. This requires immediate emergency evaluation.',
        extracted_symptoms: symptoms,
        predicted_conditions: [{ condition: 'Critical presentation', probability: 0.97 }],
        rag_used: false,
        offline: true,
      };
    }
  }

  // Urgent patterns
  const urgentSymptoms = ['high_fever', 'rash_with_fever', 'severe_abdominal_pain', 'vomiting_blood'];
  if (urgentSymptoms.some(s => symptoms.includes(s))) {
    return {
      urgency_level: 'URGENT',
      confidence_score: 0.80,
      requires_human_review: false,
      recommended_action: '🟡 URGENT: Patient must see a doctor within 2-4 hours.',
      clinical_explanation: 'Symptoms indicate a condition requiring prompt medical evaluation.',
      extracted_symptoms: symptoms,
      predicted_conditions: [{ condition: 'Urgent medical condition', probability: 0.80 }],
      rag_used: false,
      offline: true,
    };
  }

  return {
    urgency_level: 'NON_URGENT',
    confidence_score: 0.70,
    requires_human_review: false,
    recommended_action: '🟢 NON-URGENT: Manage at clinic or provide home care.',
    clinical_explanation: 'No immediate danger signs detected. Monitor and advise follow-up if symptoms worsen.',
    extracted_symptoms: symptoms,
    predicted_conditions: [{ condition: 'Non-urgent presentation', probability: 0.70 }],
    rag_used: false,
    offline: true,
  };
}

// ── MAIN TRIAGE FUNCTION ──
export async function runTriage(request) {
  const netState = await NetInfo.fetch();
  const isConnected = netState.isConnected && netState.isInternetReachable;

  if (isConnected) {
    try {
      const response = await api.post('/triage/assess', request);
      return { ...response.data, offline: false };
    } catch (error) {
      console.warn('API failed, falling back to offline classifier:', error.message);
    }
  }

  // Offline classification
  const symptoms = extractSymptomsOffline(request.chief_complaint);
  const result = classifyOffline(symptoms, request.patient_age, request.patient_sex);

  // Queue for sync when online
  await addToOfflineQueue(request);

  return result;
}

// ── SYNC OFFLINE QUEUE ──
export async function syncOfflineQueue(deviceId) {
  const queue = await getOfflineQueue();
  if (queue.length === 0) return { synced: 0 };

  try {
    const response = await api.post('/triage/sync', { cases: queue, device_id: deviceId });
    await clearOfflineQueue();
    return response.data;
  } catch (error) {
    console.error('Sync failed:', error.message);
    throw error;
  }
}

export async function getCases(params = {}) {
  const response = await api.get('/cases/', { params });
  return response.data;
}

export async function getStats(clinicId) {
  const response = await api.get('/cases/stats/summary', { params: { clinic_id: clinicId } });
  return response.data;
}

export async function doctorOverride(caseId, override) {
  const response = await api.patch(`/cases/${caseId}/override`, override);
  return response.data;
}

export default api;
