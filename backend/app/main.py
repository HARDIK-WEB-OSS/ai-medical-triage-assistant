from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.core.config import get_settings
from app.core.database import create_tables
from app.api import triage, cases

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="AI Medical Triage Assistant API",
    description="""
    Low-Cost Edge AI for Rural Healthcare Triage.
    
    ## Key Endpoints
    - **POST /triage/assess** — Run AI triage on patient symptoms
    - **POST /triage/sync** — Sync offline cases from device
    - **GET /cases/** — List all triage cases  
    - **PATCH /cases/{id}/override** — Doctor override for AI decision
    - **GET /cases/stats/summary** — Dashboard statistics
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS — allow mobile app and web demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(triage.router, prefix="/api/v1")
app.include_router(cases.router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    logger.info("Starting AI Medical Triage Assistant API...")
    try:
        create_tables()
        logger.info("Database tables ready.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    logger.info("API ready. Docs at /docs")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AI Medical Triage Assistant",
        "version": "1.0.0"
    }


@app.get("/")
def root():
    return {
        "message": "AI Medical Triage Assistant API",
        "docs": "/docs",
        "health": "/health"
    }
