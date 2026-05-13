from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.campaign_routes import router as campaign_router
from app.api.health_routes import router as health_router
from app.api.rulebook_routes import router as rulebook_router


app = FastAPI(
    title="Campaign Recommendation MVP",
    version="1.0.0",
    description="FastAPI backend for LangGraph-controlled campaign planning demo.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(rulebook_router)
app.include_router(campaign_router)
