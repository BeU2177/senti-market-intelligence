"""FastAPI main application entry point."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.health import router as health_router
from api.routes.market import router as market_router
from api.routes.predictions import router as prediction_router
from api.routes.sentiment import router as sentiment_router
from api.routes.analysis import router as analysis_router
from utils.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Senti Market Intelligence API",
    description="Production Multi-Market Financial Intelligence, PyTorch Deep Learning & Grounded AI Analysis API",
    version="v1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(health_router)
app.include_router(market_router)
app.include_router(prediction_router)
app.include_router(sentiment_router)
app.include_router(analysis_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler preventing raw stack traces in production responses."""
    logger.error(f"Global exception on {request.method} {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Production Server Error", "detail": str(exc)},
    )
