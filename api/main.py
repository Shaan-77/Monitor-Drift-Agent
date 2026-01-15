"""
Main FastAPI application for Monitor/Drift Agent.

This module creates and configures the FastAPI application,
including all routers and middleware.
"""

# Initialize error capture early to catch all terminal errors
try:
    from utils.error_capture import initialize_error_capture
    import os
    _error_log_file = os.getenv("ERROR_LOG_FILE", "errors.log")
    initialize_error_capture(_error_log_file)
except Exception:
    # If error capture fails, continue without it
    pass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Try to import slowapi for rate limiting error handling
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False
    RateLimitExceeded = Exception

# Import routers
from api.metrics import router as metrics_router
from api.policies import router as policies_router
from api.alerts import router as alerts_router

# Import settings
try:
    from config.settings import get_settings
except ImportError:
    def get_settings():
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    
    Handles startup and shutdown events:
    - Startup: Initialize database schemas
    - Shutdown: Cleanup (if needed)
    """
    # Startup: Initialize database
    try:
        from data_collection.database import initialize_database
        initialized = initialize_database()
        if initialized:
            print("✓ Database initialized successfully - all schemas created")
        else:
            print("⚠ Warning: Database initialization failed or was skipped")
            print("  The API will start, but database features may not work.")
            print("  To fix: Set DATABASE_URL environment variable with correct credentials.")
            print("  Example: export DATABASE_URL=postgresql://username:password@localhost:5432/monitor_drift")
    except RuntimeError as e:
        # psycopg2 not available or other critical error
        print(f"✗ Error during database initialization: {e}")
        print("  The API will start, but database features may not work.")
        if "psycopg2" in str(e).lower():
            print("  To fix: Install psycopg2-binary: pip install psycopg2-binary")
    except Exception as e:
        # Other errors (connection failures, etc.)
        print(f"✗ Error during database initialization: {e}")
        print("  The API will start, but database features may not work.")
        print("  To fix: Check your DATABASE_URL environment variable and ensure PostgreSQL is running.")
    
    yield  # Application runs here
    
    # Shutdown: Cleanup if needed (none required for now)
    # Connection cleanup is handled by individual functions


# Create FastAPI app
app = FastAPI(
    title="Monitor/Drift Agent API",
    description="API for collecting and managing system metrics, alerts, and policies",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register rate limit error handler if slowapi is available
if SLOWAPI_AVAILABLE:
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        """
        Handle rate limit exceeded errors.
        
        Returns:
            JSON response with 429 status code
        """
        response = JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": str(exc),
                "detail": "Too many requests. Please try again later."
            }
        )
        try:
            response = _rate_limit_exceeded_handler(request, exc, response)
        except Exception:
            pass  # If handler fails, use our default response
        return response

# Initialize rate limiter and attach to app if available
if SLOWAPI_AVAILABLE:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from config.settings import get_settings as get_app_settings
    
    app_settings = get_app_settings()
    if app_settings and app_settings.api_enable_rate_limit:
        limiter = Limiter(
            app=app,
            key_func=get_remote_address
        )
        # Attach limiter to metrics router so it can be used in decorators
        metrics_router.state.limiter = limiter
        # Also make it available globally for the metrics module
        import api.metrics as metrics_module
        metrics_module.limiter = limiter

# Include routers
app.include_router(metrics_router)
app.include_router(policies_router)
app.include_router(alerts_router)


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        JSON response indicating API health status and database connectivity
    """
    # Check database connection status
    database_status = "unknown"
    try:
        from data_collection.database import connect_to_db, PSYCOPG2_AVAILABLE
        
        if not PSYCOPG2_AVAILABLE:
            database_status = "unavailable (psycopg2 not installed)"
        else:
            conn = connect_to_db()
            if conn:
                conn.close()
                database_status = "connected"
            else:
                database_status = "disconnected"
    except Exception as e:
        database_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "service": "Monitor/Drift Agent API",
        "version": "1.0.0",
        "database": database_status
    }


@app.get("/")
async def root():
    """
    Root endpoint providing API information.
    
    Returns:
        JSON response with API information
    """
    return {
        "name": "Monitor/Drift Agent API",
        "version": "1.0.0",
        "endpoints": {
            "metrics": "/api/metrics",
            "metrics_collect": "/api/metrics/collect",
            "metrics_stored": "/api/metrics/cloud-and-server",
            "alerts": "/api/alerts",
            "policies": "/api/policies",
            "health": "/health",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get settings
    settings = get_settings()
    if settings is None:
        # Default values if settings not available
        host = "0.0.0.0"
        port = 8000
        debug = False
    else:
        host = settings.api_host
        port = settings.api_port
        debug = settings.api_debug
    
    # Run the application
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )
