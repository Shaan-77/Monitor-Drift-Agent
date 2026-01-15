"""
Main FastAPI application for Monitor/Drift Agent.

This module creates and configures the FastAPI application,
including all routers and middleware.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

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

# Import settings
try:
    from config.settings import get_settings
except ImportError:
    def get_settings():
        return None

# Create FastAPI app
app = FastAPI(
    title="Monitor/Drift Agent API",
    description="API for collecting and managing system metrics, alerts, and policies",
    version="1.0.0"
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


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        JSON response indicating API health status
    """
    return {
        "status": "healthy",
        "service": "Monitor/Drift Agent API",
        "version": "1.0.0"
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
            "policies": "/api/policies",
            "health": "/health"
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
