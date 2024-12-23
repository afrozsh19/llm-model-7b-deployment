import os
import torch
import traceback
import logging

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


CONFIG = {
    "model_name": os.getenv("MODEL_NAME", ""),
    "model_path": os.getenv("MODEL_DIR", "../models/"),
    "debug": os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG",
    "log_level": os.getenv("LOG_LEVEL", "INFO").upper(),
    "use_bits_and_bytes": os.getenv("BITS_AND_BYTES", "false").lower() == "true"
}


def get_device_in_use():
    """
    Returns the current devide that the code is running.
    """
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_error_response(e, include_traceback: bool):
    """
    Returns a formatted error JSON response
    """
    response = {"error": True, "message": str(e)}
    if include_traceback:
        response["traceback"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    
    return response


async def validation_error_response(request: Request, e: RequestValidationError):
    """
    Returns a formatted error JSON response for a request validation error
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=get_error_response(e, CONFIG["debug"])
    )


async def python_error_response(request: Request, e: Exception):
    """
    Returns a formatted error JSON response for a python generic code error
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=get_error_response(e, CONFIG["debug"])
    )


def log_config():
    """Configure logging for the module."""

    log_level = CONFIG["log_level"]

    valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    if log_level not in valid_log_levels:
        log_level = "INFO"

    logging.basicConfig(
        level=log_level, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging level set to: {log_level}")
    
    return logger