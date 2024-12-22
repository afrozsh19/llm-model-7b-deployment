import os
import torch
import traceback

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


CONFIG = {
    "model_name": os.getenv("MODEL_NAME", "NickyNicky/experimental-Mistral-1b-V00"),
    "model_path": os.getenv("MODEL_DIR", "../models/mistral-1b/"),
    "debug": os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG",
    "log_level": os.getenv("LOG_LEVEL", "INFO").upper(),
    "num_gpu_instances": os.getenv("NUM_GPU_INSTANCES", 1),
    "num_cpu_instances": os.getenv("NUM_CPU_INSTANCES", 1),
    "num_replicas": os.getenv("NUM_REPLICAS", 1),
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
