from typing import List, Dict, Any
from pydantic import BaseModel, Field


class MessageRecord(BaseModel):
    role: str = Field(..., example="user", description="Role of the message record")
    content: str = Field(..., example="Hello!", description="Content of the message record")


class InferenceRequest(BaseModel):
    messages: List[MessageRecord]
    params: Dict[str, Any] = Field(
        default={}, 
        example={"max_length": 100, "temperature": 0.7},
        description="Model supported parameters for Inference"
    )


class InferenceResponse(BaseModel):
    status_code: int = Field(..., description="Indication for an error occurance")
    response: str = Field(..., description="Model response")


class ErrorResponse(BaseModel):
    status_code: int = Field(..., description="Indication for an error occurance")
    message: str = Field(..., description="Error message")
    traceback: str = Field(None, description="Detailed error message for debugging")
    