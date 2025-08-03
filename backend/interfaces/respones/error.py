from pydantic import BaseModel, Field

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message describing what went wrong")
    code: int = Field(..., description="HTTP status code associated with the error")
    details: dict = Field(default_factory=dict, description="Additional details about the error, if any")