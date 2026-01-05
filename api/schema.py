# Pydantic schemas for API

from pydantic import BaseModel

class PredictionRequest(BaseModel):
    """Request schema for prediction"""
    pass

class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    pass
