from pydantic import BaseModel, Field
from typing import List

class PredictInput(BaseModel):
    # Validação via Field com regras ge (greater or equal) e le (less or equal)
    pixels: List[float] = Field(
        ..., 
        description="Lista com exatamente 784 valores de pixels normalizados entre 0.0 e 1.0",
        min_items=784,
        max_items=784
    )

class PredictOutput(BaseModel):
    digit_predicted: int = Field(..., description="O dígito numérico previsto pelo modelo (0-9)", ge=0, le=9)
    status: str = Field(..., description="Status da inferência")
