from fastapi import APIRouter, HTTPException, status
from schemas import PredictInput, PredictOutput
from services import mnist_service

router = APIRouter()

@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Verifica se o serviço está operando e se o modelo reside na RAM"""
    if mnist_service.model is not None:
        return {
            "status": "healthy",
            "model_loaded": True,
            "message": "API operacional e modelo pronto em RAM."
        }
    
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Modelo não carregado na memória RAM."
    )

@router.post("/v1/predict", response_model=PredictOutput, status_code=status.HTTP_200_OK)
async def predict_digit(payload: PredictInput):
    """Consome a entrada validada e devolve o resultado da predição"""
    try:
        prediction = mnist_service.predict(payload.pixels)
        return PredictOutput(
            digit_predicted=prediction,
            status="Inferência realizada com sucesso"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno no processamento: {str(e)}"
        )
