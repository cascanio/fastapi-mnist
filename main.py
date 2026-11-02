from contextlib import asynccontextmanager
from fastapi import FastAPI
from routers import router
from services import mnist_service

# 1. Configuração do Ciclo de Vida (Lifespan) da Aplicação
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da API.
    Tudo que está ANTES do 'yield' roda na inicialização do servidor (Startup).
    Tudo que está DEPOIS roda no encerramento do servidor (Shutdown).
    """
    print("🚀 [STARTUP] Inicializando a API...")
    try:
        # Define o caminho do modelo serializado
        model_path = "models/mnist_model.joblib"
        
        print(f"📥 [STARTUP] Carregando o modelo MNIST de '{model_path}' para a memória RAM...")
        # Carrega o modelo na memória RAM uma única vez utilizando o serviço centralizado
        mnist_service.load_model(model_path)
        print("✅ [STARTUP] Modelo carregado com sucesso e pronto para inferências!")
        
    except Exception as e:
        print(f"❌ [STARTUP] ERRO CRÍTICO ao carregar o modelo: {str(e)}")
        print("⚠️ A API subirá sem o modelo carregado. Verifique o arquivo binário.")

    # O 'yield' separa o momento de inicialização do momento de encerramento
    yield
    
    print("🛑 [SHUTDOWN] Encerrando a API e limpando recursos...")

# 2. Inicialização do FastAPI associando o gerenciador lifespan
app = FastAPI(
    title="API de Análise Preditiva - Classificador MNIST",
    description="Interface de microsserviço assíncrono para predição de dígitos manuscritos.",
    version="1.0.0",
    lifespan=lifespan
)

# 3. Inclusão dos roteadores de rotas desacoplados (/health e /v1/predict)
app.include_router(router)

# Rota raiz opcional apenas para fornecer um guia rápido ao acessar http://localhost:8000/
@app.get("/")
async def root():
    return {
        "message": "API de Predição MNIST operando com sucesso!",
        "documentation": "/docs",
        "health_check": "/health"
    }
