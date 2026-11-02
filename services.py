import joblib
import numpy as np

class MNISTService:
    def __init__(self):
        self.model = None

    def load_model(self, model_path: str):
        """Carrega o modelo apenas uma vez na memória RAM"""
        self.model = joblib.load(model_path)

    def predict(self, pixels: list) -> int:
        """Executa a inferência matemática isolada"""
        if self.model is None:
            raise RuntimeError("O modelo preditivo não foi carregado na memória.")
        
        # Converte para o formato de matriz esperado pelo scikit-learn (1, 784)
        input_data = np.array(pixels).reshape(1, -1)
        prediction = self.model.predict(input_data)
        return int(prediction[0])

# Instância única global do serviço para ser compartilhada pela aplicação
mnist_service = MNISTService()
