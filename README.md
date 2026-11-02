# 🤖 API de Análise Preditiva - Classificador MNIST

Este projeto consiste em um microsserviço web completo e pronto para produção, desenvolvido para disponibilizar um modelo preditivo capaz de reconhecer números manuscritos (de 0 a 9) a partir do famoso conjunto de dados MNIST. 

A solução foi construída utilizando **FastAPI** e **Pydantic** para garantir validações estritas de tipo, baixo tempo de resposta e aderência às práticas modernas de Engenharia de Software e MLOps.

---

## 📝 1. Nome e Resumo do Projeto
* **Problema de Negócio:** Automatização da triagem, leitura e classificação de dígitos numéricos escritos à mão em documentos digitalizados. O objetivo é retirar o modelo preditivo do ambiente isolado de experimentação (Jupyter Notebooks) e fornecê-lo como uma API assíncrona, robusta, escalável e de baixa latência pronta para ser integrada por qualquer aplicação externa do mercado.

---

## 📂 2. Origem do Modelo
* **Origem:** O modelo preditivo utilizado nesta API foi adaptado do ecossistema desenvolvido no mini-projeto anterior de classificação multiclasse (referência: `https://github.com/cascanio/mini-projeto-mnist`).
* **Abordagem Técnica:** Foi utilizado um classificador baseado no algoritmo **Random Forest** (Floresta de Decisão) treinado com um subconjunto de 10.000 imagens do dataset público MNIST_784, normalizado no intervalo entre `0.0` e `1.0` via caminhos relativos de dados.

---

## 🛠️ 3. Tecnologias Utilizadas
A solução foi desenvolvida utilizando a linguagem **Python 3.12+** e as seguintes dependências listadas em ambiente isolado:
* **FastAPI** (Construção do microsserviço assíncrono e rotas HTTP)
* **Pydantic** (Modelagem de esquemas e validação de tipo estrita em tempo de execução)
* **Scikit-Learn** (Mecanismo do modelo preditivo e processamento de dados)
* **Joblib** (Persistência e desserialização do artefato de ML em disco)
* **Uvicorn** (Servidor de aplicação padrão ASGI)
* **Numpy** (Manipulação eficiente de matrizes para inferência matemática)

---

## ⚙️ 4. Instruções de Instalação e Execução

Siga os comandos abaixo no seu terminal para clonar o repositório, preparar o ambiente virtual isolado e iniciar o microsserviço:

```bash
# 1. Clonar o repositório
git clone https://github.com/cascanio/fastapi-mnist
cd fastapi-mnist

# 2. Criar o ambiente virtual (venv)
python -m venv .venv

# 3. Ativar o ambiente virtual
# No Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# No Linux/Mac:
source .venv/bin/activate

# 4. Instalar as dependências do projeto
pip install -r requirements.txt

# 5. Iniciar o servidor Uvicorn com hot-reload ativo
uvicorn main:app --reload
```

---

## 📡 5. Documentação da API

Uma vez que o servidor esteja rodando localmente, a documentação Swagger interativa gerada automaticamente pode ser acessada através do endereço: `http://127.0.0`.

### Exemplos de Payload JSON para Chamadas:

#### Rota 1: `GET /health`
* **Descrição:** Monitoramento de integridade operacional. Verifica se o microsserviço está online e se o arquivo `.joblib` foi carregado devidamente na memória RAM durante a fase de startup (lifespan).
* **Exemplo de Resposta (Status 200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "API operacional e modelo pronto em RAM."
}
```

#### Rota 2: `POST /v1/predict`
* **Descrição:** Envio do vetor numérico de pixels para inferência do modelo. O corpo da mensagem exige uma lista com exatamente 784 números flutuantes situados obrigatoriamente entre `0.0` e `1.0`.
* **Exemplo de Requisição (JSON):**
```json
{
  "pixels": [0.0, 0.0, 0.5, 0.9, 1.0, 0.0, "...[enviar exatamente 784 elementos]..."]
}
```
* **Exemplo de Resposta (Status 200 OK):**
```json
{
  "digit_predicted": 2,
  "status": "Inferência realizada com sucesso"
}
```

---

## 🎥 6. Links das Evidências
* **Vídeo Explicativo do Projeto:** [].
