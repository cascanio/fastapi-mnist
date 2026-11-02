import os
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

print("📥 Baixando um subconjunto do MNIST (aguarde)...")

# CORREÇÃO: Mudamos o parser para 'liac-arff' para evitar o bug de conversão do Pandas
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')

# Convertendo explicitamente para arrays numéricos do NumPy para garantir compatibilidade completa
X = X[:10000] / 255.0  # Seleciona 10k amostras e normaliza o brilho dos pixels (0 a 1)
y = y[:10000].astype(int)

# Divisão de dados (Treino, Validação e Teste)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)

print("🏋️ Treinando o modelo Random Forest...")
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Validação e Métricas
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average='macro')

print(f"✅ Modelo treinado! Acurácia: {acc:.4f} | F1-Score: {f1:.4f}")

# Cria a pasta de modelos se não existir e serializa o artefato
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/mnist_model.joblib")
print("💾 Modelo salvo com sucesso em models/mnist_model.joblib!")
