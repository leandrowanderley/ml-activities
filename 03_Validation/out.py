import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

try:
    df_treino = pd.read_csv('abalone_dataset.csv')
    print("Arquivo 'abalone_dataset.csv' carregado com sucesso.")
except FileNotFoundError:
    print("ERRO: Arquivo 'abalone_dataset.csv' não encontrado. Verifique se ele está na mesma pasta do seu script.")
    exit()

casas_decimais = 4
df_treino = df_treino.round(casas_decimais)

df_treino_processado = pd.get_dummies(df_treino, columns=['sex'], drop_first=False)
print("Pré-processamento concluído.")


X = df_treino_processado.drop('type', axis=1)

y = df_treino_processado['type']

print("-" * 50)
print("Iniciando a etapa de validação do modelo...")
print("-" * 50)


# --- Construção e Validação do Modelo ---

# 1. Criar uma instância do modelo RandomForestClassifier
#    - n_estimators=150: O modelo usará 150 árvores de decisão.
#    - random_state=42: Garante que o resultado seja o mesmo sempre que o código for executado.
#    - n_jobs=-1: Usa todos os processadores disponíveis para acelerar o treinamento.
modelo = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)

# 2. Executar a Validação Cruzada para medir a Acurácia
#    - cv=10: Divide os dados em 10 partes. O modelo treina em 9 e testa em 1, repetindo o processo 10 vezes.
#    - scoring='accuracy': A métrica para avaliar o modelo é a acurácia.
scores_acuracia = cross_val_score(modelo, X, y, cv=10, scoring='accuracy')

# 3. Imprimir os resultados da acurácia
print("\n--- RESULTADOS DA ACURÁCIA (Validação Cruzada) ---")
print(f"Acurácia Média: {scores_acuracia.mean():.4f} (Este é o resultado mais importante!)")
print(f"Desvio Padrão da Acurácia: {scores_acuracia.std():.4f}")
print("-" * 50)


# --- Análise Detalhada por Classe ---

# 1. Gerar previsões usando a validação cruzada para uma análise mais profunda
previsoes_cv = cross_val_predict(modelo, X, y, cv=10)

# 2. Criar o Relatório de Classificação
#    Este relatório mostra Precision, Recall e F1-Score para cada classe (1, 2 e 3).
report = classification_report(y, previsoes_cv)

print("\n--- RELATÓRIO DE CLASSIFICAÇÃO DETALHADO ---")
print("Este relatório mostra os pontos fortes e fracos do modelo para cada classe.")
print(report)
