import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

try:
    df_treino = pd.read_csv('abalone_dataset.csv')
except FileNotFoundError:
    print("Arquivo 'abalone_dataset.csv' não encontrado.")
    exit()

# Pré-processamento, transformando variáveis (M, I, e F) em numéricas
df_treino = df_treino.round(4)
df_treino_processado = pd.get_dummies(df_treino, columns=['sex'])

# Separando features (X) e target (y)
X = df_treino_processado.drop('type', axis=1)
y = df_treino_processado['type']

# Árvore de Decisão
# modelo = DecisionTreeClassifier(random_state=42)

# k-NN
modelo = KNeighborsClassifier(n_neighbors=5)

# Gaussian Naive Bayes
# modelo = GaussianNB()

# Usando validação cruzada (10 folds)
scores_acuracia = cross_val_score(modelo, X, y, cv=10, scoring='accuracy')

print("\nAcurácia com Validação Cruzada:")
print(f"Acurácia Média: {scores_acuracia.mean():.4f}")
print(f"Desvio padrão: {scores_acuracia.std():.4f}")
print("")

# Para uma análise mais profunda, vamos ver o desempenho por classe
previsoes_cv = cross_val_predict(modelo, X, y, cv=10)
report = classification_report(y, previsoes_cv)

print("\nReport por Classe:")
print("Aqui vemos onde o modelo acerta e onde ele tem mais dificuldade.")
print(report)