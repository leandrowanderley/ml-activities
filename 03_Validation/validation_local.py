# -*- coding: utf-8 -*-
"""
KDD para Abalone (Type ∈ {1,2,3})

Etapas:
1) Seleção & Entendimento dos Dados
2) Pré-processamento (consistência de colunas, tratamento de NaN)
3) Transformação (One-Hot em 'sex', padronização de numéricas)
4) Modelagem (k-NN por padrão; opção SVM RBF)
5) Avaliação (CV 10-fold, relatório por classe)
6) Preparação para envio (alinha colunas/dummies e gera previsões)
"""

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ============================
# Função principal de KDD
# ============================
def run_kdd_abalone(
    train_path: str | Path,
    app_path: str | Path | None = None,
    algo: str = "knn",           # "knn" (padrão) ou "svm"
    knn_k: int = 5,              # vizinhos no k-NN
    svm_c: float = 10.0,         # C do SVM RBF
    n_folds: int = 10,
    random_state: int = 42,
):
    """
    Executa o pipeline KDD na base do Abalone e (opcionalmente) prevê no app.

    Retorna:
        dict com métricas de CV, relatório por classe (texto), matriz de confusão (array),
        modelo final treinado, transformador (ColumnTransformer) e previsões (se app_path fornecido).
    """
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Arquivo de treino não encontrado: {train_path}")

    # -------------------------
    # 1) Seleção & Leitura
    # -------------------------
    df = pd.read_csv(train_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "sex" not in df.columns or "type" not in df.columns:
        raise ValueError("As colunas obrigatórias 'sex' e 'type' não foram encontradas no CSV.")

    # -------------------------
    # 2) Pré-processamento
    # -------------------------
    # Define colunas
    target_col = "type"
    cat_cols = ["sex"]
    num_cols = [c for c in df.columns if c not in cat_cols + [target_col]]

    # Se houver NaN: imputação simples
    num_transform = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_transform = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transform, num_cols),
            ("cat", cat_transform, cat_cols),
        ],
        remainder="drop",
    )

    # -------------------------
    # 3) Modelagem
    # -------------------------
    if algo.lower() == "svm":
        clf = SVC(kernel="rbf", C=svm_c, gamma="scale", random_state=random_state)
    else:
        clf = KNeighborsClassifier(n_neighbors=knn_k)

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", clf),
    ])

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # -------------------------
    # 4) Avaliação (CV 10-fold)
    # -------------------------
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    y_pred_cv = cross_val_predict(pipe, X, y, cv=cv)

    report_txt = classification_report(y, y_pred_cv, digits=4)
    cm = confusion_matrix(y, y_pred_cv)

    # -------------------------
    # 5) Treino final (100%)
    # -------------------------
    pipe.fit(X, y)

    # -------------------------
    # 6) (Opcional) Prever no APP
    # -------------------------
    preds_app = None
    if app_path is not None:
        app_path = Path(app_path)
        if not app_path.exists():
            raise FileNotFoundError(f"Arquivo de aplicação não encontrado: {app_path}")
        df_app = pd.read_csv(app_path)
        df_app.columns = [c.strip().lower().replace(" ", "_") for c in df_app.columns]

        # Garante colunas essenciais
        if "sex" not in df_app.columns:
            raise ValueError("A coluna 'sex' não foi encontrada no arquivo de aplicação (app).")

        # Seleciona apenas colunas conhecidas (numéricas + 'sex'); ignora 'type' se existir
        use_cols = [c for c in df_app.columns if c in num_cols + cat_cols]
        X_app = df_app[use_cols].copy()

        preds_app = pipe.predict(X_app)  # rótulos 1/2/3

    return {
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_std": float(scores.std()),
        "classification_report": report_txt,
        "confusion_matrix": cm,
        "final_model": pipe,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "predictions_app": preds_app,  # None se app_path não for fornecido
    }


# ============================
# Exemplo de uso direto
# ============================
if __name__ == "__main__":
    # Ajuste os caminhos conforme sua máquina/projeto
    TRAIN = "03_Validation/abalone_dataset.csv"     # treino/validação
    APP   = "03_Validation/abalone_app.csv"         # aplicação (para envio)

    results = run_kdd_abalone(
        train_path=TRAIN,
        app_path=None,         # coloque APP se já tiver o arquivo de aplicação
        algo="knn",            # "knn" (padrão) ou "svm"
        knn_k=5,
        svm_c=10.0,
        n_folds=10,
    )

    print("\n=== Avaliação (CV 10-fold) ===")
    print(f"Acurácia Média: {results['cv_accuracy_mean']:.4f}")
    print(f"Desvio Padrão : {results['cv_accuracy_std']:.4f}")

    print("\n=== Report por Classe (CV) ===")
    print(results["classification_report"])

    print("Matriz de Confusão (CV):")
    print(results["confusion_matrix"])

    # Caso queira prever no app, rode novamente com app_path=APP e salve para envio:
    # results = run_kdd_abalone(TRAIN, APP, algo="knn", knn_k=5)
    # preds = results["predictions_app"]
    # pd.Series(preds, name="type").to_csv("predicoes_abalone_app.csv", index=False)
    #
    # Envio (fora deste script): poste 'predictions' como JSON conforme o enunciado:
    # payload = {"dev_key": "Eneas", "predictions": pd.Series(preds).to_json(orient="values")}
    # requests.post("https://aydanomachado.com/mlclass/03_Validation.php", data=payload)