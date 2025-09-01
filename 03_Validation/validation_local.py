# -*- coding: utf-8 -*-
"""
KDD Simplificado — Abalone
- sex: M/F→A, I→I; depois A→1 e I→0 (numérico)
- Remove linhas com zeros improváveis (length/diameter/height/weight)
- Modelo direto (KNN por padrão; pode trocar para SVC)
- CV 10-fold + relatório por classe
- Imagens: confusion_matrix.png e correlation_matrix.png
- Envio automático no final (padrão do curso)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 

# ----------------------------
# CONFIG
# ----------------------------
TRAIN_PATH = Path("03_Validation/abalone_dataset.csv")
APP_PATH   = Path("03_Validation/abalone_app.csv")

URL = "https://aydanomachado.com/mlclass/03_Validation.php"
DEV_KEY = "VG" 

KNN_K = 5
N_FOLDS = 10
RANDOM_STATE = 42

# ----------------------------
# Helpers
# ----------------------------
def map_sex_to_ai(series: pd.Series) -> pd.Series:
    return series.map(lambda x: 'A' if x in ['M','F'] else ('I' if x == 'I' else x))

def sex_ai_to_numeric(series: pd.Series) -> pd.Series:
    return series.map({'A': 1, 'I': 0}).astype(float)

def remove_zero_improvavel(df: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
    """Remove linhas com zero em colunas físicas (robusto p/ string, -0.0, etc)."""
    cols = list(df.columns)
    phys_keys = ["length", "diameter", "height", "weight"]
    phys_cols = [c for c in cols if any(k in c for k in phys_keys)]

    for c in phys_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    zero_counts = {}
    for c in phys_cols:
        if c in df.columns:
            cnt = int((df[c].abs() <= tol).fillna(False).sum())
            zero_counts[c] = cnt

    print(f" - Colunas com regra de zero (medidas físicas): {phys_cols}")
    print(" - Linhas com zeros 'improváveis':")
    if any(zero_counts.values()):
        for col, val in zero_counts.items():
            print(f"   {col}: {val}")
    else:
        print("   nenhuma")

    mask_ok = np.ones(len(df), dtype=bool)
    for c in phys_cols:
        if c in df.columns:
            mask_ok &= ~((df[c].abs() <= tol).fillna(False))

    removed = int((~mask_ok).sum())
    if removed:
        print(f" - Removendo {removed} linhas com zeros improváveis.")
    return df.loc[mask_ok].reset_index(drop=True)

def plot_confusion(cm: np.ndarray, classes: list[str], out_path: Path):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', aspect='auto', cmap='Blues')
    plt.title('Matriz de Confusão (CV)')
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_correlations(df_num: pd.DataFrame, out_path: Path):
    corr = df_num.corr(numeric_only=True)
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(corr, interpolation='nearest', aspect='auto', cmap='Blues')
    plt.title('Correlação entre variáveis numéricas')
    plt.colorbar()
    ticks = np.arange(len(corr.columns))
    plt.xticks(ticks, corr.columns, rotation=45, ha='right')
    plt.yticks(ticks, corr.columns)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    # === Leitura treino
    df = pd.read_csv(TRAIN_PATH)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["sex"] = map_sex_to_ai(df["sex"])
    df["sex_num"] = sex_ai_to_numeric(df["sex"])

    target = "type"
    num_cols = [c for c in df.columns if c not in [target, "sex"]]

    print("=== Diagnóstico rápido (treino) ===")
    print(" - Nulos por coluna:")
    for col, val in df.isna().sum().items():
        print(f"   {col}: {val}")
    df = remove_zero_improvavel(df)
    print(f" - Linhas após limpeza: {len(df)}")

    out_dir = TRAIN_PATH.parent
    plot_correlations(df[num_cols], out_path=out_dir / "correlation_matrix.png")
    print(f" - Heatmap de correlações salvo em: {out_dir/'correlation_matrix.png'}")

    # === Modelo
    # model = KNeighborsClassifier(n_neighbors=KNN_K)
    model = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=RANDOM_STATE)
    X, y = df[num_cols].copy(), df[target].astype(int)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f" - Acurácia média (CV): {scores.mean():.4f} | desvio padrão: {scores.std():.4f}")

    y_pred_cv = cross_val_predict(model, X, y, cv=cv)
    print("=== Report por classe (CV) ===")
    print(classification_report(y, y_pred_cv, digits=4))

    cm = confusion_matrix(y, y_pred_cv)
    print("Matriz de Confusão (CV):\n", cm)
    plot_confusion(cm, classes=["1","2","3"], out_path=out_dir / "confusion_matrix.png")
    print(f" - Matriz de confusão salva em: {out_dir/'confusion_matrix.png'}")

    # === Treino final
    model.fit(X, y)

    # === Aplicar no app
    df_app = pd.read_csv(APP_PATH)
    df_app.columns = [c.strip().lower().replace(" ", "_") for c in df_app.columns]
    print(" - Nulos por coluna:")
    for col, val in df_app.isna().sum().items():
        print(f"   {col}: {val}")
    df_app = remove_zero_improvavel(df_app)
    print(f" - Linhas após limpeza: {len(df_app)}")

    df_app["sex"] = map_sex_to_ai(df_app["sex"])
    df_app["sex_num"] = sex_ai_to_numeric(df_app["sex"])

    X_app = df_app[[c for c in num_cols if c in df_app.columns]].copy()
    X_app = remove_zero_improvavel(X_app)

    y_pred = model.predict(X_app)
    print(f" - Total de previsões: {len(y_pred)} | Amostra 10: {y_pred[:10].tolist()}")

    # === Envio automático ===
    print(" - Enviando para o servidor...")
    # data = {
    #     "dev_key": DEV_KEY,
    #     "predictions": pd.Series(y_pred).to_json(orient="values")
    # }
    # r = requests.post(url=URL, data=data)
    # print(" - Resposta do servidor:\n", r.text, "\n")

    print("=== Concluído ===")

if __name__ == "__main__":
    main()