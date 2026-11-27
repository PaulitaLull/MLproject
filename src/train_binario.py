import pandas as pd
import os 
import matplotlib.pyplot as plt

# Ruta de datos
b = r"..\data\raw"

# Nombre exacto del archivo principal 
nombre_archivo = "affect_features_and_labels_folded_all_timesteps.csv"
ruta_completa = os.path.join(b, nombre_archivo)

# Cargamos el archivo en un DataFrame principal y miramos su shape e info
df = pd.read_csv(ruta_completa, low_memory=False)


print("Dimensiones del DataFrame:", df.shape)

# Lista de posibles columnas de targets que aparecen en las columnas
columnas_objetivo = [
    'frustration', 'frustration_1', 'frustration_2', 'frustration_3',
    'boredom', 'boredom_1', 'boredom_2', 'boredom_3',
    'confusion', 'confusion_1', 'confusion_2', 'confusion_3'
]

# Vemos cuántos valores NO nulos tiene cada una
for col in columnas_objetivo:
    if col in df.columns:
        validos = df[col].notna().sum()
        print(f"Columna '{col}': {validos} registros válidos (no nulos)")


# Emociones escogidas para el target:
emo = ["confusion", "concentration", "boredom", "frustration"]
# Eliminamos las filas que tengan nulos en las 4 emos
df_emo = df.dropna(subset=emo, how="all")
# Escogemos las filas que tengan 1 en una de las 4 emociones
df_multi = df_emo[df_emo[emo].sum(axis=1) == 1].copy() 
# Creamos la col target, en la que escribimos la emocion con valor 1 de las 4 emos 
df_multi["target"] = df_multi[emo].idxmax(axis=1)
print(df_multi["target"].value_counts(normalize=True))

emo = ["confusion", "concentration", "boredom", "frustration"]
ids       = ["rowid", "clip_id", "clip_sequence", "fold", "user_id", "clip_sequence_id"]
targets   = emo + ["on_task", "off_task"]
basura   = [f"{e}_{k}" for e in emo for k in (1, 2, 3)]
contexto  = ["urbanicity"]
features  = [c for c in df_emo.columns if c.startswith(("avg_", "max_", "min_", "sum_"))]

quitar = ["rowid", "clip_id", "clip_sequence", "fold", "clip_sequence_id"]   # ids que no usaremos
quitar += emo + ["on_task", "off_task"]                                      # targets originales
quitar += basura                                                           # las 12 columnas basura

df_modelo = df_multi.drop(columns=quitar)
print(df_modelo.shape) 

y = df_modelo["target"]
groups = df_modelo["user_id"]   # columna que usaremos para agrupar a los alumnos
X = df_modelo.drop(columns=["target", "user_id"])

# Realizamos la división train test con sgkf para que no separe a un mismo alumno en diferntes folds
from sklearn.model_selection import StratifiedGroupKFold

# hacemos 5 folds, y next deja el primero a test (el 20%)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
idx_train, idx_test = next(sgkf.split(X, y, groups))

X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
g_train, g_test = groups.iloc[idx_train], groups.iloc[idx_test]

print(X_train.shape, X_test.shape)
print("alumnos en común:", len(set(g_train) & set(g_test)))      # debe ser 0
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

n_valores = X_train.nunique()
constantes= n_valores[n_valores==1].index.tolist()
constantes

import numpy as np
X_train = X_train.drop(columns=constantes)
X_test  = X_test.drop(columns=constantes)  
#     
# Miramos la correlación absoluta entre todas las parejas de columnas
corr = X_train.corr().abs()
# Nos quedamos con una sola mitad de la tabla, para no contar cada pareja dos veces
mitad = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

# Lista de parejas, de más a menos correlacionadas
pares = mitad.stack().sort_values(ascending=False)
altos = pares[pares > 0.95]

print(len(altos), "parejas con correlación > 0.95")
print(altos)

# Aplicaremos log a aquellas features con colas largas mayores a 20
logaritmicas = X_train.columns[X_train.max() > 20].tolist()
print(len(logaritmicas), logaritmicas)

# Seleccionamso en que casos usaremos solo avg y en cuales tambiés con sum
ambas = ["correct", "attemptCount", "hint", "hintCount", "hintTotal", "bottomHint",
         "past8BottomOut", "stlHintUsed", "frIsHelpRequest",
         "frPast8HelpRequest", "scaffold", "frWorkingInSchool"]
solo_avg = ["timeTaken", "totalFrTimeOnSkill", "frPast8WrongCount",
            "totalFrAttempted", "totalFrSkillOpportunities", "totalFrPastWrongCount",
            "totalFrPercentPastWrong", "original"]
cols_reducida = [f"avg_{v}" for v in ambas + solo_avg] + [f"sum_{v}" for v in ambas]
print(len(cols_reducida), [c for c in cols_reducida if c not in X_train.columns])  

cols_modelo = cols_reducida + ["urbanicity"]
X_train_red = X_train[cols_modelo]
X_test_red  = X_test[cols_modelo]

#Las cols a tratar en el preprocesamiento
cols_log= [c for c in cols_reducida if X_train_red[c].max() > 20] 
cols_cat = ["urbanicity"] 
cols_num = [c for c in cols_reducida if c not in cols_log] 

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

#------PREPROCESAMIENTO PARA REG.LOGÍSTICA-------------
pipeline_log_scal = Pipeline([
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", StandardScaler())
]).set_output(transform="pandas")

pipeline_cat= Pipeline([
    ("onehot", OneHotEncoder( sparse_output=False, handle_unknown="ignore"))
])

pipeline_scal = Pipeline([("scaler", StandardScaler())])

prep_rl = ColumnTransformer(
    [
        ("log_scal", pipeline_log_scal, cols_log),          # grupo A
        ("num",      pipeline_scal,      cols_num),         # grupo B
        ("cat",      pipeline_cat,      cols_cat),    # grupo C
    ],
    remainder="drop",
    verbose_feature_names_out=False
).set_output(transform="pandas")

X_train_prep_rl = prep_rl.fit_transform(X_train_red)
print(X_train_prep_rl.shape)
print("nulos:", X_train_prep_rl.isna().sum().sum())
print(X_train_prep_rl[cols_log + cols_num].mean().abs().max().round(6)) 

#-----------PREPROCESAMIENTO ÁRBOLES---------------
prep_arbol = ColumnTransformer(
    [
        ("num", "passthrough", cols_log + cols_num),   
        ("cat", pipeline_cat, cols_cat)            
    ],
    remainder="drop",
    verbose_feature_names_out=False
).set_output(transform="pandas")

X_train_prep_arbol = prep_arbol.fit_transform(X_train_red)
print(X_train_prep_arbol.shape, "| nulos:", X_train_prep_arbol.isna().sum().sum())
print(X_train_prep_arbol[cols_num].equals(X_train_red[cols_num]))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

modelos = {
    "Logistic Regression": (prep_rl, LogisticRegression(max_iter=10000, random_state=42)),
    "Random Forest": (prep_arbol, RandomForestClassifier(random_state=42)),
    "LightGBM": (prep_arbol, LGBMClassifier(random_state=42, verbose=-1))
}

# Nuevo mapeo de la columna target para convertirla en binaria (concentrado 1/ no concentrado 0)
mapa_target = {
    "concentración": 1,
    "concentration": 1,
    "boredom": 0,
    "frustration": 0,
    "confusion": 0,
}

y_train_bin = y_train.map(mapa_target)

metricas_cv = {}
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for nombre, (prep, modelo) in modelos.items():
    pipe = Pipeline([("prep", prep), ("modelo", modelo)])
    metricas_cv[nombre] = cross_val_score(pipe, X_train_red, y_train_bin, groups=g_train,
                                          cv=cv, scoring="f1_macro", n_jobs=-1).mean()

pd.Series(metricas_cv).sort_values(ascending=False) 


from sklearn.metrics import make_scorer, f1_score, classification_report

#---------------OPTIMIZACIÓN DE PARAMS-------------------
pipe = Pipeline([
    ("prep", prep_rl),                 # valor inicial: el grid lo cambia
    ("modelo", LogisticRegression())
])

param_grid = [
    {
        "prep":   [prep_rl],
        "modelo": [LogisticRegression(max_iter=10000, random_state=42)],
        "modelo__C": [0.01, 0.1, 1, 10],
        "modelo__class_weight": [None, "balanced"],
    },
    {
        "prep":   [prep_arbol],
        "modelo": [RandomForestClassifier(random_state=42, n_jobs=1)],
        "modelo__n_estimators": [200, 500],
        "modelo__max_depth": [None, 4, 6, 8],
        "modelo__min_samples_leaf": [5, 10, 15, 20],
        "modelo__class_weight": [None, "balanced","balanced_subsample"],
    },
    {
        "prep":   [prep_arbol],
        "modelo": [LGBMClassifier(random_state=42, verbose=-1, n_jobs=1)],
        "modelo__n_estimators": [100, 250],
        "modelo__learning_rate": [0.05, 0.1],
        "modelo__max_depth": [3, 6],
        "modelo__reg_lambda": [0, 5],
        "modelo__min_child_samples": [10, 20],
        "modelo__class_weight": [None, "balanced"],
    },
]

scoring = {"roc_auc": "roc_auc",
           "f1_macro": "f1_macro",
           "balanced_acc": "balanced_accuracy"}

# scoring = {"f1_raras": f1_raras, "roc_auc_ovr": "roc_auc_ovr", "f1_macro": "f1_macro"}

grid = GridSearchCV(pipe, param_grid, cv=cv, scoring=scoring,
                    refit="f1_macro", n_jobs=-1)
grid.fit(X_train_red, y_train_bin, groups=g_train)

from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

print(type(grid.best_params_["modelo"]).__name__)  #Confirma que es Random Classifier
                                

pred_train = cross_val_predict(clone(grid.best_estimator_), X_train_red, y_train_bin,
                         groups=g_train, cv=cv, n_jobs=-1)

print(classification_report(y_train_bin, pred_train, target_names=["no_concentracion", "concentracion"], zero_division=0))

from sklearn.metrics import roc_auc_score
proba = cross_val_predict(clone(grid.best_estimator_), X_train_red, y_train_bin,
                          groups=g_train, cv=cv, method="predict_proba", n_jobs=-1)

print(proba.shape) 
print("AUC validación:", round(roc_auc_score((y_train_bin == 0).astype(int), proba[:, 0]), 3))
print("F1 macro validación:", round(f1_score(y_train_bin, pred_train, average="macro"), 3))

from sklearn.metrics import ConfusionMatrixDisplay
disp=ConfusionMatrixDisplay.from_predictions(
    y_train_bin, pred_train,
    display_labels=["no_concentrado", "concentrado"],
    normalize="true", values_format=".0%",
    cmap="Blues", xticks_rotation=30)
plt.show()
disp.figure_.savefig("../imgs/matriz_confusion_val_bin.png", dpi=300, bbox_inches="tight")

#---------GUARDADO DEL MEJOR MODELO----------------
import joblib
import os

dir_modelos = "..\modelos"
joblib.dump(grid.best_estimator_, os.path.join(dir_modelos, "final_model_binario.pkl"))

#--------------GUARDADO X_test y_test para siguiente fase-------------------------------

y_test_bin = y_test.map(mapa_target)

os.makedirs(r"..\data\test", exist_ok=True)
X_test_red.to_csv(r"..\data\test/X_test_binario.csv", index=False)
y_test_bin.to_csv(r"..\data\test/y_test_binario.csv", index=False)