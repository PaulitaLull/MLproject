# --------------------- FASE DE PREDICCIÓN--------------------------
import pandas as pd
import os
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

dir_modelos = r"..\modelos"
ruta_modelo = os.path.join(dir_modelos, "final_model_binario.pkl")

model = joblib.load(ruta_modelo)

# Evaluación sobre tus datos de test
X_test_red = pd.read_csv(r"..\data\test\X_test_binario.csv")
y_test_bin = pd.read_csv(r"..\data\test\y_test_binario.csv")



pred_test  = model.predict(X_test_red)
proba_test = model.predict_proba(X_test_red)

print(classification_report(y_test_bin, pred_test, target_names=["no_concentrado", "concentrado"], zero_division=0))
print("f1_macro   :", round(f1_score(y_test_bin, pred_test, average="macro"), 3))

from sklearn.metrics import ConfusionMatrixDisplay
disp2=ConfusionMatrixDisplay.from_predictions(
    y_test_bin, pred_test,
    display_labels=["no_concentrado", "concentrado"],
    normalize="true", values_format=".0%",
    cmap="Blues", xticks_rotation=30)
plt.show()
disp2.figure_.savefig("../imgs/matriz_confusion_test_bin.png", dpi=300, bbox_inches="tight")

