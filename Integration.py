import pickle
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, classification_report

# Charger le modèle pré-entraîné
model_path = "diabet_model.sav"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Charger les données (ajustez en fonction de vos données)
# Par exemple, X_test et y_test viennent du notebook algo-checkpoint.ipynb
import numpy as np
X_test = np.array([[...], [...]])  # Remplacez par les vraies données de test
y_test = np.array([...])  # Labels réels

# Activer MLflow pour le suivi
mlflow.set_experiment("diabetes_prediction")

with mlflow.start_run():
    # Prédictions
    predictions = model.predict(X_test)
    
    # Calcul des métriques
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    # Enregistrer les métriques dans MLflow
    mlflow.log_metric("accuracy", acc)
    for label, metrics in report.items():
        if isinstance(metrics, dict):  # Ignorer les labels globaux
            for metric, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric}", value)

    # Enregistrer le modèle dans MLflow
    mlflow.sklearn.log_model(model, "diabet_model")

    print(f"Expérience enregistrée avec précision : {acc:.2f}")
