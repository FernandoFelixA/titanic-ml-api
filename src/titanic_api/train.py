from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, MinMaxScaler, QuantileTransformer)


# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "titanic_clean.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PIPELINE_PATH = ARTIFACTS_DIR / "pipeline.pkl"

def main():
    # Cargar datos
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Validar estructura del dataset
    expected_columns = {
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    }

    if set(df.columns) != expected_columns:
        raise ValueError(
            "Las columnas del dataset no coinciden con las esperadas."
        )

    if df.empty:
        raise ValueError("El dataset está vacío.")

    if df.isnull().any().any():
        raise ValueError("El dataset contiene valores nulos.")

    if not df["Survived"].isin([0,1]).all():
        raise ValueError("La columna Survived contiene valores no válidos.")

    print(f"Dataset cargado correctamente: {df.shape}")
    print(f"Columnas: {list(df.columns)}")

    # Separar predictores y variable objetivo
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Definir preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(), ["Sex", "Embarked"]),
            (
                "age",
                QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=500
                ),
                ["Age"],
            ),
            (
                "fare",
                QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=500
                ),
                ["Fare"],
            ),
        ],
        remainder="passthrough"
    )

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=100,
        stratify=y)

    # Construir pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("scaler", MinMaxScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=100))
    ])

    # Entrenar pipeline
    pipeline.fit(X_train, y_train)

    print("Entrenamiento completado.")

    # Evaluar modelo
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy en test: {accuracy:.4f}")

    # Guardar el pipeline
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(PIPELINE_PATH, "wb") as archivo:
        pickle.dump(pipeline, archivo)

    print(f"Pipeline guardado en: {PIPELINE_PATH}")


if __name__ == "__main__":
    main()