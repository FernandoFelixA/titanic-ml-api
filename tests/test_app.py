from src.titanic_api.app import app
import pytest
import importlib

app_module = importlib.import_module("src.titanic_api.app")

def test_rechaza_contenido_no_json():
    cliente = app.test_client()

    respuesta = cliente.post(
        '/predecir',
        data='Esto no es JSON',
        content_type='text/plain',
    )

    assert respuesta.status_code == 415

def test_rechaza_json_no_objeto():
    cliente = app.test_client()

    respuesta = cliente.post(
        '/predecir',
        json=[1, 2, 3],
    )

    assert respuesta.status_code == 400

def test_rechaza_json_sin_columnas():
    cliente = app.test_client()

    respuesta = cliente.post(
        '/predecir',
        json={},
    )

    assert respuesta.status_code == 400
    assert respuesta.json['error'] == 'Error de validacion'

def test_rechaza_columnas_faltantes():
    cliente = app.test_client()

    respuesta = cliente.post(
        '/predecir',
        json={'Pclass':3, 'Sex': 'male'},
    )

    assert respuesta.status_code == 400
    assert respuesta.json['error'] == "Error de validacion"

def test_predice_pasajero_valido():
    cliente = app.test_client()

    datos ={
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
    }

    respuesta = cliente.post('/predecir', json=datos)

    assert respuesta.status_code == 200
    assert 'prediccion' in respuesta.json
    assert respuesta.json['prediccion'] in [0, 1]

@pytest.mark.parametrize(
    "campo, valor",
    [
        ("Pclass", 4),
        ("Pclass", 2.5),
        ("Age", -5),
        ("Age", 101),
        ("Age", None),
        ("Sex", "unknown"),
        ("Embarked", "X"),
        ("SibSp", -1),
        ("Parch", 11),
        ("Fare", -1),
        ("Fare", 601),
    ],
)

def test_rechaza_valores_invalidos(campo, valor):
    cliente = app.test_client()

    datos = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
    }

    datos[campo] = valor

    respuesta = cliente.post('/predecir', json=datos)

    assert respuesta.status_code == 400
    assert respuesta.json['error'] == 'Error de validacion'

def test_error_controlado_si_falla_pipeline(monkeypatch):
    cliente = app.test_client()

    def prediccion_fallida(datos):
        raise RuntimeError("Fallo simulado del modelo")

    monkeypatch.setattr(
        app_module.pipeline,
        "predict",
        prediccion_fallida
    )

    datos = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
    }

    respuesta = cliente.post("/predecir", json=datos)

    assert respuesta.status_code == 500
    assert respuesta.is_json
    assert respuesta.json == {
        'error': 'Error interno al generar la prediccion'
    }

def test_error_si_pipeline_devuelve_clase_inesperada(monkeypatch):
    cliente = app.test_client()

    monkeypatch.setattr(
        app_module.pipeline,
        "predict",
        lambda datos: [2]
    )

    datos = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
    }

    respuesta = cliente.post("/predecir", json=datos)

    assert respuesta.status_code == 500
    assert respuesta.is_json
    assert respuesta.json == {
        'error': 'Error interno al generar la prediccion'
    }

def test_error_si_pipeline_no_devuelve_prediccion(monkeypatch):
    cliente = app.test_client()

    monkeypatch.setattr(
        app_module.pipeline,
        "predict",
        lambda datos: None
    )

    datos = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }

    respuesta = cliente.post("/predecir", json=datos)

    assert respuesta.status_code == 500
    assert respuesta.is_json
    assert respuesta.json == {
        "error": "Error interno al generar la prediccion"
    }

def test_error_si_pipeline_devuelve_lista_vacia(monkeypatch):
    cliente = app.test_client()

    monkeypatch.setattr(
        app_module.pipeline,
        "predict",
        lambda datos: []
    )

    datos = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }

    respuesta = cliente.post("/predecir", json=datos)

    assert respuesta.status_code == 500
    assert respuesta.is_json
    assert respuesta.json == {
        "error": "Error interno al generar la prediccion"
    }

def test_error_si_pipeline_devuelve_escalar(monkeypatch):
    cliente = app.test_client()

    monkeypatch.setattr(
        app_module.pipeline,
        "predict",
        lambda datos: 1
    )

    datos = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }

    respuesta = cliente.post("/predecir", json=datos)

    assert respuesta.status_code == 500
    assert respuesta.is_json
    assert respuesta.json == {
        "error": "Error interno al generar la prediccion"
    }