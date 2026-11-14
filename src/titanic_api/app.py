#Importar paquetes
from flask import Flask, request, jsonify
import math
import pickle
import pandas as pd
from pathlib import Path

#Construir ruta del pipeline
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "artifacts" / "pipeline.pkl"

#Cargar el modelo
with open(MODEL_PATH, 'rb') as file:
    pipeline = pickle.load(file)

#Crear app de Flask
app = Flask(__name__)

@app.route('/predecir', methods=['POST'])
def predict():
    #Obtener JSON
    #1. Verificar que la solicitud declara contenido JSON
    if not request.is_json:
        return jsonify({
            'error': 'El contenido debe ser JSON.'
        }), 415

    #2. Intentar obtener el JSON
    try:
        data = request.get_json()
    except Exception:
        return jsonify({
            'error': 'El JSON enviado no es valido.'
        }), 400

    #3. Verificar que recibimos un objeto JSON
    if not isinstance(data, dict):
        return jsonify({
            'error': 'Se esperaba un objeto JSON.'
        }), 400

    errores = []

    #4. Verificar las columnas que se ingresaron
    columnas_requeridas = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    columnas_faltantes = [n for n in columnas_requeridas if n not in data]
    columnas_adicionales = [n for n in data if n not in columnas_requeridas]

    if columnas_faltantes:
        errores.append({'error': 'Faltan columnas obligatorias.', 'columnas_faltantes': columnas_faltantes})

    if columnas_adicionales:
        errores.append({'error': 'Se recibieron columnas no permitidas.', 'columnas_adicionales': columnas_adicionales})

    #6. Validar Pclass
    if 'Pclass' in data:
        valoresPclass = [1, 2, 3]
        if type(data['Pclass']) is not int:
            errores.append({'error': 'Se esperaba Pclass como numero entero'})
        elif data['Pclass'] not in valoresPclass:
            errores.append({'error': 'El valor ingresado de Pclass no existe'})

    #7. Validar Age

    if 'Age' in data:
        if data['Age'] is None:
            errores.append({'error': 'La edad no puede ser None'})
        elif not isinstance(data['Age'], (float, int)) or isinstance(data['Age'], bool):
            errores.append({'error': 'La edad debe ser un numero'})
        elif not math.isfinite(data['Age']):
            errores.append({'error': 'La edad debe ser un numero finito'})
        elif not 0 <= data['Age'] <= 100:
            errores.append({'error': 'La edad esta fuera del rango permitido'})

    #8. Validar Sex
    if 'Sex' in data:
        valoresSex = ["male", "female"]
        if not isinstance(data['Sex'], str):
            errores.append({'error': 'Se esperaba Sex como un texto'})
        elif data['Sex'].lower() not in valoresSex:
            errores.append({'error': 'Los valores permitidos de Sex son "male" y "female"'})
        else:
            data['Sex'] = data['Sex'].lower()

    #9. Validar Embarked
    if 'Embarked' in data:
        valoresEmbarked = ["S", "C", "Q"]
        if data['Embarked'] is None:
            errores.append({'error': 'Se ingreso un valor Embarked tipo None'})
        elif not isinstance(data['Embarked'], str):
            errores.append({'error': 'El valor de Embarked debe ser un texto'})
        elif data['Embarked'].upper() not in valoresEmbarked:
            errores.append({'error': 'Se ingreso un valor de Embarked no permitido. Valores permitidos: "S", "C", "Q"'})
        else:
            data['Embarked'] = data['Embarked'].upper()

    #10. Validar SibSp
    if 'SibSp' in data:
        if not isinstance(data['SibSp'], int) or isinstance(data['SibSp'], bool):
            errores.append({'error': 'El valor de SibSp debe ser un numero entero'})
        elif 0 > data['SibSp'] or data['SibSp'] > 10:
            errores.append({'error': 'Se ingreso un numero de SibSp no permitido. Valores permitidos: 0-10'})

    #11. Validar Parch
    if 'Parch' in data:
        if not isinstance(data['Parch'], int) or isinstance(data['Parch'], bool):
            errores.append({'error': 'El valor de Parch debe ser un numero entero'})
        elif data['Parch'] < 0 or data['Parch'] > 10:
            errores.append({'error': 'El valor de Parch no es permitido. Valores permitidos: 0-10'})

    #12. Validar Fare
    if 'Fare' in data:
        if not isinstance(data['Fare'], (int, float)) or isinstance(data['Fare'], bool):
            errores.append({'error': 'Fare debe ser un numero entero o decimal'})
        elif not math.isfinite(data['Fare']):
            errores.append({'error': 'Fare debe ser un valor finito'})
        elif 0 > data['Fare'] or data['Fare'] > 600:
            errores.append({'error': 'Se debe ingresar un valor de Fare entre 0 y 600'})


    if errores:
        return jsonify({
            'error': 'Error de validacion',
            'detalles': errores
        }), 400

    #Crear un DataFrame de pandas a partir del JSON
    input_data = pd.DataFrame([data])

    #Hacer predicción usando el modelo que tiene el pipeline que hará la transformación
    try:
        prediccion = pipeline.predict(input_data)
    except Exception:
        app.logger.exception('Error al generar la prediccion')
        return jsonify({
            'error': 'Error interno al generar la prediccion'
        }), 500

    #Regresar predicción en formato JSON
    try:
        salida_invalida = (
            prediccion is None
            or len(prediccion) != 1
            or prediccion[0] not in (0, 1)
        )

    except (TypeError, IndexError, KeyError):
        salida_invalida = True

    if salida_invalida:
        app.logger.error("El pipeline devolvió una predicción inesperada")

        return jsonify({
            'error': 'Error interno al generar la prediccion'
        }), 500

    return jsonify({'prediccion': int(prediccion[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
