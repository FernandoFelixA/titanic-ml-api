#Importar paquetes
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

#Cargar el modelo
with open("pipeline.pkl", 'rb') as file:
    modelo = pickle.load(file)

#Crear app de Flask
app = Flask(__name__)

@app.route('/predecir', methods=['POST'])
def predict():
    #Obtener JSON
    data = request.get_json()

    #Crear un DataFrame de pandas a partir del JSON
    input_data = pd.DataFrame([data])

    #Hacer predicción usando el modelo que tiene el pipeline que hará la transformación
    prediccion = modelo.predict(input_data)

    #Regresar predicción en formato JSON
    return jsonify({'prediccion': int(prediccion[0])})

if __name__ == '__main__':
    app.run(debug=True)

