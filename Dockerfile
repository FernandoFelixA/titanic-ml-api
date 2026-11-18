FROM python:3.12-slim

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar archivos del proyecto
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# Crear y utilizar usuario sin privilegios
RUN useradd --create-home appuser
USER appuser

# Puerto de la API
EXPOSE 5000

# Iniciar aplicación
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "src.titanic_api.app:app"]
