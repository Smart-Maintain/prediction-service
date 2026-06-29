# Use the official lightweight Python image
FROM python:3.11-slim

WORKDIR /app

COPY deploy/requirements-api.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src /app/src
COPY artifacts/onnx /app/artifacts/onnx

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "prediction_service.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
