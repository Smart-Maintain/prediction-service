# Use the official lightweight Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies (No PyTorch needed! Making the image super light)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API and ONNX model files to the working directory
COPY main.py .
COPY multitask_model.onnx .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
