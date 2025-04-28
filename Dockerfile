# Dockerfile for f1-mlops API service
FROM python:3.10-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and artifacts
COPY src/ src/
COPY artifacts/ artifacts/

# Expose port and run app
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
