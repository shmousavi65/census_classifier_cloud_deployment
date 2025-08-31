FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Uvicorn will serve the FastAPI app
EXPOSE 8000
CMD ["uvicorn", "development.main:app", "--host", "0.0.0.0", "--port", "8000"]
