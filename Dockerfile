FROM python:3.13-slim

WORKDIR /app

# Update
RUN apt-get update && apt-get install -y

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifact and application code
COPY model.bin predict.py ./

EXPOSE 8000

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
