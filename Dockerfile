FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Koyeb uses port 8080 by default for web services
EXPOSE 8080

# Run the app (Notice the port change to 8080)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]