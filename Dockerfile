# 1. Use Python 3.10
FROM python:3.10-slim

# 2. Install Git and required system dependencies for Xet
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# 3. Set directory
WORKDIR /app

# 4. Install Python dependencies
# Adding 'hf-xet' to ensure the Xet files can be resolved
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "huggingface_hub[hf_xet]"

# 5. Copy code
COPY . .

# 6. Expose port
EXPOSE 7860

# 7. Start
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]