# 1. Start with a computer that already has Python 3.10 installed
FROM python:3.10-slim

# 2. Create a folder inside that computer called /app
WORKDIR /app

# 3. Copy your requirements list from your laptop into that /app folder
COPY requirements.txt .

# 4. Run the command to install all your libraries (torch, fastapi, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy ALL your code (src folder, ui folder, etc.) into the /app folder
COPY . .

# 6. Hugging Face Spaces specifically listens to port 7860
EXPOSE 7860

# 7. The "Start Button": This runs your server exactly like you did locally
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]