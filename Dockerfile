FROM python:3.12-slim

WORKDIR /app

# Install the CPU-only PyTorch wheel before the application requirements.
# No CUDA libraries are needed for inference, keeping the image much smaller.
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
