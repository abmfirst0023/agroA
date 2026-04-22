FROM python:3.10-slim

WORKDIR /app

# Install libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your model and code
COPY . .

# Run using Gunicorn (Standard for Railway)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]

