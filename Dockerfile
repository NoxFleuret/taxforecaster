FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy logic
COPY . .

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
