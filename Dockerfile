FROM python:3.9-slim

WORKDIR /app

# System dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create directories used by the application
RUN mkdir -p input_videos output_videos models stubs

# Run application
ENTRYPOINT ["python", "main.py"]