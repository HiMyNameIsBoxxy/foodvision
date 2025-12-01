FROM python:3.12-slim-bookworm

# Install system libs required by PIL, OpenCV, open_clip, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy dependency list
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set working directory
WORKDIR /app

# Copy all project files TO container
COPY ./src /app/src
COPY ./data /app/data
COPY ./frontend /app/frontend    

# Expose FastAPI port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
