# Use the official Python base image with the desired version
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the necessary dependencies for libGL.so.1 and OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    xvfb \
    libglu1-mesa-glx \
    pkg-config \
    libglu1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code to the container
COPY . .

# Expose the port your application listens on (if applicable)
EXPOSE 8000

# Set the command to run your application using uvicorn (modify as needed)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
