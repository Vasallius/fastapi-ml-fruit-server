# Use the official NVIDIA CUDA base image with Python 3.8
FROM nvidia/cuda:11.0-base-ubuntu20.04

# Set the working directory inside the container
WORKDIR /app

# Install Python 3.8
RUN apt-get update && \
    apt-get install -y python3.8 python3.8-venv python3.8-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install the Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the port your application listens on (if applicable)
EXPOSE 8000

# Set the command to run your application using uvicorn (modify as needed)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
