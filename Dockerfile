# Use an official Python runtime as a parent image (Python 3.10+ recommended)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents (including api, ondc_env, frontend) into the container
COPY . .

# Expose port 8000 for the FastAPI server and frontend
EXPOSE 8000

# Command to run the uvicorn server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
