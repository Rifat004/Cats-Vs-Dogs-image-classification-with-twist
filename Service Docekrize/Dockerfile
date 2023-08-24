# Using an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
# COPY . /app               #to copy all contents
COPY app.py /app/
COPY pretrained_model.onnx /app/
COPY scratch_model.onnx /app/
COPY requirements.txt /app/


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Expose port 80 for the FastAPI application
EXPOSE 80

# # Define environment variable
# ENV PYTHONPATH "${PYTHONPATH}:/app"

# Run the FastAPI application using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]


