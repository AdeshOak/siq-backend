# Use an official Python runtime as a parent image
FROM python:3.11.2-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the app will run on
EXPOSE 8080

# Define environment variable for production
ENV FLASK_ENV=production

# Set the command to run the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
