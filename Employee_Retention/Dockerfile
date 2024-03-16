FROM python:3.8-slim-buster

# Install AWS CLI
RUN apt update -y && apt install awscli -y

# Set working directory
WORKDIR /Employee_Retention

# Copy project files to the working directory
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port on which the application will run
EXPOSE 8000

# Command to run the FastAPI application using uvicorn server
CMD ["uvicorn", "application.src.app:app", "--host", "0.0.0.0", "--port", "8000"]
