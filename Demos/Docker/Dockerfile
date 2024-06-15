# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install required packages
RUN apt-get update && apt-get install -y \
    apache2 \
    apache2-utils \
    libapache2-mod-wsgi-py3 \
    net-tools \
    dos2unix \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Convert line endings of start.sh and make it executable
RUN dos2unix /app/start.sh
RUN chmod +x /app/start.sh

# Set up Apache
RUN echo "ServerName localhost" >> /etc/apache2/apache2.conf
COPY 000-default.conf /etc/apache2/sites-available/000-default.conf
RUN a2enmod wsgi

# Expose port 80
EXPOSE 80

# Start Streamlit and Apache
CMD ["/app/start.sh"]