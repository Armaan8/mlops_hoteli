# 1) Base image: lightweight Python
FROM python:3.11-slim

# 2) Work inside /app in the container
WORKDIR /app

# 3) Copy & install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy all your code & folders
COPY . .

# 5) When someone runs the container, start the file watcher
CMD ["python", "file_watcher.py"]
