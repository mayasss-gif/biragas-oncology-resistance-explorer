FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV BIRAGAS_PORT=8080
EXPOSE 8080
CMD ["python3", "backend/server.py"]
