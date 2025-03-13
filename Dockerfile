FROM python:3.10-slim

# Установка рабочей директории
WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install -r requirements.txt

# Копирование исходного кода
COPY . .

# Установка переменных среды
ENV PYTHONPATH=/app
ENV PORT=8000

# Открываем порт
EXPOSE 8000

# Запуск приложения через uvicorn
CMD ["uvicorn", "web_app.app:app", "--host", "0.0.0.0", "--port", "8000"]