FROM python:3.10-slim

# Установка базовых зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка Python-зависимостей
RUN pip install --timeout=100 -r requirements.txt

# Создание директории для данных и весов модели
RUN mkdir -p /app/weights

# Копирование исходного кода
COPY . .

# Создание конфигурации для Jupyter
RUN mkdir -p /root/.jupyter
RUN echo "c.NotebookApp.ip = '0.0.0.0'" > /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.port = 8888" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.notebook_dir = '/app'" >> /root/.jupyter/jupyter_notebook_config.py

# Открытие порта
EXPOSE 8888

# Запуск Jupyter Notebook при старте контейнера
CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--port=8888", "--no-browser"] 