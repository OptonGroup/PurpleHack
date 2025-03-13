#!/usr/bin/env python
"""
Веб-интерфейс для оптимизатора календарного плана проекта.

Предоставляет API и пользовательский интерфейс для загрузки, оптимизации 
и выгрузки планов проектов в формате JSON.
"""
import os
import json
import logging
import uuid
import requests
import aiohttp
from typing import Dict, Any, List, Optional, Union, Set
from fastapi import FastAPI, Request, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, date

# Импортируем API оптимизатора
from optimizer_api import (
    validate_input_json, 
    optimize_plan,
    get_optimization_status,
    get_all_optimization_results
)

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Директории и настройки
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Создаем необходимые директории
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, 'css'), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, 'js'), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, 'img'), exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# Инициализируем FastAPI
app = FastAPI(
    title="Оптимизатор календарного плана проекта",
    description="API и веб-интерфейс для оптимизации календарного плана проекта",
    version="1.0.0"
)

# Добавляем middleware для сессий
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтируем статические файлы и шаблоны
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Настраиваем Jinja2 для корректной работы с русскими символами
def jinja2_json_filter(obj, **kwargs):
    return json_dumps(obj, indent=2)

templates.env.filters['tojson'] = jinja2_json_filter

# Модели данных
class OptimizationRequest(BaseModel):
    """Модель для запроса оптимизации"""
    data: Dict[str, Any] = Field(..., description="Данные проекта в формате JSON")
    duration_weight: float = Field(7.0, description="Вес длительности проекта")
    resource_weight: float = Field(3.0, description="Вес использования ресурсов")
    cost_weight: float = Field(1.0, description="Вес стоимости")
    num_episodes: int = Field(500, description="Количество эпизодов обучения")
    use_pretrained_model: bool = Field(False, description="Использовать предобученную модель")
    model_path: Optional[str] = Field(None, description="Путь к предобученной модели")
    algorithm: str = Field("reinforcement_learning", description="Алгоритм оптимизации")
    # Параметры для алгоритма обучения с подкреплением
    learning_rate: float = Field(0.001, description="Темп обучения для алгоритма обучения с подкреплением")
    gamma: float = Field(0.99, description="Коэффициент дисконтирования для алгоритма обучения с подкреплением")
    # Параметры для алгоритма имитации отжига
    initial_temperature: float = Field(100.0, description="Начальная температура для алгоритма имитации отжига")
    cooling_rate: float = Field(0.95, description="Скорость охлаждения для алгоритма имитации отжига")
    min_temperature: float = Field(0.1, description="Минимальная температура для алгоритма имитации отжига")
    iterations_per_temp: int = Field(100, description="Количество итераций на каждой температуре для алгоритма имитации отжига")
    max_iterations: int = Field(10000, description="Максимальное количество итераций для алгоритма имитации отжига")

class ValidationResponse(BaseModel):
    """Модель для ответа на валидацию JSON"""
    is_valid: bool = Field(..., description="Результат валидации")
    errors: List[str] = Field([], description="Список ошибок")
    warnings: List[str] = Field([], description="Список предупреждений")

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

def json_dumps(obj, **kwargs):
    """Кастомная функция для сериализации JSON с поддержкой русских символов"""
    return json.dumps(obj, ensure_ascii=False, cls=JSONEncoder, **kwargs)

def json_loads(text, **kwargs):
    """Кастомная функция для десериализации JSON с поддержкой русских символов"""
    return json.loads(text, **kwargs)

# Маршруты для веб-интерфейса
@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """
    Главная страница веб-интерфейса
    """
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

@app.get("/docs-json", response_class=HTMLResponse)
async def docs_page(request: Request):
    """
    Страница с документацией по формату JSON
    """
    return templates.TemplateResponse(
        "docs.html", 
        {"request": request}
    )

@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    """
    Страница с результатами оптимизации
    """
    results = get_all_optimization_results()
    return templates.TemplateResponse(
        "results.html", 
        {"request": request, "results": results}
    )

@app.get("/result/{job_id}", response_class=HTMLResponse)
async def result_page(request: Request, job_id: str):
    """
    Страница с детальным результатом оптимизации
    """
    status = get_optimization_status(job_id)
    
    if status["status"] == "not_found":
        return RedirectResponse(url="/results")
    
    # Если оптимизация завершена, загружаем результаты
    result_data = {}
    if status["status"] == "completed" and os.path.exists(status["output_file"]):
        try:
            with open(status["output_file"], "r", encoding="utf-8") as f:
                optimized_data = json.load(f)
            
            # Проверяем, нужно ли восстановить оригинальную структуру
            if job_id == request.session.get("optimization_job_id"):
                original_structure = request.session.get("original_structure")
                structure_info = request.session.get("structure_info", {})
                tasks_key = structure_info.get("tasks_key", "tasks")
                
                if original_structure:
                    # Заменяем только задачи в оригинальной структуре с правильным ключом
                    original_structure[tasks_key] = optimized_data.get("tasks", [])
                    result_data = original_structure
                else:
                    result_data = optimized_data
            else:
                result_data = optimized_data
                
        except Exception as e:
            logger.error(f"Ошибка при чтении результата: {str(e)}")
    
    return templates.TemplateResponse(
        "result.html", 
        {
            "request": request, 
            "job_id": job_id, 
            "status": status,
            "result": result_data,
            "plan_data": result_data  # Добавляем plan_data для совместимости с шаблоном календаря
        }
    )

# API маршруты
@app.post("/api/validate", response_model=ValidationResponse)
async def validate_json(file: UploadFile = File(...)):
    """
    Валидирует загруженный JSON файл
    """
    # Сохраняем файл во временную директорию
    temp_file_path = os.path.join(TEMP_DIR, file.filename)
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())
    
    # Валидируем JSON
    is_valid, errors, warnings = validate_input_json(temp_file_path)
    
    return ValidationResponse(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings
    )

@app.post("/api/optimize")
async def optimize_json(
    background_tasks: BackgroundTasks,
    request: OptimizationRequest
):
    """
    API для запуска оптимизации JSON-данных.
    """
    # Запускаем оптимизацию
    job_id = optimize_plan(
        data=request.data,
        duration_weight=request.duration_weight,
        resource_weight=request.resource_weight,
        cost_weight=request.cost_weight,
        num_episodes=request.num_episodes,
        use_pretrained_model=request.use_pretrained_model,
        model_path=request.model_path,
        background_tasks=background_tasks,
        algorithm=request.algorithm,
        # Параметры для алгоритма обучения с подкреплением
        learning_rate=request.learning_rate,
        gamma=request.gamma,
        # Параметры для алгоритма имитации отжига
        initial_temperature=request.initial_temperature,
        cooling_rate=request.cooling_rate,
        min_temperature=request.min_temperature,
        iterations_per_temp=request.iterations_per_temp,
        max_iterations=request.max_iterations
    )
    
    # Возвращаем ID задачи
    return {"job_id": job_id, "status": "pending"}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """
    Возвращает статус задачи оптимизации
    """
    return get_optimization_status(job_id)

@app.get("/api/results")
async def get_results():
    """
    Возвращает список всех задач оптимизации
    """
    return get_all_optimization_results()

@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    """
    Возвращает результат оптимизации
    """
    status = get_optimization_status(job_id)
    
    if status["status"] == "not_found":
        return JSONResponse(
            status_code=404,
            content={"error": "Задача не найдена"}
        )
    
    if status["status"] != "completed":
        return status
    
    # Загружаем результаты из файла
    try:
        with open(status["output_file"], "r", encoding="utf-8") as f:
            result_data = json.load(f)
            
        return {
            "status": status,
            "result": result_data
        }
    
    except Exception as e:
        logger.error(f"Ошибка при чтении результата: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Ошибка при чтении результата: {str(e)}"}
        )

@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    """
    Загружает результат оптимизации в виде JSON файла
    """
    status = get_optimization_status(job_id)
    
    if status["status"] == "not_found":
        return JSONResponse(
            status_code=404,
            content={"error": "Задача не найдена"}
        )
    
    if status["status"] != "completed":
        return JSONResponse(
            status_code=400,
            content={"error": "Оптимизация еще не завершена"}
        )
    
    if not os.path.exists(status["output_file"]):
        return JSONResponse(
            status_code=404,
            content={"error": "Файл результата не найден"}
        )
    
    return FileResponse(
        status["output_file"],
        media_type="application/json",
        filename=f"optimized_plan_{job_id}.json"
    )

# Маршрут для загрузки файла через форму
@app.post("/upload")
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    duration_weight: float = Form(7.0),
    resource_weight: float = Form(3.0),
    cost_weight: float = Form(1.0),
    num_episodes: int = Form(500),
    use_pretrained_model: bool = Form(False),
    algorithm: str = Form("reinforcement_learning"),
    initial_temperature: float = Form(100.0),
    cooling_rate: float = Form(0.95),
    min_temperature: float = Form(0.1),
    iterations_per_temp: int = Form(100),
    max_iterations: int = Form(10000)
):
    """
    Загружает файл JSON и запускает оптимизацию.
    """
    # Создаем временный файл для сохранения загруженного JSON
    temp_file = os.path.join(TEMP_DIR, f"upload_{uuid.uuid4()}.json")
    
    # Сохраняем загруженный файл
    with open(temp_file, "wb") as f:
        f.write(await file.read())
    
    # Проверяем файл на валидность
    is_valid, errors, warnings = validate_input_json(temp_file)
    
    if not is_valid:
        # Если файл невалидный, возвращаем ошибки
        os.remove(temp_file)  # Удаляем временный файл
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Ошибка валидации файла", "details": errors}
        )
    
    # Загружаем JSON
    with open(temp_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Удаляем временный файл
    os.remove(temp_file)
    
    # Запускаем оптимизацию
    job_id = optimize_plan(
        data=data,
        duration_weight=duration_weight,
        resource_weight=resource_weight,
        cost_weight=cost_weight,
        num_episodes=num_episodes,
        use_pretrained_model=use_pretrained_model,
        background_tasks=background_tasks,
        algorithm=algorithm,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        min_temperature=min_temperature,
        iterations_per_temp=iterations_per_temp,
        max_iterations=max_iterations
    )
    
    # Перенаправляем на страницу статуса
    return RedirectResponse(url=f"/result/{job_id}", status_code=303)

# Маршруты для работы с сервером Sbertech
@app.get("/server-load", response_class=HTMLResponse)
async def server_load_page(request: Request):
    """
    Страница для загрузки плана с сервера Sbertech
    """
    return templates.TemplateResponse(
        "server_load.html",
        {"request": request}
    )

@app.post("/server-load", response_class=HTMLResponse)
async def load_from_server(
    request: Request,
    id_1: str = Form(...),
    id_2: str = Form(...),
    bearer: str = Form(...)
):
    """
    Загрузка плана с сервера Sbertech
    """
    try:
        # Формируем URL для запроса
        base_url = "https://saas.works.sbertech.ru/gantt/rest/srvc/v1/accountingObjects"
        url = f"{base_url}/{id_1}/plans/{id_2}"
        
        logger.info(f"Отправка запроса на URL: {url}")
        
        # Формируем заголовки
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json; charset=utf-8",
            "X-Dspc-Tenant": "SBT-TNT",
            "Authorization": f"Bearer {bearer}",
            "User-Agent": "Mozilla/5.0"
        }
        
        logger.info("Заголовки запроса (без Authorization):", 
                   {k: v for k, v in headers.items() if k != "Authorization"})
        
        # Выполняем запрос к серверу
        async with aiohttp.ClientSession(json_serialize=json_dumps) as session:
            async with session.get(url, headers=headers) as response:
                logger.info(f"Получен ответ с кодом: {response.status}")
                
                if response.status == 200:
                    text = await response.text()
                    plan_data = json_loads(text)
                    logger.info("План успешно загружен")
                    
                    # Сохраняем план в сессии
                    request.session["plan_data"] = plan_data
                    
                    return templates.TemplateResponse(
                        "server_load.html",
                        {
                            "request": request,
                            "plan_data": plan_data,
                            "success": "План успешно загружен"
                        }
                    )
                elif response.status == 401:
                    error_text = await response.text(encoding='utf-8')
                    logger.error(f"Ошибка авторизации: {error_text}")
                    return templates.TemplateResponse(
                        "server_load.html",
                        {
                            "request": request,
                            "error": "Ошибка авторизации. Пожалуйста, проверьте правильность Bearer токена."
                        }
                    )
                elif response.status == 404:
                    error_text = await response.text()
                    logger.error(f"План не найден: {error_text}")
                    return templates.TemplateResponse(
                        "server_load.html",
                        {
                            "request": request,
                            "error": f"План не найден. Проверьте правильность ID: {id_1}/{id_2}"
                        }
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"Ошибка при загрузке плана: {error_text}")
                    return templates.TemplateResponse(
                        "server_load.html",
                        {
                            "request": request,
                            "error": f"Ошибка при загрузке плана (код {response.status}): {error_text}"
                        }
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Ошибка сети при выполнении запроса: {str(e)}")
        return templates.TemplateResponse(
            "server_load.html",
            {
                "request": request,
                "error": f"Ошибка сети при выполнении запроса: {str(e)}"
            }
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {str(e)}")
        return templates.TemplateResponse(
            "server_load.html",
            {
                "request": request,
                "error": f"Неожиданная ошибка: {str(e)}"
            }
        )

@app.post("/optimize-server-plan")
async def optimize_server_plan(
    request: Request,
    background_tasks: BackgroundTasks,
    data_source: str = Form(None),
    json_file: UploadFile = File(None),
    plan_data: str = Form(None),
    original_structure: bool = Form(True),
    algorithm: str = Form("reinforcement_learning"),
    duration_weight: float = Form(7.0),
    resource_weight: float = Form(3.0), 
    cost_weight: float = Form(1.0),
    num_episodes: int = Form(500),
    use_pretrained_model: bool = Form(False),
    initial_temperature: float = Form(100.0),
    cooling_rate: float = Form(0.95),
    min_temperature: float = Form(0.1),
    iterations_per_temp: int = Form(100),
    max_iterations: int = Form(10000)
):
    """
    Оптимизация плана, загруженного с сервера или из файла
    """
    try:
        # Определяем источник данных и получаем JSON
        if data_source == "file" and json_file:
            # Читаем данные из файла по частям
            contents = []
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := await json_file.read(chunk_size):
                contents.append(chunk)
            file_content = b''.join(contents)
            
            try:
                data = json_loads(file_content.decode('utf-8'))
            except UnicodeDecodeError:
                try:
                    data = json_loads(file_content.decode('latin-1'))
                except:
                    return templates.TemplateResponse(
                        "server_load.html",
                        {
                            "request": request,
                            "error": "Ошибка при чтении файла. Убедитесь, что это валидный JSON файл в кодировке UTF-8."
                        }
                    )
        elif plan_data:
            try:
                data = json_loads(plan_data)
            except:
                return templates.TemplateResponse(
                    "server_load.html",
                    {
                        "request": request,
                        "error": "Ошибка при разборе JSON данных."
                    }
                )
        else:
            return templates.TemplateResponse(
                "server_load.html",
                {
                    "request": request,
                    "error": "Не найдены данные плана. Сначала загрузите план с сервера или выберите файл."
                }
            )

        # Сохраняем оригинальную структуру JSON
        original_data = data.copy()
        
        # Получаем задачи из данных (учитываем разные возможные форматы)
        tasks = []
        tasks_key = ""  # Для запоминания ключа, где находятся задачи
        
        if "tasks" in data:
            tasks = data.get("tasks", [])
            tasks_key = "tasks"
        else:
            # Ищем задачи в других возможных местах структуры
            for key, value in data.items():
                if isinstance(value, list) and value and isinstance(value[0], dict) and "id" in value[0]:
                    tasks = value
                    tasks_key = key
                    break
        
        if not tasks:
            return templates.TemplateResponse(
                "server_load.html",
                {
                    "request": request,
                    "error": "В загруженных данных не найдены задачи. Проверьте формат JSON файла."
                }
            )
        
        # Проверяем наличие ресурсов и добавляем их, если нет
        resources = []
        resources_key = ""
        
        # Сначала ищем ресурсы в данных
        if "resources" in data:
            resources = data.get("resources", [])
            resources_key = "resources"
        else:
            # Ищем ресурсы в других возможных местах структуры
            for key, value in data.items():
                if (isinstance(value, list) and value and isinstance(value[0], dict) and 
                    ("capacity" in value[0] or "name" in value[0]) and key != tasks_key):
                    resources = value
                    resources_key = key
                    break
        
        # Если ресурсов нет, собираем уникальные ресурсы из задач
        if not resources:
            unique_resources = set()
            for task in tasks:
                if "assignedResourceId" in task and task["assignedResourceId"]:
                    unique_resources.add(task["assignedResourceId"])
                elif "resource" in task and task["resource"]:
                    unique_resources.add(task["resource"])
            
            # Создаем ресурсы из уникальных идентификаторов
            for resource_id in unique_resources:
                resources.append({
                    "id": resource_id,
                    "name": f"Ресурс {resource_id}",
                    "capacity": 1
                })
            
            # Если ресурсы были созданы, но ключ не определен, используем "resources"
            if resources and not resources_key:
                resources_key = "resources"
                # Добавляем ресурсы в исходную структуру
                original_data[resources_key] = resources
        
        # Сохраняем информацию о структуре в сессии
        request.session["structure_info"] = {
            "tasks_key": tasks_key,
            "resources_key": resources_key
        }
        
        # Создаем запрос на оптимизацию
        optimization_request = OptimizationRequest(
            data={
                "tasks": tasks,
                "resources": resources
            },
            duration_weight=duration_weight,
            resource_weight=resource_weight,
            cost_weight=cost_weight,
            num_episodes=num_episodes,
            use_pretrained_model=use_pretrained_model,
            algorithm=algorithm,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            min_temperature=min_temperature,
            iterations_per_temp=iterations_per_temp,
            max_iterations=max_iterations
        )
        
        # Запускаем оптимизацию
        result = await optimize_json(background_tasks, optimization_request)
        
        if original_structure:
            # Сохраняем оригинальную структуру в сессии для последующего использования
            request.session["original_structure"] = original_data
            request.session["optimization_job_id"] = result["job_id"]
        
        # Перенаправляем на страницу с результатами
        return RedirectResponse(url=f"/result/{result['job_id']}", status_code=303)
        
    except Exception as e:
        logger.error(f"Ошибка при оптимизации плана: {str(e)}", exc_info=True)
        return templates.TemplateResponse(
            "server_load.html",
            {
                "request": request,
                "error": f"Ошибка при оптимизации плана: {str(e)}"
            }
        )

@app.post("/create-new-plan")
async def create_new_plan(
    request: Request,
    id_1: str = Form(...),
    bearer: str = Form(...),
    plan_data: str = Form(...)
):
    """
    Создание нового плана на сервере Sbertech
    """
    try:
        # Формируем URL для запроса
        base_url = "https://saas.works.sbertech.ru/gantt/rest/srvc/v1/accountingObjects"
        url = f"{base_url}/{id_1}/plans/new"
        
        # Формируем заголовки
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json; charset=utf-8",
            "X-Dspc-Tenant": "SBT-TNT",
            "Authorization": f"Bearer {bearer}",
            "User-Agent": "Mozilla/5.0"
        }
        
        # Преобразуем данные плана
        data = json_loads(plan_data)
        
        # Выполняем запрос к серверу
        async with aiohttp.ClientSession(json_serialize=json_dumps) as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    text = await response.text()
                    result_data = json_loads(text)
                    return templates.TemplateResponse(
                        "server_load.html",
                        {
                            "request": request,
                            "success": "Новый план успешно создан",
                            "plan_data": result_data
                        }
                    )
                else:
                    error_text = await response.text()
                    return templates.TemplateResponse(
                        "server_load.html",
                        {
                            "request": request,
                            "error": f"Ошибка при создании плана: {error_text}",
                            "plan_data": plan_data
                        }
                    )
    except Exception as e:
        return templates.TemplateResponse(
            "server_load.html",
            {
                "request": request,
                "error": f"Ошибка при выполнении запроса: {str(e)}",
                "plan_data": plan_data
            }
        )

@app.get("/download-plan")
async def download_plan(request: Request):
    """
    Скачивание текущего плана в формате JSON
    """
    # Создаем временный файл
    temp_file = os.path.join(TEMP_DIR, f"plan_{uuid.uuid4()}.json")
    
    try:
        # Получаем данные из сессии или другого хранилища
        # В данном случае мы просто возвращаем последний загруженный план
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(request.session.get("plan_data", {}), f, ensure_ascii=False, indent=2)
        
        return FileResponse(
            temp_file,
            media_type="application/json",
            filename="plan.json"
        )
    finally:
        # Удаляем временный файл после отправки
        if os.path.exists(temp_file):
            os.remove(temp_file)

# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 