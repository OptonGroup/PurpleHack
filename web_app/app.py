#!/usr/bin/env python
"""
Веб-интерфейс для оптимизатора календарного плана проекта.

Предоставляет API и пользовательский интерфейс для загрузки, оптимизации 
и выгрузки планов проектов в формате JSON.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from fastapi import FastAPI, Request, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# Инициализируем FastAPI
app = FastAPI(
    title="Оптимизатор календарного плана проекта",
    description="API и веб-интерфейс для оптимизации календарного плана проекта",
    version="1.0.0"
)

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

class ValidationResponse(BaseModel):
    """Модель для ответа на валидацию JSON"""
    is_valid: bool = Field(..., description="Результат валидации")
    errors: List[str] = Field([], description="Список ошибок")
    warnings: List[str] = Field([], description="Список предупреждений")

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
                result_data = json.load(f)
        except Exception as e:
            logger.error(f"Ошибка при чтении результата: {str(e)}")
    
    return templates.TemplateResponse(
        "result.html", 
        {
            "request": request, 
            "job_id": job_id, 
            "status": status,
            "result": result_data
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
    Запускает оптимизацию плана проекта
    """
    try:
        # Получаем идентификатор задачи
        job_id = optimize_plan(
            data=request.data,
            duration_weight=request.duration_weight,
            resource_weight=request.resource_weight,
            cost_weight=request.cost_weight,
            num_episodes=request.num_episodes,
            use_pretrained_model=request.use_pretrained_model,
            model_path=request.model_path,
            background_tasks=background_tasks
        )
        
        return {"job_id": job_id, "status": "pending"}
    
    except Exception as e:
        logger.exception("Ошибка при запуске оптимизации")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

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
    use_pretrained_model: bool = Form(False)
):
    """
    Загружает файл и запускает оптимизацию
    """
    try:
        # Сохраняем файл во временную директорию
        temp_file_path = os.path.join(TEMP_DIR, file.filename)
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
        
        # Валидируем JSON
        is_valid, errors, warnings = validate_input_json(temp_file_path)
        
        if not is_valid:
            return templates.TemplateResponse(
                "index.html", 
                {
                    "request": request,
                    "errors": errors,
                    "warnings": warnings
                }
            )
        
        # Загружаем данные из файла
        with open(temp_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Выбираем модель, если требуется
        model_path = None
        if use_pretrained_model:
            # Здесь нужно реализовать выбор модели
            # Пока используем заглушку
            model_path = None
        
        # Запускаем оптимизацию
        job_id = optimize_plan(
            data=data,
            duration_weight=duration_weight,
            resource_weight=resource_weight,
            cost_weight=cost_weight,
            num_episodes=num_episodes,
            use_pretrained_model=use_pretrained_model,
            model_path=model_path,
            background_tasks=background_tasks
        )
        
        # Перенаправляем на страницу с результатом
        return RedirectResponse(url=f"/result/{job_id}", status_code=303)
    
    except Exception as e:
        logger.exception("Ошибка при загрузке файла")
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "errors": [f"Ошибка при загрузке файла: {str(e)}"],
                "warnings": []
            }
        )

# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True) 