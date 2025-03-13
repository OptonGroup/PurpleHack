/**
 * Основной JavaScript файл для веб-приложения "Оптимизатор календарного плана"
 */

// Функция для переключения отображения параметров алгоритма имитации отжига
function toggleAlgorithmParams() {
    const algorithmSelect = document.getElementById('algorithm');
    const annealingParams = document.getElementById('annealing-params');
    const rlParams = document.getElementById('rl-params');
    
    // Информационные карточки
    const rlInfoCard = document.getElementById('rl-info-card');
    const saInfoCard = document.getElementById('sa-info-card');
    
    if (algorithmSelect && annealingParams && rlParams) {
        if (algorithmSelect.value === 'simulated_annealing') {
            // Показываем параметры SA и скрываем параметры RL
            annealingParams.style.display = 'block';
            rlParams.style.display = 'none';
            
            // Показываем карточку SA и скрываем карточку RL
            if (rlInfoCard) rlInfoCard.style.display = 'none';
            if (saInfoCard) saInfoCard.style.display = 'block';
        } else {
            // Показываем параметры RL и скрываем параметры SA
            annealingParams.style.display = 'none';
            rlParams.style.display = 'block';
            
            // Показываем карточку RL и скрываем карточку SA
            if (rlInfoCard) rlInfoCard.style.display = 'block';
            if (saInfoCard) saInfoCard.style.display = 'none';
        }
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // Инициализация всплывающих подсказок Bootstrap
    var tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    if (tooltips.length > 0) {
        Array.from(tooltips).map(tooltip => new bootstrap.Tooltip(tooltip));
    }
    
    // Инициализация переключателя параметров алгоритма
    const algorithmSelect = document.getElementById('algorithm');
    if (algorithmSelect) {
        // Устанавливаем начальное состояние
        toggleAlgorithmParams();
        
        // Обработчик события изменения
        algorithmSelect.addEventListener('change', toggleAlgorithmParams);
    }

    // Форма загрузки файла
    var uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function() {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Загрузка...';
            }
        });
    }

    // Загрузка JSON через API
    const jsonUploadForm = document.getElementById('json-api-form');
    if (jsonUploadForm) {
        jsonUploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const fileInput = this.querySelector('input[type="file"]');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Пожалуйста, выберите файл.');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            fetch('/api/validate', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.is_valid) {
                    // Если файл валидный, отправляем на оптимизацию
                    const algorithm = document.getElementById('algorithm').value;
                    
                    // Основные параметры оптимизации
                    const optimizationParams = {
                        duration_weight: parseFloat(document.getElementById('duration_weight').value),
                        resource_weight: parseFloat(document.getElementById('resource_weight').value),
                        cost_weight: parseFloat(document.getElementById('cost_weight').value),
                        use_pretrained_model: document.getElementById('use_pretrained_model').checked,
                        algorithm: algorithm
                    };
                    
                    // Если выбран RL, добавляем его параметры
                    if (algorithm === 'reinforcement_learning') {
                        optimizationParams.num_episodes = parseInt(document.getElementById('num_episodes').value);
                        // Добавляем новые параметры RL
                        if (document.getElementById('learning_rate')) {
                            optimizationParams.learning_rate = parseFloat(document.getElementById('learning_rate').value);
                        }
                        if (document.getElementById('gamma')) {
                            optimizationParams.gamma = parseFloat(document.getElementById('gamma').value);
                        }
                    }
                    // Если выбран алгоритм имитации отжига, добавляем его параметры
                    else if (algorithm === 'simulated_annealing') {
                        optimizationParams.initial_temperature = parseFloat(document.getElementById('initial_temperature').value);
                        optimizationParams.cooling_rate = parseFloat(document.getElementById('cooling_rate').value);
                        optimizationParams.min_temperature = parseFloat(document.getElementById('min_temperature').value);
                        optimizationParams.iterations_per_temp = parseInt(document.getElementById('iterations_per_temp').value);
                        optimizationParams.max_iterations = parseInt(document.getElementById('max_iterations').value);
                    }
                    
                    // Чтение файла как JSON
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const fileContent = JSON.parse(e.target.result);
                        
                        // Отправка на оптимизацию
                        fetch('/api/optimize', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                data: fileContent,
                                ...optimizationParams
                            })
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.job_id) {
                                // Перенаправление на страницу с результатом
                                window.location.href = `/result/${data.job_id}`;
                            } else {
                                alert('Ошибка при запуске оптимизации');
                            }
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            alert('Ошибка при отправке данных: ' + error);
                        });
                    };
                    
                    reader.readAsText(file);
                } else {
                    // Если файл не валидный, показываем ошибки
                    let errorMessage = 'Файл содержит ошибки:\n';
                    data.errors.forEach(error => {
                        errorMessage += `- ${error}\n`;
                    });
                    
                    if (data.warnings.length > 0) {
                        errorMessage += '\nПредупреждения:\n';
                        data.warnings.forEach(warning => {
                            errorMessage += `- ${warning}\n`;
                        });
                    }
                    
                    alert(errorMessage);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Ошибка валидации файла: ' + error);
            });
        });
    }

    // Функция для проверки статуса оптимизации
    function checkOptimizationStatus(jobId) {
        fetch(`/api/status/${jobId}`)
            .then(response => response.json())
            .then(data => {
                // Обновляем информацию о статусе
                const statusElement = document.getElementById('optimization-status');
                if (statusElement) {
                    if (data.status === 'completed') {
                        statusElement.innerHTML = '<span class="badge bg-success">Завершена</span>';
                        // Перезагружаем страницу для отображения результатов
                        window.location.reload();
                    } else if (data.status === 'failed') {
                        statusElement.innerHTML = '<span class="badge bg-danger">Ошибка</span>';
                        if (data.error) {
                            document.getElementById('error-message').textContent = data.error;
                            document.getElementById('error-container').classList.remove('d-none');
                        }
                    } else {
                        // Продолжаем проверять статус
                        setTimeout(() => checkOptimizationStatus(jobId), 3000);
                    }
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
    }

    // Запускаем проверку статуса, если находимся на странице результата
    const jobIdElement = document.getElementById('job-id');
    if (jobIdElement) {
        const jobId = jobIdElement.dataset.jobId;
        const status = jobIdElement.dataset.status;
        
        if (status === 'running' || status === 'pending') {
            checkOptimizationStatus(jobId);
        }
    }
}); 