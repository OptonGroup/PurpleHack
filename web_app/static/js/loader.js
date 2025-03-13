/**
 * Упрощенный скрипт для отслеживания прогресса оптимизации
 */

// Глобальные переменные для элементов интерфейса
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-percentage');
const loadingText = document.getElementById('loading-text');
const currentAction = document.getElementById('current-action');
const debugConsole = document.getElementById('debug-console');

// Сообщения о действиях на разных этапах оптимизации
const actionMessages = [
    "Анализ зависимостей задач...",
    "Построение графа зависимостей...",
    "Оптимизация распределения ресурсов...",
    "Расчет критического пути...",
    "Минимизация длительности проекта...",
    "Балансировка нагрузки ресурсов...",
    "Устранение конфликтов в расписании...",
    "Финальная оптимизация..."
];

/**
 * Добавляет сообщение в консоль отладки
 * @param {string} message - Текст сообщения
 */
function logMessage(message) {
    if (!debugConsole) return;
    
    const logEntry = document.createElement('div');
    logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    debugConsole.appendChild(logEntry);
    debugConsole.scrollTop = debugConsole.scrollHeight;
    console.log(message);
}

/**
 * Обновляет визуальное отображение прогресса
 * @param {number} progress - Значение прогресса от 0 до 1
 */
function updateProgress(progress) {
    const percent = Math.round(progress * 100);
    
    logMessage(`Обновление прогресса: ${percent}%`);
    
    // Обновляем прогресс-бар
    if (progressBar) {
        progressBar.style.width = `${percent}%`;
        progressBar.setAttribute('aria-valuenow', percent);
    }
    
    // Обновляем текст с процентами
    if (progressText) {
        progressText.textContent = `${percent}%`;
    }
    
    // Обновляем сообщение о текущем действии
    if (currentAction) {
        const actionIndex = Math.min(
            Math.floor(progress * actionMessages.length),
            actionMessages.length - 1
        );
        currentAction.textContent = actionMessages[actionIndex];
    }
    
    // Обновляем текст загрузки
    if (loadingText) {
        if (percent > 75) {
            loadingText.textContent = "Завершающий этап оптимизации...";
        } else if (percent > 50) {
            loadingText.textContent = "Оптимизация в процессе...";
        } else if (percent > 25) {
            loadingText.textContent = "Оптимизация активно выполняется...";
        }
    }
}

/**
 * Запрашивает статус оптимизации и обновляет интерфейс
 * @param {string} jobId - Идентификатор задачи оптимизации
 */
function checkStatus(jobId) {
    logMessage(`Запрос статуса для задачи: ${jobId}`);
    
    fetch(`/api/status/${jobId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ошибка! статус: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            logMessage(`Получен ответ: status=${data.status}, progress=${data.progress}`);
            
            if (data.status === "pending" || data.status === "running") {
                // Обновляем прогресс
                updateProgress(data.progress);
                
                // Продолжаем проверку статуса
                setTimeout(() => checkStatus(jobId), 2000);
            } else {
                // Если статус изменился на "completed" или "failed"
                logMessage(`Статус изменился на: ${data.status}, перезагружаем страницу...`);
                window.location.reload();
            }
        })
        .catch(error => {
            logMessage(`Ошибка при запросе статуса: ${error.message}`);
            // В случае ошибки пробуем снова через 5 секунд
            setTimeout(() => checkStatus(jobId), 5000);
        });
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    logMessage('Инициализация отслеживания статуса...');
    
    // Находим элемент с идентификатором задачи
    const jobIdElement = document.getElementById('job-id');
    
    if (jobIdElement) {
        const jobId = jobIdElement.dataset.jobId;
        const status = jobIdElement.dataset.status;
        
        logMessage(`ID задачи: ${jobId}, статус: ${status}`);
        
        if (status === 'pending' || status === 'running') {
            // Начинаем отслеживание статуса
            logMessage('Начинаем отслеживание прогресса...');
            checkStatus(jobId);
        } else {
            logMessage(`Задача в статусе: ${status}, отслеживание не требуется`);
        }
    } else {
        logMessage('Элемент с ID задачи не найден!');
    }
}); 