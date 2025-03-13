/**
 * Скрипт для управления индикатором загрузки и отображения прогресса оптимизации
 */

class ProgressLoader {
    constructor(jobId) {
        this.jobId = jobId;
        this.isRunning = false;
        this.updateInterval = 2000; // Интервал обновления в миллисекундах
        this.progressBar = document.getElementById('progress-bar');
        this.progressPercentage = document.getElementById('progress-percentage');
        this.loadingText = document.getElementById('loading-text');
        this.currentAction = document.getElementById('current-action');
        this.intervalId = null;
        
        // Создаем элемент для отображения времени
        this.timeInfoElement = document.createElement('div');
        this.timeInfoElement.className = 'mt-3 text-muted';
        this.timeInfoElement.id = 'time-info';
        
        // Добавляем его после процентов прогресса
        if (this.progressPercentage && this.progressPercentage.parentNode) {
            this.progressPercentage.parentNode.insertBefore(
                this.timeInfoElement, 
                this.progressPercentage.nextSibling
            );
        }
        
        // Сообщения о действиях, которые выполняются во время оптимизации
        this.actionMessages = [
            "Анализ зависимостей задач...",
            "Построение графа зависимостей...",
            "Оптимизация распределения ресурсов...",
            "Расчет критического пути...",
            "Минимизация длительности проекта...",
            "Балансировка нагрузки ресурсов...",
            "Устранение конфликтов в расписании...",
            "Финальная оптимизация..."
        ];
    }
    
    /**
     * Запуск процесса отслеживания прогресса
     */
    startTracking() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        
        // Первичное обновление без задержки
        this.updateProgress();
        
        // Запускаем периодическое обновление
        this.intervalId = setInterval(() => this.updateProgress(), this.updateInterval);
    }
    
    /**
     * Остановка процесса отслеживания прогресса
     */
    stopTracking() {
        if (!this.isRunning) return;
        
        this.isRunning = false;
        
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }
    
    /**
     * Обновление сообщения о текущем действии в зависимости от прогресса
     * @param {number} progress - Процент выполнения (0-100)
     */
    updateProgressMessage(progress) {
        if (!this.currentAction) return;
        
        const actionIndex = Math.floor((progress / 100) * (this.actionMessages.length - 1));
        this.currentAction.textContent = this.actionMessages[actionIndex];
    }
    
    /**
     * Обновление текста в зависимости от прогресса
     * @param {number} progress - Процент выполнения (0-100)
     */
    updateLoadingText(progress) {
        if (!this.loadingText) return;
        
        if (progress > 75) {
            this.loadingText.textContent = "Завершающий этап оптимизации...";
        } else if (progress > 50) {
            this.loadingText.textContent = "Оптимизация в процессе...";
        } else if (progress > 25) {
            this.loadingText.textContent = "Оптимизация активно выполняется...";
        }
    }
    
    /**
     * Обновление визуального отображения прогресса
     * @param {number} progress - Процент выполнения (0-100)
     */
    updateVisualProgress(progress) {
        if (this.progressBar) {
            this.progressBar.style.width = `${progress}%`;
            this.progressBar.setAttribute('aria-valuenow', progress);
        }
        
        if (this.progressPercentage) {
            this.progressPercentage.textContent = `${progress}%`;
        }
    }
    
    /**
     * Обновление информации о времени
     * @param {string} elapsedTime - Прошедшее время
     * @param {string} remainingTime - Оставшееся время
     */
    updateTimeInfo(elapsedTime, remainingTime) {
        if (!this.timeInfoElement) return;
        
        let timeInfo = `<i class="bi bi-clock"></i> Прошло: ${elapsedTime || 'менее секунды'}`;
        
        if (remainingTime) {
            timeInfo += ` • Осталось: ${remainingTime}`;
        }
        
        this.timeInfoElement.innerHTML = timeInfo;
    }
    
    /**
     * Запрос статуса оптимизации и обновление интерфейса
     */
    updateProgress() {
        fetch(`/api/status/${this.jobId}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === "pending" || data.status === "running") {
                    // Обновляем прогресс-бар
                    const progress = Math.round(data.progress * 100);
                    
                    this.updateVisualProgress(progress);
                    this.updateProgressMessage(progress);
                    this.updateLoadingText(progress);
                    
                    // Обновляем информацию о времени
                    this.updateTimeInfo(data.elapsed_time, data.estimated_time_remaining);
                    
                    // Если прогресс близок к завершению, уменьшаем интервал обновления
                    if (progress > 90 && this.updateInterval > 1000) {
                        this.stopTracking();
                        this.updateInterval = 1000;
                        this.startTracking();
                    }
                } else {
                    // Если статус изменился на "completed" или "failed", перезагружаем страницу
                    this.stopTracking();
                    window.location.reload();
                }
            })
            .catch(error => {
                console.error('Ошибка при проверке статуса:', error);
            });
    }
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    const jobIdElement = document.getElementById('job-id');
    
    if (jobIdElement) {
        const jobId = jobIdElement.dataset.jobId;
        const status = jobIdElement.dataset.status;
        
        // Если задача в процессе выполнения, запускаем отслеживание прогресса
        if (status === 'pending' || status === 'running') {
            const loader = new ProgressLoader(jobId);
            loader.startTracking();
        }
    }
}); 