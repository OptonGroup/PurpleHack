// Глобальные переменные
let anomalyData = []; // Массив для хранения данных аномалий
let confirmedAnomalies = []; // Подтвержденные аномалии
let potentialAnomalies = []; // Потенциальные аномалии
let currentPage = 1;
let itemsPerPage = 10;
let filteredData = []; // Отфильтрованные данные

// DOM элементы
const totalAnomaliesEl = document.getElementById('totalAnomalies');
const confirmedHacksEl = document.getElementById('confirmedHacks');
const potentialHacksEl = document.getElementById('potentialHacks');
const anomalyTableBody = document.getElementById('anomalyTableBody');
const pageIndicator = document.getElementById('pageIndicator');
const prevPageBtn = document.getElementById('prevPage');
const nextPageBtn = document.getElementById('nextPage');
const searchInput = document.getElementById('searchInput');
const sortSelect = document.getElementById('sortSelect');
const userTypeFilter = document.getElementById('userTypeFilter');
const statusFilter = document.getElementById('statusFilter');
const trafficFilter = document.getElementById('trafficFilter');
const trafficValue = document.getElementById('trafficValue');
const applyFiltersBtn = document.getElementById('applyFilters');
const refreshBtn = document.getElementById('refreshBtn');
const detailModal = document.getElementById('detailModal');
const closeModal = document.querySelector('.close-modal');

// Иконка для модального окна
const shieldIconSvg = `<svg width="30" height="30" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 2L4 5V11.09C4 16.14 7.41 20.85 12 22C16.59 20.85 20 16.14 20 11.09V5L12 2Z" stroke="#4a6baf" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>`;

// Создаем SVG иконку для использования
document.addEventListener('DOMContentLoaded', function() {
    // Создаем SVG иконку
    const iconContainer = document.createElement('div');
    iconContainer.innerHTML = shieldIconSvg;
    const logoImg = document.querySelector('.logo');
    logoImg.replaceWith(iconContainer.firstChild);
    
    // Инициализация приложения
    init();
    
    // Событие для кнопки обновления
    refreshBtn.addEventListener('click', loadData);
    
    // События для пагинации
    prevPageBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            renderTable();
        }
    });
    
    nextPageBtn.addEventListener('click', () => {
        const maxPage = Math.ceil(filteredData.length / itemsPerPage);
        if (currentPage < maxPage) {
            currentPage++;
            renderTable();
        }
    });
    
    // События для фильтрации
    searchInput.addEventListener('input', () => {
        currentPage = 1;
        applyFilters();
    });
    
    sortSelect.addEventListener('change', () => {
        currentPage = 1;
        applyFilters();
    });
    
    applyFiltersBtn.addEventListener('click', () => {
        currentPage = 1;
        applyFilters();
    });
    
    trafficFilter.addEventListener('input', () => {
        const value = trafficFilter.value;
        if (value === '0') {
            trafficValue.textContent = 'Все';
        } else {
            trafficValue.textContent = `> ${value}`;
        }
    });
    
    // Обработка модального окна
    closeModal.addEventListener('click', () => {
        detailModal.style.display = 'none';
    });
    
    window.addEventListener('click', (event) => {
        if (event.target === detailModal) {
            detailModal.style.display = 'none';
        }
    });
    
    // Обработчики для кнопок в модальном окне
    document.getElementById('confirmAnomaly').addEventListener('click', () => {
        const anomalyId = document.getElementById('detailId').textContent;
        confirmAnomaly(anomalyId);
    });
    
    document.getElementById('falsePositive').addEventListener('click', () => {
        const anomalyId = document.getElementById('detailId').textContent;
        markAsFalsePositive(anomalyId);
    });
    
    document.getElementById('investigateMore').addEventListener('click', () => {
        // В реальном приложении здесь может быть код для запуска дополнительного расследования
        alert('Запущено дополнительное расследование');
    });
});

// Инициализация приложения
function init() {
    // Загрузка данных
    loadData();
    
    // Инициализация графика тенденций
    initTrendChart();
}

// Загрузка данных
async function loadData() {
    try {
        // Эмуляция загрузки данных из CSV файлов
        await loadConfirmedAnomalies();
        await loadPotentialAnomalies();
        
        // Объединение данных
        anomalyData = [...confirmedAnomalies, ...potentialAnomalies];
        
        // Обновление статистики
        updateStats();
        
        // Применение фильтров и отображение таблицы
        applyFilters();
        
        // Обновление графика
        updateTrendChart();
    } catch (error) {
        console.error('Ошибка при загрузке данных:', error);
    }
}

// Эмуляция загрузки подтвержденных аномалий
async function loadConfirmedAnomalies() {
    // В реальном приложении здесь был бы код для загрузки данных из already_hacked.csv
    // Эмуляция данных для демонстрации
    confirmedAnomalies = [
        { Id: '100001', UID: 'USER12345', Type: 'P', IdPlan: 'BASIC', TurnOn: true, Hacked: true, Traffic: 15.7, status: 'confirmed', timestamp: new Date('2023-06-01') },
        { Id: '100002', UID: 'USER12346', Type: 'J', IdPlan: 'PREMIUM', TurnOn: true, Hacked: true, Traffic: 25.3, status: 'confirmed', timestamp: new Date('2023-06-02') },
        { Id: '100003', UID: 'USER12347', Type: 'P', IdPlan: 'STANDARD', TurnOn: true, Hacked: true, Traffic: 12.8, status: 'confirmed', timestamp: new Date('2023-06-03') },
        { Id: '100004', UID: 'USER12348', Type: 'J', IdPlan: 'BUSINESS', TurnOn: true, Hacked: true, Traffic: 45.2, status: 'confirmed', timestamp: new Date('2023-06-04') },
        { Id: '100005', UID: 'USER12349', Type: 'P', IdPlan: 'BASIC', TurnOn: true, Hacked: true, Traffic: 8.5, status: 'confirmed', timestamp: new Date('2023-06-05') },
    ];
    
    return confirmedAnomalies;
}

// Эмуляция загрузки потенциальных аномалий
async function loadPotentialAnomalies() {
    // В реальном приложении здесь был бы код для загрузки данных из probably_hacked.csv
    // Эмуляция данных для демонстрации
    potentialAnomalies = [
        { Id: '200001', UID: 'USER23456', Type: 'P', IdPlan: 'BASIC', TurnOn: true, Hacked: false, Traffic: 10.2, status: 'potential', timestamp: new Date('2023-06-06') },
        { Id: '200002', UID: 'USER23457', Type: 'J', IdPlan: 'PREMIUM', TurnOn: true, Hacked: false, Traffic: 18.9, status: 'potential', timestamp: new Date('2023-06-07') },
        { Id: '200003', UID: 'USER23458', Type: 'P', IdPlan: 'STANDARD', TurnOn: false, Hacked: false, Traffic: 7.3, status: 'potential', timestamp: new Date('2023-06-08') },
        { Id: '200004', UID: 'USER23459', Type: 'J', IdPlan: 'BUSINESS', TurnOn: true, Hacked: false, Traffic: 30.1, status: 'potential', timestamp: new Date('2023-06-09') },
        { Id: '200005', UID: 'USER23460', Type: 'P', IdPlan: 'BASIC', TurnOn: true, Hacked: false, Traffic: 9.7, status: 'potential', timestamp: new Date('2023-06-10') },
        { Id: '200006', UID: 'USER23461', Type: 'J', IdPlan: 'PREMIUM', TurnOn: true, Hacked: false, Traffic: 22.4, status: 'potential', timestamp: new Date('2023-06-11') },
        { Id: '200007', UID: 'USER23462', Type: 'P', IdPlan: 'STANDARD', TurnOn: true, Hacked: false, Traffic: 14.6, status: 'potential', timestamp: new Date('2023-06-12') },
        { Id: '200008', UID: 'USER23463', Type: 'J', IdPlan: 'BUSINESS', TurnOn: false, Hacked: false, Traffic: 35.8, status: 'potential', timestamp: new Date('2023-06-13') },
    ];
    
    return potentialAnomalies;
}

// Обновление статистики
function updateStats() {
    totalAnomaliesEl.textContent = anomalyData.length;
    confirmedHacksEl.textContent = confirmedAnomalies.length;
    potentialHacksEl.textContent = potentialAnomalies.length;
}

// Применение фильтров
function applyFilters() {
    const searchTerm = searchInput.value.toLowerCase();
    const userType = userTypeFilter.value;
    const status = statusFilter.value;
    const traffic = parseInt(trafficFilter.value) || 0;
    
    // Фильтрация данных
    filteredData = anomalyData.filter(item => {
        // Поиск по ID и UID
        const matchesSearch = item.Id.toLowerCase().includes(searchTerm) || 
                              item.UID.toLowerCase().includes(searchTerm);
        
        // Фильтр по типу пользователя
        const matchesUserType = userType === 'all' || item.Type === userType;
        
        // Фильтр по статусу
        const matchesStatus = status === 'all' || 
                             (status === 'confirmed' && item.status === 'confirmed') ||
                             (status === 'potential' && item.status === 'potential');
        
        // Фильтр по трафику
        const matchesTraffic = traffic === 0 || item.Traffic > traffic;
        
        return matchesSearch && matchesUserType && matchesStatus && matchesTraffic;
    });
    
    // Сортировка данных
    const sortOption = sortSelect.value;
    
    switch (sortOption) {
        case 'time_desc':
            filteredData.sort((a, b) => b.timestamp - a.timestamp);
            break;
        case 'time_asc':
            filteredData.sort((a, b) => a.timestamp - b.timestamp);
            break;
        case 'traffic_desc':
            filteredData.sort((a, b) => b.Traffic - a.Traffic);
            break;
        case 'traffic_asc':
            filteredData.sort((a, b) => a.Traffic - b.Traffic);
            break;
    }
    
    // Отображение отфильтрованных данных
    renderTable();
}

// Отображение таблицы с аномалиями
function renderTable() {
    // Очистка таблицы
    anomalyTableBody.innerHTML = '';
    
    // Вычисление данных для текущей страницы
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const currentPageData = filteredData.slice(startIndex, endIndex);
    
    // Создание строк таблицы
    currentPageData.forEach(item => {
        const row = document.createElement('tr');
        
        // ID сессии
        const idCell = document.createElement('td');
        idCell.textContent = item.Id;
        row.appendChild(idCell);
        
        // UID
        const uidCell = document.createElement('td');
        uidCell.textContent = item.UID;
        row.appendChild(uidCell);
        
        // Тип
        const typeCell = document.createElement('td');
        typeCell.textContent = item.Type === 'P' ? 'Физ. лицо' : 'Юр. лицо';
        row.appendChild(typeCell);
        
        // Тарифный план
        const planCell = document.createElement('td');
        planCell.textContent = item.IdPlan;
        row.appendChild(planCell);
        
        // Статус
        const statusCell = document.createElement('td');
        const statusSpan = document.createElement('span');
        statusSpan.textContent = item.status === 'confirmed' ? 'Подтвержден' : 'Потенциальный';
        statusSpan.className = item.status === 'confirmed' ? 'status-confirmed' : 'status-potential';
        statusCell.appendChild(statusSpan);
        row.appendChild(statusCell);
        
        // Трафик
        const trafficCell = document.createElement('td');
        trafficCell.textContent = `${item.Traffic.toFixed(1)} Мб/с`;
        row.appendChild(trafficCell);
        
        // Действия
        const actionCell = document.createElement('td');
        actionCell.className = 'action-cell';
        
        const viewButton = document.createElement('button');
        viewButton.textContent = 'Детали';
        viewButton.className = 'view-button';
        viewButton.addEventListener('click', () => showDetails(item));
        actionCell.appendChild(viewButton);
        
        if (item.status === 'potential') {
            const confirmButton = document.createElement('button');
            confirmButton.textContent = 'Подтвердить';
            confirmButton.className = 'confirm-button';
            confirmButton.addEventListener('click', () => confirmAnomaly(item.Id));
            actionCell.appendChild(confirmButton);
            
            const rejectButton = document.createElement('button');
            rejectButton.textContent = 'Отклонить';
            rejectButton.className = 'reject-button';
            rejectButton.addEventListener('click', () => markAsFalsePositive(item.Id));
            actionCell.appendChild(rejectButton);
        }
        
        row.appendChild(actionCell);
        
        // Добавление строки в таблицу
        anomalyTableBody.appendChild(row);
    });
    
    // Обновление индикатора страниц
    const maxPage = Math.ceil(filteredData.length / itemsPerPage);
    pageIndicator.textContent = `Страница ${currentPage} из ${maxPage || 1}`;
    
    // Включение/отключение кнопок пагинации
    prevPageBtn.disabled = currentPage === 1;
    nextPageBtn.disabled = currentPage >= maxPage;
}

// Инициализация графика тенденций
function initTrendChart() {
    const ctx = document.getElementById('anomalyTrendChart').getContext('2d');
    
    window.anomalyTrendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Подтвержденные аномалии',
                    data: [],
                    backgroundColor: 'rgba(40, 167, 69, 0.2)',
                    borderColor: 'rgba(40, 167, 69, 1)',
                    borderWidth: 2,
                    tension: 0.3
                },
                {
                    label: 'Потенциальные аномалии',
                    data: [],
                    backgroundColor: 'rgba(255, 193, 7, 0.2)',
                    borderColor: 'rgba(255, 193, 7, 1)',
                    borderWidth: 2,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Дата'
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Количество аномалий'
                    },
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}

// Обновление графика тенденций
function updateTrendChart() {
    // Группировка данных по дате
    const dateGroups = {};
    
    // Добавление данных в группы
    anomalyData.forEach(item => {
        const dateKey = item.timestamp.toISOString().split('T')[0];
        
        if (!dateGroups[dateKey]) {
            dateGroups[dateKey] = { confirmed: 0, potential: 0 };
        }
        
        if (item.status === 'confirmed') {
            dateGroups[dateKey].confirmed++;
        } else {
            dateGroups[dateKey].potential++;
        }
    });
    
    // Сортировка дат
    const sortedDates = Object.keys(dateGroups).sort();
    
    // Подготовка данных для графика
    const confirmedData = [];
    const potentialData = [];
    
    sortedDates.forEach(date => {
        confirmedData.push(dateGroups[date].confirmed);
        potentialData.push(dateGroups[date].potential);
    });
    
    // Обновление данных графика
    window.anomalyTrendChart.data.labels = sortedDates.map(date => {
        const [year, month, day] = date.split('-');
        return `${day}.${month}.${year}`;
    });
    window.anomalyTrendChart.data.datasets[0].data = confirmedData;
    window.anomalyTrendChart.data.datasets[1].data = potentialData;
    window.anomalyTrendChart.update();
}

// Показать детали аномалии
function showDetails(item) {
    // Заполнение данных в модальном окне
    document.getElementById('detailId').textContent = item.Id;
    document.getElementById('detailUid').textContent = item.UID;
    document.getElementById('detailType').textContent = item.Type === 'P' ? 'Физическое лицо' : 'Юридическое лицо';
    document.getElementById('detailPlan').textContent = item.IdPlan;
    document.getElementById('detailStatus').textContent = item.TurnOn ? 'Активен' : 'Не активен';
    document.getElementById('detailTraffic').textContent = `${item.Traffic.toFixed(1)} Мб/с`;
    
    // Создание графика поведения
    initBehaviorChart(item);
    
    // Отображение модального окна
    detailModal.style.display = 'block';
}

// Инициализация графика поведения
function initBehaviorChart(item) {
    // Удаление существующего графика
    const canvas = document.getElementById('behaviorChart');
    const ctx = canvas.getContext('2d');
    
    // Создание нового графика
    // Здесь мы симулируем данные поведения для демонстрации
    // В реальном приложении нужно загружать исторические данные пользователя
    const normalData = generateNormalBehaviorData();
    const anomalyData = generateAnomalyData(normalData, item);
    
    const behaviorChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Нормальное поведение',
                    data: normalData,
                    backgroundColor: 'rgba(0, 123, 255, 0.5)',
                    pointRadius: 5
                },
                {
                    label: 'Аномальное поведение',
                    data: anomalyData,
                    backgroundColor: 'rgba(220, 53, 69, 0.7)',
                    pointRadius: 7
                },
                {
                    label: 'Текущая сессия',
                    data: [{
                        x: item.Traffic,
                        y: Math.random() * 20 + 20 // Симуляция второго параметра
                    }],
                    backgroundColor: 'rgba(255, 193, 7, 1)',
                    pointRadius: 8,
                    pointStyle: 'triangle'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Средний трафик (Мб/с)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Продолжительность сессий (мин)'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = context.dataset.data[context.dataIndex];
                            return `Трафик: ${point.x.toFixed(1)} Мб/с, Длительность: ${point.y.toFixed(1)} мин`;
                        }
                    }
                }
            }
        }
    });
}

// Генерация данных нормального поведения
function generateNormalBehaviorData() {
    const data = [];
    
    // Симуляция кластера нормального поведения
    const centerX = 10; // Центр по оси X
    const centerY = 15; // Центр по оси Y
    
    for (let i = 0; i < 20; i++) {
        data.push({
            x: centerX + (Math.random() - 0.5) * 5,
            y: centerY + (Math.random() - 0.5) * 10
        });
    }
    
    return data;
}

// Генерация данных аномального поведения
function generateAnomalyData(normalData, item) {
    const data = [];
    
    // Симуляция кластера аномального поведения
    // Мы используем значение трафика из переданного элемента
    const centerX = item.Traffic * 0.9; // Центр по оси X, близкий к значению трафика
    const centerY = 35; // Центр по оси Y (выше, чем у нормального поведения)
    
    for (let i = 0; i < 5; i++) {
        data.push({
            x: centerX + (Math.random() - 0.5) * 5,
            y: centerY + (Math.random() - 0.5) * 5
        });
    }
    
    return data;
}

// Подтверждение аномалии
function confirmAnomaly(id) {
    // Находим аномалию по ID
    const index = anomalyData.findIndex(item => item.Id === id);
    
    if (index !== -1) {
        // Обновляем статус аномалии
        anomalyData[index].status = 'confirmed';
        
        // Обновляем списки
        potentialAnomalies = potentialAnomalies.filter(item => item.Id !== id);
        confirmedAnomalies.push(anomalyData[index]);
        
        // Обновляем статистику и таблицу
        updateStats();
        applyFilters();
        updateTrendChart();
        
        // Закрываем модальное окно, если оно открыто
        detailModal.style.display = 'none';
        
        // Уведомление пользователя
        alert(`Аномалия ${id} подтверждена`);
    }
}

// Отметка аномалии как ложное срабатывание
function markAsFalsePositive(id) {
    // Находим аномалию по ID
    const index = anomalyData.findIndex(item => item.Id === id);
    
    if (index !== -1) {
        // Удаляем аномалию из списков
        anomalyData = anomalyData.filter(item => item.Id !== id);
        potentialAnomalies = potentialAnomalies.filter(item => item.Id !== id);
        confirmedAnomalies = confirmedAnomalies.filter(item => item.Id !== id);
        
        // Обновляем статистику и таблицу
        updateStats();
        applyFilters();
        updateTrendChart();
        
        // Закрываем модальное окно, если оно открыто
        detailModal.style.display = 'none';
        
        // Уведомление пользователя
        alert(`Аномалия ${id} отмечена как ложное срабатывание`);
    }
} 