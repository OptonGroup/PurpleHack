# Прогресс по задаче "Обнаружение аномалий в данных о потреблении интернет-трафика" (Arenadata)

## Описание задачи

Разработка системы для обнаружения нетипичного потребления интернет-трафика абонентами компании-оператора связи с целью определения взлома оборудования абонента. Необходимо построить витрины данных с интервалом расчета 1 час и составить список взломанных абонентов.

## Статус проекта

- [ ]  Готов первый прототип датасета - переход к выбору архитектуры и постановке задачи с точки зрения обучения
## Что сделали

- 1: Сгруппировал данные и получил представление одного файла по активности трафика взаимосвязанно со всеми сущностями 
- 2: Произвел анализ сущностей и нашел основыне закономерности, которые будут представлены позже
- 3: Создал первый прототип датасета на котором уже можно рабоать с задачей


## заметки

- Идея 1: Смотреть разность исходящих и входящих трафиков
- Идея 2: Смотреть относительное значение трафиков (относительно длительности)
- Идея 3: Обучить автоэкодер для восставновления исходящего и входящего трафика скалированных значение, учитвая план и нагрузку коммутатора - выбросы детектить как аномалию
