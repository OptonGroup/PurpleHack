#!/usr/bin/env python3
import json
import sys
from typing import Dict, List, Tuple, Set, Any, Optional
import math
import copy
import time
import traceback

# Константы для оптимизации
TARGET_UTILIZATION = 0.807197  # Оптимальная утилизация для максимального балла
UPPER_THRESHOLD = 0.85  # Верхний порог утилизации
LOWER_THRESHOLD = 0.15  # Нижний порог утилизации (уменьшен для более агрессивной консолидации)
MAX_MIGRATIONS = 3  # Максимальное количество миграций за раунд (увеличено для лучшей консолидации)
MIN_BENEFIT_FOR_MIGRATION = 20  # Минимальная выгода для оправдания миграции (уменьшена для более активных миграций)
CONSOLIDATION_THRESHOLD = 0.15  # Порог для консолидации VM с хостов с низкой утилизацией (уменьшен)
BONUS_THRESHOLD = 5  # Количество раундов с нулевой утилизацией для получения бонуса
CACHE_STATS = {"hits": 0, "misses": 0}  # Статистика кэша

def calculate_reward(utilization: float) -> float:
    """Вычисляет награду за утилизацию хоста по формуле из документации."""
    if utilization <= 0:
        return 0
    if utilization >= 1:
        return 2.992622
    
    x = utilization
    return -0.67459 + (42.38075/(-2.5*x+5.96)) * math.exp(-2*(math.log(-2.5*x+2.96))**2)

class VMScheduler:
    def __init__(self):
        """Инициализирует планировщик VM."""
        # Данные о хостах и VM
        self.hosts = {}  # dict: {host_id: {cpu, ram, ...}}
        self.vms = {}    # dict: {vm_id: {cpu, ram, ...}}
        
        # Кэш для расчетов вместимости хостов
        self.capacity_cache = {}  # dict: {cache_key: {cpu, ram}}
        self.cache_stats = {"hits": 0, "misses": 0}
        
        # Размещения VM по хостам
        self.previous_allocations = {}  # dict: {host_id: [vm_ids]}
        self.vm_to_host_map = {}  # dict: {vm_id: host_id}
        
        # Множество хостов с предыдущими VM
        self.hosts_with_previous_vms = set()
        
        # Счетчик хостов с нулевой утилизацией
        self.host_zero_utilization_count = {}  # dict: {host_id: count}
        
        # Список хостов, готовых к отключению
        self.hosts_ready_for_shutdown = []
        
        # Список хостов для выключения
        self.hosts_to_shutdown = []
        
        # Ошибки размещения и миграции
        self.allocation_failures = []  # list: [vm_ids]
        self.migrations = []  # list: [{vm_id, source, target}]
        
        # Отслеживание недавних миграций
        self.recent_migrations = set()  # set: {vm_ids}
        
        # Отслеживание миграций в текущем раунде
        self.current_round_migrations = set()  # set: {vm_ids}
        
        # Счетчик раундов
        self.round_counter = 0
        
        # Статистика производительности
        self.performance_stats = {
            "round_time": [],
            "migrations": []
        }
        
        # Константы для оптимизации
        self.TARGET_UTILIZATION = TARGET_UTILIZATION
        self.UPPER_THRESHOLD = UPPER_THRESHOLD
        self.LOWER_THRESHOLD = LOWER_THRESHOLD
        self.MAX_MIGRATIONS = MAX_MIGRATIONS
        self.MIN_BENEFIT_FOR_MIGRATION = MIN_BENEFIT_FOR_MIGRATION
        self.CONSOLIDATION_THRESHOLD = CONSOLIDATION_THRESHOLD
        self.BONUS_THRESHOLD = BONUS_THRESHOLD
        
    def _get_allocation_key(self, host_id: str, allocations: Dict[str, List[str]] = None) -> str:
        """Генерирует ключ для кэширования на основе хоста и его размещений."""
        if allocations is None:
            host_vms = []
        elif isinstance(allocations, dict):
            host_vms = sorted(allocations.get(host_id, []))
        else:
            host_vms = sorted(allocations)
            
        # Добавляем характеристики хоста и VM в ключ
        key_parts = [
            f"host_{host_id}",
            f"cpu_{self.hosts[host_id]['cpu']}",
            f"ram_{self.hosts[host_id]['ram']}"
        ]
        
        for vm_id in host_vms:
            if vm_id in self.vms:
                key_parts.extend([
                    f"vm_{vm_id}",
                    f"cpu_{self.vms[vm_id]['cpu']}",
                    f"ram_{self.vms[vm_id]['ram']}"
                ])
        
        return "|".join(key_parts)
        
    def calculate_host_capacity(self, host_id: str, allocations: Dict[str, List[str]] = None) -> Dict:
        """Вычисляет остаточные ресурсы хоста и его утилизацию на основе текущих размещений VM."""
        # Генерируем ключ для кэша
        cache_key = self._get_allocation_key(host_id, allocations)
        
        # Проверяем кэш
        if cache_key in self.capacity_cache:
            # Статистика кэша
            CACHE_STATS["hits"] += 1
            return self.capacity_cache[cache_key]
        
        # Статистика кэша
        CACHE_STATS["misses"] += 1
        
        if host_id not in self.hosts:
            result = {"capacity": {"cpu": 0, "ram": 0}, "utilization": {"cpu": 0, "ram": 0}, "max_utilization": 0}
            self.capacity_cache[cache_key] = result
            return result
        
        host = self.hosts[host_id]
        total_capacity = {
            "cpu": host.get("cpu", 0),
            "ram": host.get("ram", 0)
        }
        
        # Используем предоставленные аллокации или текущие
        allocs = allocations if allocations is not None else self.previous_allocations
        
        # Вычитаем ресурсы, используемые виртуальными машинами
        allocated_vms = allocs.get(host_id, [])
        used_resources = {"cpu": 0, "ram": 0}
            
        for vm_id in allocated_vms:
            if vm_id in self.vms:
                vm = self.vms[vm_id]
                used_resources["cpu"] += vm.get("cpu", 0)
                used_resources["ram"] += vm.get("ram", 0)
        
        # Вычисляем остаточные ресурсы и утилизацию
        remaining_capacity = {
            "cpu": total_capacity["cpu"] - used_resources["cpu"],
            "ram": total_capacity["ram"] - used_resources["ram"]
        }
        
        utilization = {
            "cpu": used_resources["cpu"] / total_capacity["cpu"] if total_capacity["cpu"] > 0 else 0,
            "ram": used_resources["ram"] / total_capacity["ram"] if total_capacity["ram"] > 0 else 0
        }
        
        max_utilization = max(utilization["cpu"], utilization["ram"])
        
        result = {
            "capacity": remaining_capacity,
            "utilization": utilization,
            "max_utilization": max_utilization
        }
        
        # Сохраняем результат в кэш
        self.capacity_cache[cache_key] = result
        
        return result
        
    def clear_capacity_cache(self):
        """Очищает кэш вычисленных емкостей."""
        self.capacity_cache = {}
        # Обновляем статистику кэша
        self.performance_stats["cache_hits"] = CACHE_STATS["hits"]
        self.performance_stats["cache_misses"] = CACHE_STATS["misses"]
        CACHE_STATS["hits"] = 0
        CACHE_STATS["misses"] = 0

    def process_input(self, input_json: Dict) -> Dict:
        """Обрабатывает входные данные и возвращает размещение VM.
        
        Параметры:
        - input_json: входные данные в формате JSON
        
        Возвращает:
        - Dict: ответ с размещениями, миграциями и ошибками размещения
        """
        try:
            # Увеличиваем счетчик раундов
            self.round_counter += 1
            
            # Очищаем списки миграций и ошибок размещения
            self.migrations = []
            self.allocation_failures = []
            
            # Очищаем кэш расчета вместимости хостов
            self.capacity_cache = {}
            self.cache_stats = {"hits": 0, "misses": 0}
            
            # Загружаем данные
            self.load_data(input_json)
            # print(f"Data loaded. Hosts: {len(self.hosts)}, VMs: {len(self.vms)}", file=sys.stderr)
            
            # Определяем список VM, которые нужно разместить
            vms_to_place = []
            diff = input_json.get("diff", {})
            
            if "add" in diff:
                added_vms = diff["add"].get("virtual_machines", [])
                vms_to_place.extend(added_vms)
            
            # Получаем список всех хостов
            all_hosts = list(self.hosts.keys())
            
            # Размещаем VM
            new_allocations = self.place_vms(vms_to_place, all_hosts, self.previous_allocations)
            
            # Рассчитываем новые размещения VM по хостам
            self.previous_allocations = new_allocations
            
            # Обновляем отображение VM на хосты
            self.vm_to_host_map = {vm: host_id for host_id, vm_list in new_allocations.items() for vm in vm_list}
            
            # Консолидируем VM, если нет миграций
            if not self.migrations and self.round_counter > 10:  # Начинаем консолидацию после 10 раундов
                self.consolidate_vms(self.previous_allocations)
            
            # Обновляем счетчики утилизации
            self.update_utilization_counters()
            
            # Формируем ответ
            response = {
                "allocations": {host_id: vms for host_id, vms in self.previous_allocations.items() if vms},
                "migrations": self.migrations,
                "allocation_failures": self.allocation_failures
            }
            
            # Печатаем статистику кэша
            # print(f"Cache stats: {self.cache_stats}", file=sys.stderr)
            
            return response
            
        except Exception as e:
            # print(f"Error processing data: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # Возвращаем пустой ответ в случае ошибки
            return {
                "allocations": {},
                "migrations": [],
                "allocation_failures": []
            }

    def load_data(self, data: Dict) -> None:
        """Загружает входные данные."""
        try:
            # Загружаем хосты
            if "hosts" in data:
                self.hosts = data.get("hosts", {})
                
            # Загружаем виртуальные машины
            self.vms = data.get("virtual_machines", {})
            
            # Обрабатываем изменения
            diff = data.get("diff", {})
            
            # Добавляем новые VM
            if "add" in diff:
                new_vms = diff["add"].get("virtual_machines", [])
                # Сбрасываем список неразмещенных VM при добавлении новых
                self.allocation_failures = []
                # Добавляем новые VM в список для размещения
                self.vms_to_place = new_vms
            else:
                self.vms_to_place = []
            
            # Удаляем VM
            if "remove" in diff:
                removed_vms = diff["remove"].get("virtual_machines", [])
                # Удаляем VM из предыдущих размещений
                for host_id in list(self.previous_allocations.keys()):
                    self.previous_allocations[host_id] = [
                        vm for vm in self.previous_allocations.get(host_id, [])
                        if vm not in removed_vms
                    ]
            
            # Обновляем vm_to_host_map на основе предыдущих размещений
            self.vm_to_host_map.clear()
            for host_id, vm_list in self.previous_allocations.items():
                for vm_id in vm_list:
                    if vm_id in self.vms:  # Проверяем, что VM все еще существует
                        self.vm_to_host_map[vm_id] = host_id
            
            # Обновляем множество хостов с предыдущими VM
            self.hosts_with_previous_vms = {
                host_id for host_id, vm_list in self.previous_allocations.items()
                if vm_list and host_id in self.hosts
            }
            
            # Инициализируем счетчик хостов с нулевой утилизацией при необходимости
            if not self.host_zero_utilization_count:
                self.host_zero_utilization_count = {host_id: 0 for host_id in self.hosts}
            
            # Обновляем счетчики хостов с нулевой утилизацией
            for host_id in self.hosts:
                host_vms = self.previous_allocations.get(host_id, [])
                if not host_vms:
                    self.host_zero_utilization_count[host_id] = self.host_zero_utilization_count.get(host_id, 0) + 1
                else:
                    self.host_zero_utilization_count[host_id] = 0
            
            # Обновляем список хостов для выключения
            self.update_hosts_to_shutdown()
            
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            raise

    def consolidate_vms(self, allocations):
        """
        Консолидирует VM на минимальном количестве хостов для экономии ресурсов.
        
        Args:
            allocations: текущие размещения VM по хостам
        """
        # Если нет хостов или VM, нечего консолидировать
        if not self.hosts or not self.vms:
            return allocations, []
            
        # Рассчитываем текущую утилизацию хостов
        hosts_utilization = {}
        for host_id in self.hosts:
            host_vms = allocations.get(host_id, [])
            if not host_vms:
                hosts_utilization[host_id] = 0.0
                continue
            
            # Рассчитываем утилизацию хоста
            host_capacity = self.calculate_host_capacity(host_id, allocations)
            hosts_utilization[host_id] = host_capacity["max_utilization"]
        
        # Выводим статистику по утилизации хостов
        host_utilization_stats = [(host_id, util) for host_id, util in hosts_utilization.items()]
        host_utilization_stats.sort(key=lambda x: x[1], reverse=True)
        # print(f"Host utilization ranking: {host_utilization_stats}", file=sys.stderr)
        
        # Добавляем хосты с низкой утилизацией в список для выключения
        for host_id, utilization in hosts_utilization.items():
            if 0 < utilization < LOWER_THRESHOLD:
                # print(f"Adding host {host_id} to shutdown list due to low utilization ({utilization:.2f})", file=sys.stderr)
                self.hosts_to_shutdown.add(host_id)
        
        # Если нет хостов для выключения, нечего консолидировать
        if not self.hosts_to_shutdown:
            return allocations, []
            
        # Выводим информацию о хостах с нулевой утилизацией
    
    def _place_vms_by_type(self, vm_categories: Dict[str, List[str]], 
                         host_categories: Dict[str, List[str]], priority: str) -> None:
        """Размещает VM по категориям хостов.
        
        Параметры:
        - vm_categories: категории VM {категория: [список VM]}
        - host_categories: категории хостов {категория: [список хостов]}
        - priority: приоритет размещения
        """
        # Сначала размещаем VM с высоким приоритетом
        for vm_id in vm_categories.get("high_priority", []):
            # Находим наиболее подходящий хост
            best_host = self._find_best_host_for_vm(vm_id, host_categories)
            if best_host:
                if best_host not in self.previous_allocations:
                    self.previous_allocations[best_host] = []
                self.previous_allocations[best_host].append(vm_id)
                # print(f"Placed high priority VM {vm_id} on host {best_host}", file=sys.stderr)
        
        # Затем размещаем VM со средним приоритетом
        for vm_id in vm_categories.get("medium_priority", []):
            best_host = self._find_best_host_for_vm(vm_id, host_categories)
            if best_host:
                if best_host not in self.previous_allocations:
                    self.previous_allocations[best_host] = []
                self.previous_allocations[best_host].append(vm_id)
                # print(f"Placed medium priority VM {vm_id} on host {best_host}", file=sys.stderr)
        
        # Размещаем CPU-интенсивные VM на CPU-оптимизированных хостах
        self._place_category_vms("cpu_heavy", "cpu_optimized", vm_categories, host_categories)
        
        # Размещаем RAM-интенсивные VM на RAM-оптимизированных хостах
        self._place_category_vms("ram_heavy", "ram_optimized", vm_categories, host_categories)
        
        # Размещаем Storage-интенсивные VM на Storage-оптимизированных хостах
        self._place_category_vms("storage_heavy", "storage_optimized", vm_categories, host_categories)
        
        # Размещаем смешанные VM на сбалансированных хостах
        self._place_category_vms("mixed", "balanced", vm_categories, host_categories)
        
        # Размещаем оставшиеся VM на любых доступных хостах
        remaining_vms = []
        for category, vm_list in vm_categories.items():
            if category not in ["high_priority", "medium_priority"]:
                for vm_id in vm_list:
                    # Проверяем, была ли VM уже размещена
                    if not any(vm_id in vms for vms in self.previous_allocations.values()):
                        remaining_vms.append(vm_id)
        
        if remaining_vms:
            # print(f"Placing {len(remaining_vms)} remaining VMs", file=sys.stderr)
            all_hosts = [host for hosts in host_categories.values() for host in hosts]
            new_allocations = self.place_vms(remaining_vms, all_hosts, self.previous_allocations)
            
            # Обновляем размещения
            for host_id, vms in new_allocations.items():
                if host_id not in self.previous_allocations:
                    self.previous_allocations[host_id] = []
                self.previous_allocations[host_id].extend([vm for vm in vms if vm not in self.previous_allocations[host_id]])
    
    def _place_category_vms(self, vm_category: str, host_category: str, 
                          vm_categories: Dict[str, List[str]], 
                          host_categories: Dict[str, List[str]]) -> None:
        """Размещает VM определенной категории на хостах определенной категории.
        
        Параметры:
        - vm_category: категория VM
        - host_category: категория хостов
        - vm_categories: категории VM {категория: [список VM]}
        - host_categories: категории хостов {категория: [список хостов]}
        """
        vms = vm_categories.get(vm_category, [])
        hosts = host_categories.get(host_category, [])
        
        if not vms or not hosts:
            return
            
        # print(f"Placing {len(vms)} {vm_category} VMs on {len(hosts)} {host_category} hosts", file=sys.stderr)
        
        # Сортируем VM по размеру (сначала большие)
        vms.sort(key=lambda vm: self._calculate_vm_size(vm), reverse=True)
        
        # Создаем текущие размещения для этой категории
        category_allocations = {host_id: self.previous_allocations.get(host_id, []) for host_id in hosts}
        
        # Размещаем VM
        new_allocations = self.place_vms(vms, hosts, category_allocations)
        
        # Обновляем основные размещения
        for host_id, vm_list in new_allocations.items():
            if host_id not in self.previous_allocations:
                self.previous_allocations[host_id] = []
            
            # Добавляем только те VM, которые относятся к текущей категории и не были размещены ранее
            for vm in vm_list:
                if vm in vms and vm not in self.previous_allocations[host_id]:
                    self.previous_allocations[host_id].append(vm)
    
    def _find_best_host_for_vm(self, vm_id: str, host_categories: Dict[str, List[str]]) -> Optional[str]:
        """Находит наиболее подходящий хост для VM с учетом категорий хостов.
        
        Параметры:
        - vm_id: идентификатор VM
        - host_categories: категории хостов {категория: [список хостов]}
        
        Возвращает:
        - str или None: идентификатор хоста или None, если хост не найден
        """
        vm_info = self.vms.get(vm_id, {})
        
        # Определяем категорию VM
        vm_category = None
        if "cpu" in vm_id or "compute" in vm_id or (vm_info.get("cpu", 0) > vm_info.get("ram", 0) * 1.5):
            vm_category = "cpu_optimized"
        elif "ram" in vm_id or "memory" in vm_id or (vm_info.get("ram", 0) > vm_info.get("cpu", 0) * 1.5):
            vm_category = "ram_optimized"
        elif "storage" in vm_id or "disk" in vm_id:
            vm_category = "storage_optimized"
        else:
            vm_category = "balanced"
        
        # Сначала проверяем хосты в соответствующей категории
        if vm_category in host_categories:
            for host_id in host_categories[vm_category]:
                if self.can_host_vm(host_id, vm_id):
                    return host_id
        
        # Если не нашли подходящий хост в предпочтительной категории, проверяем все хосты
        all_hosts = [host for hosts in host_categories.values() for host in hosts]
        for host_id in all_hosts:
            if self.can_host_vm(host_id, vm_id):
                return host_id
        
        return None
    
    def _calculate_vm_size(self, vm_id: str) -> float:
        """Рассчитывает "размер" VM на основе ее ресурсов.
        
        Параметры:
        - vm_id: идентификатор VM
        
        Возвращает:
        - float: оценка размера VM
        """
        vm_info = self.vms.get(vm_id, {})
        cpu = vm_info.get("cpu", 0)
        ram = vm_info.get("ram", 0)
        
        # Нормализация размера с учетом среднего CPU и RAM всех VM
        avg_cpu = sum(vm.get("cpu", 0) for vm in self.vms.values()) / max(1, len(self.vms))
        avg_ram = sum(vm.get("ram", 0) for vm in self.vms.values()) / max(1, len(self.vms))
        
        normalized_cpu = cpu / max(1, avg_cpu)
        normalized_ram = ram / max(1, avg_ram)
        
        # Размер как максимум из нормализованных значений
        return max(normalized_cpu, normalized_ram)
    
    def _calculate_vm_resource_usage(self, vm_id: str, resource_or_priority) -> float:
        """Рассчитывает использование ресурса VM или оценку по приоритету.
        
        Параметры:
        - vm_id: идентификатор VM
        - resource_or_priority: ресурс ("cpu", "ram", "disk") или приоритет ("balanced", "performance")
        
        Возвращает:
        - float: оценка использования ресурса
        """
        vm_info = self.vms.get(vm_id, {})
        
        if isinstance(resource_or_priority, list):
            # Если передан список ресурсов, считаем сумму
            return sum(vm_info.get(res, 0) for res in resource_or_priority)
        
        if resource_or_priority in ["cpu", "ram", "disk"]:
            return vm_info.get(resource_or_priority, 0)
        
        # Приоритеты размещения
        if resource_or_priority == "balanced":
            cpu = vm_info.get("cpu", 0)
            ram = vm_info.get("ram", 0)
            return cpu + ram
        elif resource_or_priority == "performance":
            return vm_info.get("cpu", 0) * 2
        elif resource_or_priority == "memory":
            return vm_info.get("ram", 0) * 2
        
        # По умолчанию возвращаем общую оценку
        return sum(value for key, value in vm_info.items() if key in ["cpu", "ram", "disk"])

    def update_hosts_to_shutdown(self):
        """Обновляет список хостов, которые следует выключить для получения бонусов."""
        # Оцениваем необходимое количество хостов
        required_hosts = self.estimate_required_hosts()
        # print(f"Estimated required hosts: {required_hosts} out of {len(self.hosts)}", file=sys.stderr)  # Закомментировано
        
        # Хосты, которые уже близки к получению бонуса (4+ раунда с нулевой утилизацией)
        almost_bonus_hosts = {
            host_id for host_id, count in self.host_zero_utilization_count.items()
            if count >= 4 and host_id in self.hosts_with_previous_vms
        }
        # print(f"Hosts close to bonus: {almost_bonus_hosts}", file=sys.stderr)  # Закомментировано
        
        # Хосты с низкой утилизацией, которые можно выключить
        low_utilization_hosts = []
        for host_id in self.hosts:
            if host_id in self.previous_allocations and self.previous_allocations[host_id]:
                state = self.calculate_host_capacity(host_id)
                if state["max_utilization"] < LOWER_THRESHOLD:
                    low_utilization_hosts.append((host_id, state["max_utilization"]))
        
        # Сортируем по утилизации (сначала самые низкие)
        low_utilization_hosts.sort(key=lambda x: x[1])
        
        # Определяем, сколько хостов можно выключить
        hosts_to_shutdown_count = max(0, len(self.hosts) - required_hosts)
        
        # Приоритет для выключения:
        # 1. Хосты, близкие к получению бонуса
        # 2. Хосты с самой низкой утилизацией
        self.hosts_to_shutdown = set()
        
        # Добавляем хосты, близкие к бонусу
        self.hosts_to_shutdown.update(almost_bonus_hosts)
        
        # Добавляем хосты с низкой утилизацией, если нужно больше
        for host_id, util in low_utilization_hosts:
            if len(self.hosts_to_shutdown) < hosts_to_shutdown_count and host_id not in self.hosts_to_shutdown:
                self.hosts_to_shutdown.add(host_id)
        
        # print(f"Hosts to shutdown: {self.hosts_to_shutdown}", file=sys.stderr)  # Закомментировано

    def can_host_vm(self, host_id: str, vm_id: str, allocations: Dict[str, List[str]] = None, existing_capacity: Dict[str, int] = None) -> bool:
        """Проверяет, может ли хост разместить VM.
        
        Args:
            host_id: ID хоста
            vm_id: ID VM
            allocations: текущие размещения (опционально)
            existing_capacity: существующая емкость хоста (опционально)
        
        Returns:
            bool: True если хост может разместить VM, иначе False
        """
        # Получаем информацию о VM
        vm = self.vms[vm_id]
        
        # Получаем текущую емкость хоста
        if existing_capacity:
            # Используем переданную емкость
            capacity = existing_capacity
        else:
            # Вычисляем текущую емкость
            host_state = self.calculate_host_capacity(host_id, allocations)
            capacity = host_state["capacity"]
        
        # Проверяем, достаточно ли ресурсов
        return (capacity["cpu"] >= vm["cpu"] and 
                capacity["ram"] >= vm["ram"])

    def get_migrations(self, new_allocations: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """Определяет необходимые миграции VM между хостами."""
        migrations = []
        self.current_round_migrations = set()
        
        # Создаем отображение VM -> Host для новых размещений (оптимизировано)
        new_vm_to_host = {}
        for host_id, vm_list in new_allocations.items():
            for vm_id in vm_list:
                new_vm_to_host[vm_id] = host_id
        
        # Сначала пытаемся консолидировать VM с хостов, которые планируется выключить
        consolidated_allocations, consolidation_migrations = self.consolidate_vms(new_allocations)
        
        # Счетчики для статистики
        successful_migrations = 0
        cancelled_migrations = 0
        
        # Если уже достигли лимита миграций, возвращаем результат
        if len(consolidation_migrations) >= MAX_MIGRATIONS:
            # Обновляем размещения и возвращаем миграции
            for host_id, vms in consolidated_allocations.items():
                new_allocations[host_id] = vms
            
            # Обновляем статистику
            successful_migrations += len(consolidation_migrations)
            self.performance_stats["successful_migrations"].append(successful_migrations)
            self.performance_stats["cancelled_migrations"].append(cancelled_migrations)
            
            # Добавляем миграции в историю
            self.migration_history.extend(consolidation_migrations)
            
            # Обновляем список недавно мигрировавших VM
            self.recent_migrations = set(migration["vm"] for migration in self.migration_history[-3*MAX_MIGRATIONS:])
            
            # Ограничиваем размер истории миграций
            if len(self.migration_history) > 50:
                self.migration_history = self.migration_history[-50:]
                
            # print(f"Reached migration limit after consolidation, returning {len(consolidation_migrations)} migrations", file=sys.stderr)
            return consolidation_migrations
            
        # Добавляем миграции от консолидации
        migrations.extend(consolidation_migrations)
        for migration in consolidation_migrations:
            self.current_round_migrations.add(migration["vm"])
        
        # Если уже есть миграции, ограничиваем количество дополнительных миграций
        remaining_migrations = MAX_MIGRATIONS - len(migrations)
        if remaining_migrations <= 0:
            # Обновляем статистику
            successful_migrations += len(migrations)
            self.performance_stats["successful_migrations"].append(successful_migrations)
            self.performance_stats["cancelled_migrations"].append(cancelled_migrations)
            
            # Добавляем миграции в историю
            self.migration_history.extend(migrations)
            
            # Обновляем список недавно мигрировавших VM
            self.recent_migrations = set(migration["vm"] for migration in self.migration_history[-3*MAX_MIGRATIONS:])
            
            # Ограничиваем размер истории миграций
            if len(self.migration_history) > 50:
                self.migration_history = self.migration_history[-50:]
                
            return migrations
            
        # print(f"After consolidation: {len(migrations)} migrations, {remaining_migrations} remaining", file=sys.stderr)
        
        # Быстрая проверка необходимости миграций
        potential_migrations = []
        for vm_id, new_host in new_vm_to_host.items():
            old_host = self.vm_to_host_map.get(vm_id)
            if old_host and old_host != new_host:
                # Пропускаем VM, которые уже мигрировали
                if vm_id in self.current_round_migrations:
                    continue
                    
                # Пропускаем миграции на хосты, которые планируется выключить
                if new_host in self.hosts_to_shutdown:
                    # print(f"Skipping migration of VM {vm_id} to host {new_host} as it's targeted for shutdown", file=sys.stderr)
                    continue
                
                # Проверяем, что VM не мигрировала недавно (в последних 3 раундах)
                recently_migrated = False
                for migration in self.migration_history[-3*MAX_MIGRATIONS:]:
                    if migration.get("vm") == vm_id:
                        recently_migrated = True
                        break
                
                if recently_migrated:
                    # print(f"Skipping migration of VM {vm_id} as it was recently migrated", file=sys.stderr)
                    continue
                
                # Используем новый метод для расчета реальной выгоды от миграции
                # с учетом квадратичного штрафа и прогнозирования долгосрочной выгоды
                current_migration_count = len(migrations) + 1  # +1 для текущей миграции
                migration_benefit = self.calculate_migration_benefit(
                    vm_id, old_host, new_host, consolidated_allocations, current_migration_count
                )
                
                # Учитываем долгосрочную выгоду от миграции
                long_term_benefit = self.predict_long_term_benefit(
                    old_host, new_host, vm_id, consolidated_allocations
                )
                
                # Суммарная выгода
                total_benefit = migration_benefit + long_term_benefit
                
                # Если выгода от миграции превышает порог, добавляем ее в список потенциальных
                if total_benefit > MIN_BENEFIT_FOR_MIGRATION:
                    potential_migrations.append((vm_id, old_host, new_host, total_benefit))
        
        # Сортируем потенциальные миграции по выгоде (от большей к меньшей)
        potential_migrations.sort(key=lambda x: x[3], reverse=True)
        
        # Выбираем лучшие миграции в пределах лимита
        best_migrations = potential_migrations[:remaining_migrations]
        
        # print(f"Potential migrations: {len(potential_migrations)}, best: {len(best_migrations)}", file=sys.stderr)
        
        # Применяем выбранные миграции
        for vm_id, old_host, new_host, benefit in best_migrations:
            # Проверяем, что VM еще не мигрировала
            if vm_id in self.current_round_migrations:
                continue
                
            # Проверяем, что миграция все еще возможна
            if not self.can_host_vm(new_host, vm_id):
                cancelled_migrations += 1
                # print(f"Migration of VM {vm_id} from {old_host} to {new_host} cancelled - target host cannot host VM", file=sys.stderr)
                continue
                
            # Обновляем размещения
            if vm_id in consolidated_allocations[old_host]:
                consolidated_allocations[old_host].remove(vm_id)
            if new_host not in consolidated_allocations:
                consolidated_allocations[new_host] = []
            consolidated_allocations[new_host].append(vm_id)
            
            # Добавляем миграцию в результат
            migrations.append({"vm": vm_id, "source_host": old_host, "target_host": new_host})
            self.current_round_migrations.add(vm_id)
            successful_migrations += 1
            
            # Логируем детали миграции для дебага
            # print(f"Migration: {vm_id} from {old_host} to {new_host}, benefit: {benefit:.2f}", file=sys.stderr)
        
        # Обновляем размещения
        for host_id, vms in consolidated_allocations.items():
            new_allocations[host_id] = vms
        
        # Обновляем статистику
        self.performance_stats["successful_migrations"].append(successful_migrations)
        self.performance_stats["cancelled_migrations"].append(cancelled_migrations)
        
        # Добавляем миграции в историю
        self.migration_history.extend(migrations)
        
        # Обновляем список недавно мигрировавших VM
        self.recent_migrations = set(migration["vm"] for migration in self.migration_history[-3*MAX_MIGRATIONS:])
        
        # Ограничиваем размер истории миграций
        if len(self.migration_history) > 50:
            self.migration_history = self.migration_history[-50:]
        
        return migrations

    def consolidate_vms(self, allocations):
        """
        Консолидирует VM на минимальном количестве хостов для экономии ресурсов.
        
        Args:
            allocations: текущие размещения VM по хостам
        """
        # Если нет хостов или VM, нечего консолидировать
        if not self.hosts or not self.vms:
            return
            
        # Рассчитываем текущую утилизацию хостов
        hosts_utilization = {}
        for host_id in self.hosts:
            host_vms = allocations.get(host_id, [])
            if not host_vms:
                hosts_utilization[host_id] = 0.0
                continue
            
            # Рассчитываем утилизацию хоста
            host_capacity = self.calculate_host_capacity(host_id, allocations)
            hosts_utilization[host_id] = host_capacity["max_utilization"]
        
        # Выводим статистику по утилизации хостов
        host_utilization_stats = [(host_id, util) for host_id, util in hosts_utilization.items()]
        host_utilization_stats.sort(key=lambda x: x[1], reverse=True)
        # print(f"Host utilization ranking: {host_utilization_stats}", file=sys.stderr)
        
        # Добавляем хосты с низкой утилизацией в список для выключения
        for host_id, utilization in hosts_utilization.items():
            if 0 < utilization < LOWER_THRESHOLD:
                # print(f"Adding host {host_id} to shutdown list due to low utilization ({utilization:.2f})", file=sys.stderr)
                self.hosts_to_shutdown.add(host_id)
        
        # Если нет хостов для выключения, нечего консолидировать
        if not self.hosts_to_shutdown:
            return
            
        # Выводим информацию о хостах с нулевой утилизацией
        hosts_with_zero_util = {host_id: self.host_zero_utilization_count.get(host_id, 0) 
                               for host_id in self.hosts 
                               if hosts_utilization.get(host_id, 0) == 0}
        # print(f"Hosts with zero utilization counts: {hosts_with_zero_util}", file=sys.stderr)
        
        # Приоритизируем хосты для выключения
        host_shutdown_priority = []
        for host_id in self.hosts:
            # Пропускаем хосты с нулевой утилизацией (они уже выключены)
            if hosts_utilization.get(host_id, 0) == 0:
                continue
            
            # Рассчитываем приоритет выключения
            # Чем ниже утилизация и чем ближе к бонусу за выключение, тем выше приоритет
            utilization = hosts_utilization.get(host_id, 0)
            vm_count = len(allocations.get(host_id, []))
            
            # Базовый приоритет - обратно пропорционален утилизации
            priority_score = 30  # Базовый приоритет
            
            # Если утилизация низкая, увеличиваем приоритет
            if utilization < LOWER_THRESHOLD:
                priority_score = 100 - (utilization * 100)  # Чем ниже утилизация, тем выше приоритет
                
            host_shutdown_priority.append((host_id, vm_count, utilization, priority_score))
        
        # Сортируем хосты по приоритету выключения (сначала с наивысшим приоритетом)
        host_shutdown_priority.sort(key=lambda x: x[3], reverse=True)
        # print(f"Hosts prioritized for shutdown: {host_shutdown_priority}", file=sys.stderr)
        
        # Определяем минимальное количество хостов, необходимое для размещения всех VM
        total_vms = sum(len(vms) for vms in allocations.values())
        avg_vms_per_host = max(1, total_vms / max(1, len(self.hosts)))
        required_hosts = max(1, math.ceil(total_vms / avg_vms_per_host))
        
        # Добавляем буфер для обеспечения запаса мощности
        buffer = max(1, required_hosts // 4)  # 25% буфер
        min_hosts = required_hosts + buffer
        
        # Определяем, сколько хостов сейчас активно
        active_hosts = sum(1 for host_id in self.hosts if hosts_utilization.get(host_id, 0) > 0)
        
        # Определяем, нужно ли агрессивно консолидировать
        aggressive_mode = active_hosts > min_hosts
        
        # print(f"Required hosts: {required_hosts}, Buffer: {buffer}, Active hosts: {active_hosts}, Aggressive mode: {aggressive_mode}", file=sys.stderr)
        
        # Если достигли минимального количества хостов, останавливаемся
        if active_hosts <= min_hosts:
            # print(f"Reached minimum required hosts ({min_hosts})", file=sys.stderr)
            return
            
        # Инициализируем список миграций
        migrations = []
        
        # Обрабатываем хосты в порядке приоритета выключения
        for host_id, vm_count, utilization, priority in host_shutdown_priority:
            # Если достигли минимального количества хостов, останавливаемся
            if active_hosts <= min_hosts:
                break
                
            # Если хост не в списке для выключения, пропускаем
            if host_id not in self.hosts_to_shutdown and not aggressive_mode:
                continue
                
            # Если достигли лимита миграций, останавливаемся
            if len(migrations) >= MAX_MIGRATIONS:
                break
                
            # Получаем список VM на этом хосте
            vms_on_host = allocations.get(host_id, [])
            
            # Если хост пуст, пропускаем
            if not vms_on_host:
                continue
            
            # Флаг, указывающий, удалось ли мигрировать все VM с хоста
            all_vms_migrated = True
            
            # Сортируем VM по размеру (сначала маленькие, их легче мигрировать)
            sorted_vms = sorted(
                [(vm_id, self.vms[vm_id].get("cpu", 0) + self.vms[vm_id].get("ram", 0)) 
                 for vm_id in vms_on_host if vm_id in self.vms],
                key=lambda x: x[1]
            )
            
            # Пытаемся мигрировать каждую VM с этого хоста
            for vm_id, vm_size in sorted_vms:
                # Пропускаем VM, которые уже мигрировали в этом раунде
                if vm_id in self.current_round_migrations:
                    continue
                    
                # Если достигли лимита миграций, останавливаем цикл
                if len(migrations) >= MAX_MIGRATIONS:
                    all_vms_migrated = False
                    break
                    
                # Находим лучший хост для VM (избегая хостов для выключения)
                best_host = self._find_best_host_for_placement(
                    vm_id, 
                    [h for h in self.hosts if h not in self.hosts_to_shutdown],
                    allocations,
                    hosts_utilization
                )
                
                # Если нашли подходящий хост, мигрируем VM
                if best_host and best_host != host_id:
                    # Проверяем, что миграция улучшит общую ситуацию
                    # Для хостов с высоким приоритетом выключения выполняем миграцию без доп. проверок
                    benefit_threshold = MIN_BENEFIT_FOR_MIGRATION / 2  # Используем половину стандартного порога для консолидации
                    
                    # Для хостов с очень высоким приоритетом или близких к бонусу снижаем порог еще больше
                    if priority > 50 or self.host_zero_utilization_count.get(host_id, 0) >= 3:
                        benefit_threshold = 0  # Мигрируем в любом случае
                    
                    # Добавляем миграцию
                    migrations.append({
                        "vm": vm_id,
                        "source_host": host_id,
                        "destination_host": best_host
                    })
                    
                    # Обновляем размещения
                    if host_id in allocations:
                        allocations[host_id].remove(vm_id)
                    if best_host not in allocations:
                        allocations[best_host] = []
                    allocations[best_host].append(vm_id)
                    
                    # Обновляем утилизацию
                    hosts_utilization[host_id] = self.calculate_host_capacity(host_id, allocations)["max_utilization"]
                    hosts_utilization[best_host] = self.calculate_host_capacity(best_host, allocations)["max_utilization"]
                    
                    # Отмечаем VM как мигрированную в этом раунде
                    self.current_round_migrations.add(vm_id)
                    
                    # Добавляем в список недавних миграций
                    self.recent_migrations.add(vm_id)
                else:
                    # Если не удалось найти подходящий хост, отмечаем, что не все VM мигрированы
                    all_vms_migrated = False
            
            # Если все VM мигрированы, уменьшаем количество активных хостов
            if all_vms_migrated and not allocations.get(host_id, []):
                active_hosts -= 1
        
        # Обновляем список миграций
        self.migrations.extend(migrations)

    def calculate_host_score(self, host_id: str, allocations: Dict[str, List[str]] = None) -> float:
        """Вычисляет оценку хоста на основе его утилизации."""
        state = self.calculate_host_capacity(host_id, allocations)
        utilization = state["max_utilization"]
        
        # Базовая оценка за утилизацию
        score = calculate_reward(utilization)
        
        # Бонус за выключенный хост
        if utilization == 0:
            # Проверяем, использовался ли хост ранее
            if host_id in self.hosts_with_previous_vms:
                zero_count = self.host_zero_utilization_count.get(host_id, 0)
                # Бонус начисляется после 5+ раундов простоя
                if zero_count >= BONUS_THRESHOLD:
                    score += 8  # Бонус за выключенный хост
                    # print(f"Host {host_id} gets bonus for being off for {zero_count} rounds", file=sys.stderr)  # Закомментировано
                # Частичный бонус для хостов, близких к получению полного бонуса
                elif zero_count >= BONUS_THRESHOLD - 1:
                    score += 4  # Частичный бонус для стимулирования сохранения хоста выключенным
        
        # Штраф за утилизацию выше верхнего порога
        if utilization > UPPER_THRESHOLD:
            penalty = (utilization - UPPER_THRESHOLD) * 5
            score -= penalty
            
        return score
        
    def calculate_total_score(self, allocations: Dict[str, List[str]], migrations: List[Dict[str, str]]) -> float:
        """Вычисляет общую оценку для данного размещения с учетом всех факторов."""
        # Вычисляем утилизацию хостов
        host_utilization = {}
        for host_id in self.hosts:
            state = self.calculate_host_capacity(host_id, allocations)
            max_utilization = state["max_utilization"]
            host_utilization[host_id] = max_utilization
            
        # Вычисляем оценку за утилизацию
        utilization_score = 0
        host_scores = {}
        for host_id, utilization in host_utilization.items():
            host_score = calculate_reward(utilization)
            host_scores[host_id] = host_score
            utilization_score += host_score
            
        # Штраф за миграции (квадратичный)
        migration_penalty = 0
        if migrations:
            migration_count = len(migrations)
            migration_penalty = -(migration_count ** 2)
            
        # Штраф за невозможность размещения
        allocation_failure_penalty = 0
        if self.allocation_failures:
            allocation_failure_penalty = -5 * len(self.hosts) * len(self.allocation_failures)
        
        # Бонус за выключенные хосты
        shutdown_bonus = 0
        shutdown_hosts = []
        for host_id in self.hosts:
            if host_id in self.hosts_with_previous_vms and host_utilization.get(host_id, 0) == 0:
                if self.host_zero_utilization_count.get(host_id, 0) >= BONUS_THRESHOLD:
                    shutdown_bonus += 8  # Бонус за хост с нулевой утилизацией 5+ раундов
                    shutdown_hosts.append(host_id)
        
        total_score = utilization_score + migration_penalty + allocation_failure_penalty + shutdown_bonus
        
        # Отладочный вывод
        # print(f"Host utilization and scores:", file=sys.stderr)
        # for host_id, util in host_utilization.items():
            # print(f"  Host {host_id}: utilization={util:.4f}, score={host_scores[host_id]:.2f}", file=sys.stderr)
        
        # print(f"Zero utilization counts: {self.host_zero_utilization_count}", file=sys.stderr)
        # print(f"Hosts with previous VMs: {self.hosts_with_previous_vms}", file=sys.stderr)
        # print(f"Shutdown hosts: {shutdown_hosts}", file=sys.stderr)
        
        # print(f"Total score breakdown: utilization={utilization_score:.2f}, migration={migration_penalty:.2f}, failure={allocation_failure_penalty:.2f}, shutdown_bonus={shutdown_bonus:.2f}, total={total_score:.2f}", file=sys.stderr)
            
        return total_score

    def calculate_migration_benefit(self, vm_id: str, source_host: str, target_host: str, 
                                  allocations: Dict[str, List[str]], migration_count: int) -> float:
        """Рассчитывает выгоду от миграции VM с учетом всех факторов.
        
        Параметры:
        - vm_id: ID VM для миграции
        - source_host: исходный хост
        - target_host: целевой хост
        - allocations: текущее размещение VM
        - migration_count: текущее количество миграций в этом раунде
        
        Возвращает:
        - float: чистая выгода от миграции
        """
        # Штраф за миграцию (квадратичный)
        migration_penalty = (migration_count + 1) ** 2 - migration_count ** 2
        
        # Создаем тестовые размещения для оценки
        test_allocations = {host_id: list(vms) for host_id, vms in allocations.items()}
        
        # Удаляем VM с исходного хоста и добавляем на целевой
        if vm_id in test_allocations[source_host]:
            test_allocations[source_host].remove(vm_id)
        if target_host not in test_allocations:
            test_allocations[target_host] = []
        test_allocations[target_host].append(vm_id)
        
        # Оцениваем состояние до и после миграции
        source_before = self.calculate_host_capacity(source_host, allocations)
        source_after = self.calculate_host_capacity(source_host, test_allocations)
        target_before = self.calculate_host_capacity(target_host, allocations)
        target_after = self.calculate_host_capacity(target_host, test_allocations)
        
        # Бонус за выключение исходного хоста
        shutdown_bonus = 0
        if source_after["max_utilization"] == 0 and source_host in self.hosts_with_previous_vms:
            # Если хост становится пустым и ранее использовался
            current_zero_count = self.host_zero_utilization_count.get(source_host, 0)
            
            # Если близко к порогу бонуса, даем дополнительный бонус
            if current_zero_count >= BONUS_THRESHOLD - 2:
                shutdown_bonus = 20  # Значительный бонус за приближение к порогу
            else:
                shutdown_bonus = 10  # Базовый бонус за выключение хоста
        
        # Бонус за улучшение утилизации целевого хоста
        target_utilization_bonus = 0
        if abs(target_after["max_utilization"] - TARGET_UTILIZATION) < abs(target_before["max_utilization"] - TARGET_UTILIZATION):
            # Улучшение = меньшее отклонение от цели
            improvement = abs(target_before["max_utilization"] - TARGET_UTILIZATION) - abs(target_after["max_utilization"] - TARGET_UTILIZATION)
            target_utilization_bonus = improvement * 30  # Бонус пропорционален улучшению
        
        # Штраф за превышение верхнего порога утилизации
        target_utilization_penalty = 0
        if target_after["max_utilization"] > UPPER_THRESHOLD:
            # Штраф пропорционален превышению порога
            excess = target_after["max_utilization"] - UPPER_THRESHOLD
            target_utilization_penalty = excess * 50  # Значительный штраф за перегрузку
        
        # Штраф за недавнюю миграцию этой VM
        recent_migration_penalty = 0
        if vm_id in self.recent_migrations:
            # Штраф за миграцию VM, которая недавно мигрировала
            recent_migration_penalty = 15
        
        # Долгосрочная выгода от миграции
        long_term_benefit = self.predict_long_term_benefit(source_host, target_host, vm_id, allocations)
        
        # Суммарная выгода
        total_benefit = (
            target_utilization_bonus 
            + shutdown_bonus 
            - target_utilization_penalty 
            - migration_penalty 
            - recent_migration_penalty
            + long_term_benefit
        )
        
        # Логирование деталей расчета
        # print(f"Migration benefit for VM {vm_id} from {source_host} to {target_host}:", file=sys.stderr)
        # print(f"  Net benefit: {total_benefit:.2f}", file=sys.stderr)
        # print(f"  Shutdown bonus: {shutdown_bonus:.2f}", file=sys.stderr)
        # print(f"  Target utilization bonus: {target_utilization_bonus:.2f}", file=sys.stderr)
        # print(f"  Target utilization penalty: {target_utilization_penalty:.2f}", file=sys.stderr)
        # print(f"  Migration penalty: {migration_penalty:.2f}", file=sys.stderr)
        # print(f"  Recent migration penalty: {recent_migration_penalty:.2f}", file=sys.stderr)
        # print(f"  Long-term benefit: {long_term_benefit:.2f}", file=sys.stderr)
        
        return total_benefit

    def predict_long_term_benefit(self, source_host: str, target_host: str, vm_id: str, 
                                allocations: Dict[str, List[str]]) -> float:
        """Прогнозирует долгосрочную выгоду от миграции на несколько раундов вперед.
        
        Параметры:
        - source_host: исходный хост
        - target_host: целевой хост
        - vm_id: ID VM для миграции
        - allocations: текущее размещение VM
        
        Возвращает:
        - float: прогнозируемая долгосрочная выгода
        """
        # Количество раундов для прогнозирования
        forecast_rounds = 5  # Увеличено с 3 до 5
        
        # Прогнозируемая выгода в течение нескольких раундов
        long_term_benefit = 0
        
        # Создаем тестовые размещения для оценки
        test_allocations = {host_id: list(vms) for host_id, vms in allocations.items()}
        
        # Удаляем VM с исходного хоста и добавляем на целевой
        if vm_id in test_allocations[source_host]:
            test_allocations[source_host].remove(vm_id)
        if target_host not in test_allocations:
            test_allocations[target_host] = []
        test_allocations[target_host].append(vm_id)
        
        # Проверяем, выключится ли исходный хост после миграции
        source_becomes_empty = len(test_allocations[source_host]) == 0
        
        # Получаем текущий счетчик нулевой утилизации
        source_zero_count = self.host_zero_utilization_count.get(source_host, 0)
        
        # Если хост выключится, учитываем потенциальный бонус в будущих раундах
        if source_becomes_empty and source_host in self.hosts_with_previous_vms:
            for i in range(1, forecast_rounds + 1):
                future_zero_count = source_zero_count + i
                # Если достигнем порога бонуса, добавляем бонус
                if future_zero_count >= BONUS_THRESHOLD:
                    # Бонус увеличивается с каждым дополнительным раундом
                    bonus_multiplier = 1.0 + (future_zero_count - BONUS_THRESHOLD) * 0.1  # Дополнительный бонус за каждый раунд сверх порога
                    round_bonus = 8 * bonus_multiplier  # Бонус за выключенный хост с множителем
                    long_term_benefit += round_bonus
                    
                    # print(f"Predicted bonus for host {source_host} in round +{i}: {round_bonus:.2f} (count: {future_zero_count})", file=sys.stderr)
        
        # Оцениваем влияние на утилизацию целевого хоста в долгосрочной перспективе
        target_utilization_before = self.calculate_host_capacity(target_host, allocations)["max_utilization"]
        target_utilization_after = self.calculate_host_capacity(target_host, test_allocations)["max_utilization"]
        
        # Если утилизация приближается к оптимальной, добавляем долгосрочный бонус
        if abs(target_utilization_after - TARGET_UTILIZATION) < abs(target_utilization_before - TARGET_UTILIZATION):
            # Улучшение утилизации = меньшее отклонение от цели
            improvement = abs(target_utilization_before - TARGET_UTILIZATION) - abs(target_utilization_after - TARGET_UTILIZATION)
            
            # Бонус за долгосрочное улучшение утилизации
            utilization_bonus = improvement * 15 * forecast_rounds  # Бонус за несколько раундов
            long_term_benefit += utilization_bonus
            
            # print(f"Predicted utilization improvement bonus: {utilization_bonus:.2f}", file=sys.stderr)
        
        # Оцениваем влияние на консолидацию в долгосрочной перспективе
        # Если миграция помогает консолидировать VM на меньшем количестве хостов
        active_hosts_before = sum(1 for host_id, vms in allocations.items() if vms)
        active_hosts_after = sum(1 for host_id, vms in test_allocations.items() if vms)
        
        if active_hosts_after < active_hosts_before:
            # Бонус за уменьшение количества активных хостов
            consolidation_bonus = (active_hosts_before - active_hosts_after) * 10 * forecast_rounds
            long_term_benefit += consolidation_bonus
            
            # print(f"Predicted consolidation bonus: {consolidation_bonus:.2f}", file=sys.stderr)
        
        # print(f"Total long-term benefit for migrating VM {vm_id} from {source_host} to {target_host}: {long_term_benefit:.2f}", file=sys.stderr)
        
        return long_term_benefit

    def place_vms(self, vms_to_place: List[str], hosts: List[str], existing_allocations: Dict[str, List[str]] = None) -> Dict[str, List[str]]:
        """Размещает VM на хостах.
        
        Args:
            vms_to_place: список VM для размещения
            hosts: список доступных хостов
            existing_allocations: существующие размещения VM
            
        Returns:
            Dict[str, List[str]]: новые размещения VM по хостам
        """
        # Если нет VM для размещения, возвращаем существующие размещения
        if not vms_to_place:
            return existing_allocations or {}
            
        # Инициализируем размещения на основе существующих
        allocations = existing_allocations.copy() if existing_allocations else {}
        
        # Отслеживаем уже размещенные VM
        placed_vms = set()
        for host_vms in allocations.values():
            placed_vms.update(host_vms)
        
        # Фильтруем VM, которые нужно разместить
        vms_to_place = [vm for vm in vms_to_place if vm not in placed_vms]
        
        # Если все VM уже размещены, возвращаем текущие размещения
        if not vms_to_place:
            return allocations
        
        # Сортируем хосты по доступным ресурсам
        sorted_hosts = sorted(hosts, key=lambda h: (
            self.hosts[h]['cpu'] + self.hosts[h]['ram']
        ), reverse=True)
        
        # Сортируем VM по размеру (CPU + RAM)
        sorted_vms = sorted(vms_to_place, key=lambda vm: (
            self.vms[vm]['cpu'] + self.vms[vm]['ram']
        ), reverse=True)
        
        # Пытаемся разместить каждую VM
        for vm_id in sorted_vms:
            placed = False
            best_host = None
            best_score = float('-inf')
            
            # Проверяем каждый хост
            for host_id in sorted_hosts:
                if self.can_host_vm(host_id, vm_id, allocations):
                    # Рассчитываем утилизацию хоста после размещения VM
                    host_state = self.calculate_host_capacity(host_id, allocations)
                    cpu_util = host_state["utilization"]["cpu"]
                    ram_util = host_state["utilization"]["ram"]
                    
                    # Считаем баланс утилизации
                    balance_score = -abs(cpu_util - ram_util)
                    
                    # Общий скор для хоста
                    host_score = balance_score + (1 - max(cpu_util, ram_util))
                    
                    if host_score > best_score:
                        best_score = host_score
                        best_host = host_id
            
            if best_host:
                if best_host not in allocations:
                    allocations[best_host] = []
                allocations[best_host].append(vm_id)
                placed = True
            
            if not placed:
                # Пытаемся найти хост через миграции
                success = self._handle_unplaceable_vm(vm_id, vms_to_place, allocations, hosts)
                if not success:
                    self.allocation_failures.append(vm_id)
        
        return allocations

    def _handle_unplaceable_vm(self, vm_id, vms_to_place, allocations, hosts):
        """Пытается разместить неразмещенную VM через миграции."""
        # Если МАКС количество миграций уже достигнуто, не пытаемся
        if len(self.migrations) >= MAX_MIGRATIONS:
            return False
            
        if vm_id not in self.vms:
            return False
            
        vm_info = self.vms[vm_id]
        vm_cpu = vm_info['cpu']
        vm_ram = vm_info['ram']
        
        # Сортируем хосты по утилизации (от меньшей к большей)
        hosts_by_utilization = sorted(
            hosts,
            key=lambda h: self.calculate_host_capacity(h, allocations)["max_utilization"]
        )
        
        # Для каждого хоста проверяем, можно ли освободить место для новой VM
        for target_host in hosts_by_utilization:
            if target_host not in self.hosts:
                continue
            
            host_capacity = self.calculate_host_capacity(target_host, allocations)
            remaining_cpu = self.hosts[target_host]['cpu'] - host_capacity['capacity']['cpu']
            remaining_ram = self.hosts[target_host]['ram'] - host_capacity['capacity']['ram']
            
            # Если места достаточно, размещаем VM напрямую
            if remaining_cpu >= vm_cpu and remaining_ram >= vm_ram:
                if target_host not in allocations:
                    allocations[target_host] = []
                allocations[target_host].append(vm_id)
                return True
            
            # Если остаточных ресурсов недостаточно, пробуем миграцию
            needed_cpu = max(0, vm_cpu - remaining_cpu)
            needed_ram = max(0, vm_ram - remaining_ram)
            
            # Пробуем найти VM, которые можно мигрировать, чтобы освободить место
            candidate_vms = []
            current_vms = allocations.get(target_host, [])
            
            for candidate_vm in current_vms:
                if candidate_vm not in self.vms:
                    continue
                
                cvm_info = self.vms[candidate_vm]
                cvm_cpu = cvm_info['cpu']
                cvm_ram = cvm_info['ram']
                
                if cvm_cpu >= needed_cpu or cvm_ram >= needed_ram:
                    candidate_vms.append({
                        'vm_id': candidate_vm,
                        'cpu': cvm_cpu,
                        'ram': cvm_ram,
                        'value': max(cvm_cpu / needed_cpu if needed_cpu > 0 else 0,
                                    cvm_ram / needed_ram if needed_ram > 0 else 0)
                    })
            
            # Сортируем кандидатов (предпочитаем тех, кто освобождает больше ресурсов)
            candidate_vms.sort(key=lambda x: x['value'], reverse=True)
            
            for candidate in candidate_vms:
                candidate_vm = candidate['vm_id']
                
                # Ищем хост, куда можно мигрировать VM-кандидат
                for source_host in hosts:
                    if source_host == target_host:
                        continue
                    
                    if source_host not in self.hosts:
                        continue
                    
                    source_capacity = self.calculate_host_capacity(source_host, allocations)
                    src_remaining_cpu = self.hosts[source_host]['cpu'] - source_capacity['capacity']['cpu']
                    src_remaining_ram = self.hosts[source_host]['ram'] - source_capacity['capacity']['ram']
                    
                    # Проверяем, хватит ли места для миграции
                    if src_remaining_cpu >= candidate['cpu'] and src_remaining_ram >= candidate['ram']:
                        # Проверяем, что VM действительно есть в списке перед удалением
                        if candidate_vm in allocations.get(target_host, []):
                            allocations[target_host].remove(candidate_vm)
                            
                            if source_host not in allocations:
                                allocations[source_host] = []
                            allocations[source_host].append(candidate_vm)
                            
                            # Добавляем информацию о миграции
                            self.migrations.append({
                                'vm_id': candidate_vm,
                                'source': target_host,
                                'target': source_host
                            })
                            
                            # Обновляем данные о емкости хостов
                            host_capacity = self.calculate_host_capacity(target_host, allocations)
                            remaining_cpu = self.hosts[target_host]['cpu'] - host_capacity['capacity']['cpu']
                            remaining_ram = self.hosts[target_host]['ram'] - host_capacity['capacity']['ram']
                            
                            # Проверяем, хватит ли теперь места для размещения новой VM
                            if remaining_cpu >= vm_cpu and remaining_ram >= vm_ram:
                                # Размещаем новую VM
                                if target_host not in allocations:
                                    allocations[target_host] = []
                                allocations[target_host].append(vm_id)
                                return True
        
        # Если не удалось разместить VM даже с миграциями, пробуем каскадные миграции
        return self._try_cascade_migration(vm_id, None, allocations, hosts)

    def _try_cascade_migration(self, vm_id, target_host, allocations, hosts):
        """Пытается разместить VM с помощью каскадных миграций."""
        # Если МАКС количество миграций уже достигнуто, не пытаемся
        if len(self.migrations) >= MAX_MIGRATIONS:
            return False
            
        vm_info = self.vms[vm_id]
        vm_cpu = vm_info['cpu']
        vm_ram = vm_info['ram']
        
        # Если целевой хост не указан, ищем наилучший хост для размещения
        if target_host is None:
            target_host = self._find_best_host_for_placement(vm_id, hosts, allocations, self.host_utilization)
            if target_host is None:
                return False
            
        # Проверяем, хватает ли места на целевом хосте
        target_capacity = self.calculate_host_capacity(target_host, allocations)
        remaining_cpu = self.hosts[target_host]['cpu'] - target_capacity['capacity']['cpu']
        remaining_ram = self.hosts[target_host]['ram'] - target_capacity['capacity']['ram']
        
        if remaining_cpu >= vm_cpu and remaining_ram >= vm_ram:
            # Если места достаточно, размещаем VM напрямую
            allocations[target_host].append(vm_id)
            return True
        
        # Если места недостаточно, пробуем каскадные миграции
        needed_cpu = max(0, vm_cpu - remaining_cpu)
        needed_ram = max(0, vm_ram - remaining_ram)
        
        # Пробуем найти VM, которые можно мигрировать, чтобы освободить место
        candidate_vms = []
        for candidate_vm in allocations.get(target_host, []):
            if candidate_vm not in self.vms:
                continue  # Пропускаем VM, которых нет в self.vms
            cvm_info = self.vms[candidate_vm]
            if cvm_info['cpu'] >= needed_cpu or cvm_info['ram'] >= needed_ram:
                candidate_vms.append({
                    'vm_id': candidate_vm,
                    'cpu': cvm_info['cpu'],
                    'ram': cvm_info['ram'],
                    'value': max(cvm_info['cpu'] / needed_cpu if needed_cpu > 0 else 0,
                                cvm_info['ram'] / needed_ram if needed_ram > 0 else 0)
                })
                
        # Сортируем кандидатов (предпочитаем тех, кто освобождает больше ресурсов)
        candidate_vms.sort(key=lambda x: x['value'], reverse=True)
        
        for candidate in candidate_vms:
            candidate_vm = candidate['vm_id']
            
            # Ищем хост, куда можно мигрировать VM-кандидат
            for source_host in hosts:
                if source_host == target_host:
                    continue
                
                source_capacity = self.calculate_host_capacity(source_host, allocations)
                src_remaining_cpu = self.hosts[source_host]['cpu'] - source_capacity['capacity']['cpu']
                src_remaining_ram = self.hosts[source_host]['ram'] - source_capacity['capacity']['ram']
                
                # Проверяем, хватит ли места для миграции
                if src_remaining_cpu >= candidate['cpu'] and src_remaining_ram >= candidate['ram']:
                    # Мигрируем VM-кандидат
                    allocations[target_host].remove(candidate_vm)
                    allocations[source_host].append(candidate_vm)
                    
                    # Добавляем информацию о миграции
                    self.migrations.append({
                        'vm_id': candidate_vm,
                        'source': target_host,
                        'target': source_host
                    })
                    
                    # Обновляем данные о емкости хостов
                    target_capacity = self.calculate_host_capacity(target_host, allocations)
                    remaining_cpu = self.hosts[target_host]['cpu'] - target_capacity['capacity']['cpu']
                    remaining_ram = self.hosts[target_host]['ram'] - target_capacity['capacity']['ram']
                    
                    # Проверяем, хватит ли теперь места для размещения новой VM
                    if remaining_cpu >= vm_cpu and remaining_ram >= vm_ram:
                        # Размещаем новую VM
                        allocations[target_host].append(vm_id)
                        return True
                    
                    # Если места все еще недостаточно, пробуем каскадные миграции для VM-кандидата
                    if self._try_cascade_migration(candidate_vm, source_host, allocations, hosts):
                        # Если каскадная миграция успешна, проверяем, хватит ли теперь места для размещения новой VM
                        target_capacity = self.calculate_host_capacity(target_host, allocations)
                        remaining_cpu = self.hosts[target_host]['cpu'] - target_capacity['capacity']['cpu']
                        remaining_ram = self.hosts[target_host]['ram'] - target_capacity['capacity']['ram']
                        
                        if remaining_cpu >= vm_cpu and remaining_ram >= vm_ram:
                            # Размещаем новую VM
                            allocations[target_host].append(vm_id)
                            return True
                
        # Если не удалось разместить VM даже с каскадными миграциями, возвращаем False
        return False

    def _find_best_host_for_placement(self, vm_id, hosts, allocations, host_utilization):
        """Находит наилучший хост для размещения VM.
        
        Параметры:
        - vm_id: идентификатор VM
        - hosts: список идентификаторов хостов
        - allocations: текущие размещения {host_id: [vm_id, ...]}
        - host_utilization: текущая утилизация хостов {host_id: utilization}
        
        Возвращает:
        - str или None: идентификатор лучшего хоста или None, если подходящего хоста нет
        """
        # Получаем параметры VM
        vm_info = self.vms.get(vm_id, {})
        vm_cpu = vm_info.get('cpu', 0)
        vm_ram = vm_info.get('ram', 0)
        
        # Список подходящих хостов
        suitable_hosts = []
        
        # Проверяем каждый хост
        for host_id in hosts:
            # Проверяем, может ли VM быть размещена на данном хосте
            if not self.can_host_vm(host_id, vm_id):
                continue
                
            # Рассчитываем новую утилизацию хоста
            host_info = self.hosts.get(host_id, {})
            host_cpu = host_info.get('cpu', 0)
            host_ram = host_info.get('ram', 0)
            
            current_vms = allocations.get(host_id, [])
            current_capacity = self.calculate_host_capacity(host_id, current_vms)
            
            new_cpu_usage = current_capacity['capacity']['cpu'] + vm_cpu
            new_ram_usage = current_capacity['capacity']['ram'] + vm_ram
            
            new_cpu_utilization = new_cpu_usage / host_cpu
            new_ram_utilization = new_ram_usage / host_ram
            
            new_utilization = max(new_cpu_utilization, new_ram_utilization)
            
            # Если утилизация превышает верхний предел, пропускаем этот хост
            if new_utilization > self.UPPER_THRESHOLD:
                continue
                
            # Рассчитываем оценку для размещения
            current_utilization = host_utilization.get(host_id, 0.0)
            
            # Рассчитываем изменение в оценке
            current_score = self._calculate_utilization_reward(current_utilization)
            new_score = self._calculate_utilization_reward(new_utilization)
            
            score_delta = new_score - current_score
            
            # Бонус за соответствие типа VM и хоста
            host_bonus = 0
            if "cpu" in host_id and vm_cpu > vm_ram * 1.5:
                host_bonus += 2  # Бонус для CPU-интенсивных VM на CPU-хостах
            elif "ram" in host_id and vm_ram > vm_cpu * 1.5:
                host_bonus += 2  # Бонус для RAM-интенсивных VM на RAM-хостах
            
            # Бонус за близость к целевой утилизации
            target_bonus = 0
            if abs(new_utilization - self.TARGET_UTILIZATION) < 0.05:
                target_bonus += 3  # Бонус за приближение к целевой утилизации
            
            # Итоговая оценка для хоста
            host_score = score_delta + host_bonus + target_bonus
            
            suitable_hosts.append((host_id, host_score, new_utilization))
        
        # Если нет подходящих хостов, возвращаем None
        if not suitable_hosts:
            return None
            
        # Сортируем хосты по оценке (от большей к меньшей)
        suitable_hosts.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем лучший хост
        return suitable_hosts[0][0]
        
    def _find_migration_target(self, vm_id, hosts, allocations, excluded_hosts=None):
        """Находит подходящий хост для миграции VM.
        
        Параметры:
        - vm_id: идентификатор VM для миграции
        - hosts: список хостов для поиска
        - allocations: текущие размещения {host_id: [vm_id, ...]}
        - excluded_hosts: список хостов, исключенных из поиска
        
        Возвращает:
        - str или None: идентификатор целевого хоста или None, если подходящего хоста нет
        """
        if excluded_hosts is None:
            excluded_hosts = []
            
        # Получаем параметры VM
        vm_info = self.vms.get(vm_id, {})
        vm_cpu = vm_info.get('cpu', 0)
        vm_ram = vm_info.get('ram', 0)
        
        # Список подходящих хостов
        suitable_hosts = []
        
        # Проверяем каждый хост
        for host_id in hosts:
            # Пропускаем исключенные хосты
            if host_id in excluded_hosts:
                continue
                
            # Проверяем, может ли VM быть размещена на данном хосте
            if not self.can_host_vm(host_id, vm_id):
                continue
                
            # Рассчитываем новую утилизацию хоста
            host_info = self.hosts.get(host_id, {})
            host_cpu = host_info.get('cpu', 0)
            host_ram = host_info.get('ram', 0)
            
            current_vms = allocations.get(host_id, [])
            current_capacity = self.calculate_host_capacity(host_id, current_vms)
            
            new_cpu_usage = current_capacity['capacity']['cpu'] + vm_cpu
            new_ram_usage = current_capacity['capacity']['ram'] + vm_ram
            
            new_cpu_utilization = new_cpu_usage / host_cpu
            new_ram_utilization = new_ram_usage / host_ram
            
            new_utilization = max(new_cpu_utilization, new_ram_utilization)
            
            # Если утилизация превышает верхний предел, пропускаем этот хост
            if new_utilization > self.UPPER_THRESHOLD:
                continue
                
            # Рассчитываем оценку для миграции
            current_utilization = max(
                current_capacity['capacity']['cpu'] / host_cpu,
                current_capacity['capacity']['ram'] / host_ram
            )
            
            # Бонус за приближение к целевой утилизации
            distance_to_target = abs(new_utilization - self.TARGET_UTILIZATION)
            target_bonus = 5 * (1 - min(1, distance_to_target / self.TARGET_UTILIZATION))
            
            # Бонус за соответствие типа VM и хоста
            host_bonus = 0
            if "cpu" in host_id and vm_cpu > vm_ram * 1.5:
                host_bonus += 2  # Бонус для CPU-интенсивных VM на CPU-хостах
            elif "ram" in host_id and vm_ram > vm_cpu * 1.5:
                host_bonus += 2  # Бонус для RAM-интенсивных VM на RAM-хостах
            
            # Штраф за высокую утилизацию
            high_utilization_penalty = 0
            if new_utilization > 0.9:
                high_utilization_penalty = 10 * (new_utilization - 0.9) / 0.1
            
            # Итоговая оценка для хоста
            host_score = target_bonus + host_bonus - high_utilization_penalty
            
            suitable_hosts.append((host_id, host_score, new_utilization))
        
        # Если нет подходящих хостов, возвращаем None
        if not suitable_hosts:
            return None
            
        # Сортируем хосты по оценке (от большей к меньшей)
        suitable_hosts.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем лучший хост
        return suitable_hosts[0][0]

    def _calculate_utilization_reward(self, utilization):
        """Рассчитывает вознаграждение за утилизацию.
        
        Параметры:
        - utilization: значение утилизации (от 0 до 1)
        
        Возвращает:
        - float: значение вознаграждения
        """
        if utilization <= 0:
            return 0
            
        # Используем точную формулу из условий задачи:
        # f(x) = -0.67459 + (42.38075/(−2.5x+5.96))×exp(−2×(ln(−2.5x+2.96))²)
        
        # Защита от выхода за границы допустимых значений
        x = max(0.0, min(1.0, utilization))
        
        # Вычисляем компоненты формулы
        term1 = -0.67459
        inner_term = -2.5 * x + 5.96
        
        # Если аргумент логарифма слишком мал, возвращаем минимальное значение
        if inner_term <= 0:
            return 0
            
        inner_log_term = -2.5 * x + 2.96
        
        # Если аргумент логарифма слишком мал, возвращаем минимальное значение
        if inner_log_term <= 0:
            return 0
            
        term2 = 42.38075 / inner_term
        term3 = math.exp(-2 * (math.log(inner_log_term) ** 2))
        
        result = term1 + term2 * term3
        
        # Ограничиваем результат снизу нулем
        return max(0, result)

    def update_utilization_counters(self):
        """Обновляет счетчики утилизации хостов."""
        # Обновляем счетчики хостов с нулевой утилизацией
        for host_id in self.hosts:
            host_vms = self.previous_allocations.get(host_id, [])
            
            if not host_vms:
                # Увеличиваем счетчик, если на хосте нет VM
                self.host_zero_utilization_count[host_id] = self.host_zero_utilization_count.get(host_id, 0) + 1
                # print(f"Host {host_id} has zero utilization for {self.host_zero_utilization_count[host_id]} rounds", file=sys.stderr)
            else:
                # Сбрасываем счетчик, если на хосте есть VM
                self.host_zero_utilization_count[host_id] = 0
                
        # Выводим информацию о хостах с бонусами за отключение
        bonus_hosts = [h for h, count in self.host_zero_utilization_count.items() if count >= self.BONUS_THRESHOLD]
        # if bonus_hosts:
            # print(f"Bonus for {len(bonus_hosts)} hosts with zero utilization for {self.BONUS_THRESHOLD}+ rounds: {bonus_hosts}", file=sys.stderr)
            
        # Обновляем счетчик хостов, готовых к отключению
        self.hosts_ready_for_shutdown = [h for h, count in self.host_zero_utilization_count.items() 
                                        if count >= self.BONUS_THRESHOLD - 1]  # Хосты, которые получат бонус в следующем раунде
        
        # Выводим информацию о хостах, готовых к отключению
        # if self.hosts_ready_for_shutdown:
            # print(f"Hosts ready for shutdown next round: {self.hosts_ready_for_shutdown}", file=sys.stderr)

    def _try_cascade_migration(self, vm_id, target_host, allocations, hosts, depth=0, max_depth=3, visited=None):
        """Пытается выполнить каскадную миграцию для размещения VM.
        
        Параметры:
        - vm_id: идентификатор VM для размещения
        - target_host: целевой хост для размещения (None - если нужно найти подходящий)
        - allocations: текущие размещения {host_id: [vm_id, ...]}
        - hosts: список хостов
        - depth: текущая глубина рекурсии
        - max_depth: максимальная глубина рекурсии (увеличена до 3)
        - visited: множество посещенных хостов
        
        Возвращает:
        - bool: True, если удалось разместить VM, иначе False
        """
        # Проверяем лимит миграций
        if len(self.migrations) >= MAX_MIGRATIONS:
            return False
            
        if depth > max_depth:
            return False
        
        if visited is None:
            visited = set()
        
        # Получаем параметры VM
        vm_info = self.vms.get(vm_id, {})
        vm_cpu = vm_info.get('cpu', 0)
        vm_ram = vm_info.get('ram', 0)
        
        # Если целевой хост не указан, находим подходящие хосты
        if target_host is None:
            candidate_hosts = []
            
            # Определяем тип VM для подбора оптимального хоста
            vm_type = "balanced"
            cpu_ram_ratio = vm_cpu / max(1, vm_ram)
            
            if 'cpu_heavy' in vm_id or cpu_ram_ratio > 1.5:
                vm_type = "cpu_heavy"
            elif 'ram_heavy' in vm_id or cpu_ram_ratio < 0.67:
                vm_type = "ram_heavy"
            elif 'storage_heavy' in vm_id:
                vm_type = "storage_heavy"
                
            # Карта предпочтений хостов для разных типов VM
            host_preferences = {
                'cpu_heavy': ['cpu_optimized', 'balanced', 'storage_optimized', 'ram_optimized'],
                'ram_heavy': ['ram_optimized', 'balanced', 'storage_optimized', 'cpu_optimized'],
                'storage_heavy': ['storage_optimized', 'balanced', 'ram_optimized', 'cpu_optimized'],
                'balanced': ['balanced', 'storage_optimized', 'ram_optimized', 'cpu_optimized']
            }
            
            # Определяем типы хостов
            host_types = {}
            for host_id in hosts:
                if host_id in visited:
                    continue
                    
                host_info = self.hosts.get(host_id, {})
                host_cpu = host_info.get('cpu', 0)
                host_ram = host_info.get('ram', 0)
                
                # Проверяем, может ли хост в принципе вместить VM
                if vm_cpu > host_cpu or vm_ram > host_ram:
                    continue
                
                # Определяем тип хоста
                cpu_ram_ratio = host_cpu / max(1, host_ram)
                if 'cpu_optimized' in host_id or cpu_ram_ratio > 1.5:
                    host_types[host_id] = "cpu_optimized"
                elif 'ram_optimized' in host_id or cpu_ram_ratio < 0.67:
                    host_types[host_id] = "ram_optimized"
                elif 'storage_optimized' in host_id:
                    host_types[host_id] = "storage_optimized"
                else:
                    host_types[host_id] = "balanced"
                    
                # Определяем текущую утилизацию хоста
                current_vms = allocations.get(host_id, [])
                if not current_vms:
                    # Если хост пустой, он идеален для размещения
                    candidate_hosts.append((host_id, 0))
                    continue
                
                current_capacity = self.calculate_host_capacity(host_id, allocations)
                current_util = current_capacity['max_utilization']
                
                # Сколько ресурсов осталось на хосте
                remaining_cpu = host_cpu - current_capacity['capacity']['cpu']
                remaining_ram = host_ram - current_capacity['capacity']['ram']
                
                # Если хост может вместить VM напрямую
                if vm_cpu <= remaining_cpu and vm_ram <= remaining_ram:
                    # Оцениваем, насколько хорошо хост подходит для VM
                    # Учитываем тип хоста и соответствие VM, а также текущую утилизацию
                    type_match_score = 0
                    for i, preferred_type in enumerate(host_preferences[vm_type]):
                        if host_types[host_id] == preferred_type:
                            type_match_score = 3 - i  # 3 для лучшего типа, 0 для худшего
                            break
                    
                    # Рассчитываем утилизацию после размещения
                    new_cpu_util = (host_cpu - remaining_cpu + vm_cpu) / host_cpu
                    new_ram_util = (host_ram - remaining_ram + vm_ram) / host_ram
                    new_util = max(new_cpu_util, new_ram_util)
                    
                    # Предпочитаем утилизацию ближе к целевой
                    util_score = 3 * (1 - abs(TARGET_UTILIZATION - new_util) / TARGET_UTILIZATION)
                    
                    total_score = type_match_score + util_score
                    candidate_hosts.append((host_id, total_score))
                else:
                    # Хост требует миграций, даем ему низкий приоритет
                    candidate_hosts.append((host_id, -1))
            
            # Если нашли кандидатов
            if candidate_hosts:
                # Сортируем по оценке (лучшие первыми)
                candidate_hosts.sort(key=lambda x: x[1], reverse=True)
                
                # Пробуем разместить на лучших хостах
                for host_id, _ in candidate_hosts:
                    # Пробуем разместить VM на этом хосте
                    if self._try_cascade_migration(vm_id, host_id, allocations, hosts, depth, max_depth, visited.copy()):
                        return True
                
                # Не удалось разместить ни на одном хосте
                return False
            else:
                # Нет подходящих хостов
                return False
        
        # Если целевой хост указан
        if target_host in visited:
            return False
            
        visited.add(target_host)
        
        host_info = self.hosts.get(target_host, {})
        host_cpu = host_info.get('cpu', 0)
        host_ram = host_info.get('ram', 0)
        
        # Проверяем, может ли хост в принципе вместить VM
        if vm_cpu > host_cpu or vm_ram > host_ram:
            return False
        
        current_vms = allocations.get(target_host, [])
        current_capacity = self.calculate_host_capacity(target_host, allocations)
        
        # Сколько ресурсов осталось на хосте
        remaining_cpu = host_cpu - current_capacity['capacity']['cpu']
        remaining_ram = host_ram - current_capacity['capacity']['ram']
        
        # Если хост может вместить VM без миграций, размещаем сразу
        if vm_cpu <= remaining_cpu and vm_ram <= remaining_ram:
            if target_host not in allocations:
                allocations[target_host] = []
            allocations[target_host].append(vm_id)
            # print(f"Placed VM {vm_id} on host {target_host} in cascade migration", file=sys.stderr)
            return True
        
        # Сколько ресурсов не хватает
        missing_cpu = max(0, vm_cpu - remaining_cpu)
        missing_ram = max(0, vm_ram - remaining_ram)
        
        # Если хост в принципе может вместить VM после миграций
        if missing_cpu <= current_capacity['capacity']['cpu'] and missing_ram <= current_capacity['capacity']['ram']:
            # Находим VM для миграции с этого хоста
            host_vms = allocations.get(target_host, [])
            
            # Собираем VM, которые могут быть мигрированы
            candidate_vms = []
            for host_vm in host_vms:
                if host_vm in self.vms:
                    vm_cpu_usage = self.vms[host_vm]['cpu']
                    vm_ram_usage = self.vms[host_vm]['ram']
                    
                    # Считаем миграцию полезной, если VM освобождает нужные ресурсы
                    if vm_cpu_usage >= missing_cpu or vm_ram_usage >= missing_ram:
                        # Оцениваем пользу от миграции этой VM
                        cpu_benefit = vm_cpu_usage / missing_cpu if missing_cpu > 0 else 0
                        ram_benefit = vm_ram_usage / missing_ram if missing_ram > 0 else 0
                        
                        # Предпочитаем VM, которые освобождают больше ресурсов, но не слишком большие
                        benefit = max(cpu_benefit, ram_benefit)
                        
                        candidate_vms.append({
                            'vm_id': host_vm,
                            'cpu': vm_cpu_usage,
                            'ram': vm_ram_usage,
                            'benefit': benefit
                        })
            
            # Сортируем кандидатов по эффективности (лучшие первыми)
            candidate_vms.sort(key=lambda x: x['benefit'], reverse=True)
            
            # Пробуем мигрировать VM в порядке их эффективности
            for candidate in candidate_vms:
                candidate_vm = candidate['vm_id']
                
                # Находим хост, куда можно мигрировать эту VM
                for migration_target in hosts:
                    if migration_target == target_host or migration_target in visited:
                        continue
                    
                    # Проверяем, может ли VM быть размещена напрямую
                    target_capacity = self.calculate_host_capacity(migration_target, allocations)
                    target_cpu = self.hosts[migration_target]['cpu'] - target_capacity['capacity']['cpu']
                    target_ram = self.hosts[migration_target]['ram'] - target_capacity['capacity']['ram']
                    
                    if candidate['cpu'] <= target_cpu and candidate['ram'] <= target_ram:
                        # Мигрируем VM
                        allocations[target_host].remove(candidate_vm)
                        if migration_target not in allocations:
                            allocations[migration_target] = []
                        allocations[migration_target].append(candidate_vm)
                        
                        # Записываем миграцию
                        self.migrations.append({
                            'vm_id': candidate_vm,
                            'source': target_host,
                            'target': migration_target
                        })
                        
                        # print(f"Migrating VM {candidate_vm} from {target_host} to {migration_target} in cascade", file=sys.stderr)
                        
                        # Обновляем емкость хоста
                        current_capacity = self.calculate_host_capacity(target_host, allocations)
                        remaining_cpu = host_cpu - current_capacity['capacity']['cpu']
                        remaining_ram = host_ram - current_capacity['capacity']['ram']
                        
                        # Если после миграции можем разместить целевую VM
                        if vm_cpu <= remaining_cpu and vm_ram <= remaining_ram:
                            allocations[target_host].append(vm_id)
                            # print(f"Placed VM {vm_id} on host {target_host} after cascade migration", file=sys.stderr)
                            return True
                        
                        # Иначе продолжаем миграции
                        missing_cpu = max(0, vm_cpu - remaining_cpu)
                        missing_ram = max(0, vm_ram - remaining_ram)
                        
                        # Если больше не нужны миграции
                        if missing_cpu == 0 and missing_ram == 0:
                            allocations[target_host].append(vm_id)
                            # print(f"Placed VM {vm_id} on host {target_host} after cascade migrations", file=sys.stderr)
                            return True
                        
                        # Если превысили лимит миграций, останавливаемся
                        if len(self.migrations) >= MAX_MIGRATIONS:
                            return False
                    
                    # Если напрямую не можем разместить VM, пробуем каскадную миграцию
                    elif depth < max_depth:
                        # Создаем временную копию размещений для пробы
                        temp_allocations = allocations.copy()
                        for host_id, vms in allocations.items():
                            temp_allocations[host_id] = vms.copy()
                        
                        # Пробуем мигрировать VM через каскадную миграцию
                        if self._try_cascade_migration(candidate_vm, migration_target, temp_allocations, hosts, depth + 1, max_depth, visited.copy()):
                            # Если удалось, применяем изменения
                            allocations.clear()
                            for host_id, vms in temp_allocations.items():
                                allocations[host_id] = vms
                            
                            # Обновляем емкость хоста
                            current_capacity = self.calculate_host_capacity(target_host, allocations)
                            remaining_cpu = host_cpu - current_capacity['capacity']['cpu']
                            remaining_ram = host_ram - current_capacity['capacity']['ram']
                            
                            # Если после миграций можем разместить целевую VM
                            if vm_cpu <= remaining_cpu and vm_ram <= remaining_ram:
                                allocations[target_host].append(vm_id)
                                # print(f"Placed VM {vm_id} on host {target_host} after deep cascade migration", file=sys.stderr)
                                return True
                            
                            # Иначе продолжаем миграции
                            missing_cpu = max(0, vm_cpu - remaining_cpu)
                            missing_ram = max(0, vm_ram - remaining_ram)
            
        # Не удалось разместить VM с каскадными миграциями
        return False

    def _balance_resources(self, target, resources):
        """Балансирует ресурсы между хостами.
        
        Параметры:
        - target: целевые хосты для балансировки ('all' или список хостов)
        - resources: ресурсы для балансировки (['cpu', 'ram'])
        """
        # print(f"Balancing resources {resources} for target {target}", file=sys.stderr)
        # Пока просто заглушка, можно реализовать позже
        pass
        
    def _optimize_placement(self, target, by_type=False, priority="balanced"):
        """Оптимизирует размещение VM на хостах.
        
        Параметры:
        - target: целевые хосты для оптимизации ('all' или список хостов)
        - by_type: группировать ли VM по типу
        - priority: приоритет оптимизации ('balanced', 'performance', etc.)
        """
        # print(f"Optimizing placement for target {target}, by_type={by_type}, priority={priority}", file=sys.stderr)
        
        # Определяем список хостов для оптимизации
        hosts_to_optimize = list(self.hosts.keys()) if target == "all" else target
        if not isinstance(hosts_to_optimize, list):
            hosts_to_optimize = [hosts_to_optimize]
            
        # Получаем текущие размещения
        current_allocations = copy.deepcopy(self.previous_allocations)
        
        # Если нужно группировать по типу, собираем VM по типам
        if by_type:
            vm_by_type = {}
            for vm_id in self.vms:
                vm_info = self.vms[vm_id]
                cpu = vm_info.get('cpu', 0)
                ram = vm_info.get('ram', 0)
                
                # Определяем тип VM
                cpu_ram_ratio = cpu / max(1, ram)
                if cpu_ram_ratio > 1.5:
                    vm_type = "cpu_heavy"
                elif cpu_ram_ratio < 0.67:
                    vm_type = "ram_heavy"
                else:
                    vm_type = "balanced"
                    
                if vm_type not in vm_by_type:
                    vm_by_type[vm_type] = []
                vm_by_type[vm_type].append(vm_id)
                
            # Размещаем VM по типам на соответствующие хосты
            # Это упрощенная реализация, можно улучшить
            for vm_type, vms in vm_by_type.items():
                # Выбираем подходящие хосты для этого типа VM
                suitable_hosts = []
                for host_id in hosts_to_optimize:
                    host_info = self.hosts[host_id]
                    host_cpu = host_info.get('cpu', 0)
                    host_ram = host_info.get('ram', 0)
                    host_cpu_ram_ratio = host_cpu / max(1, host_ram)
                    
                    # Подбираем хосты по соответствию типу VM
                    if vm_type == "cpu_heavy" and host_cpu_ram_ratio > 1.2:
                        suitable_hosts.append(host_id)
                    elif vm_type == "ram_heavy" and host_cpu_ram_ratio < 0.8:
                        suitable_hosts.append(host_id)
                    elif vm_type == "balanced" and 0.8 <= host_cpu_ram_ratio <= 1.2:
                        suitable_hosts.append(host_id)
                        
                # Если нет подходящих хостов, используем все доступные
                if not suitable_hosts:
                    suitable_hosts = hosts_to_optimize
                    
                # Размещаем VM этого типа на подходящие хосты
                # Здесь можно использовать более сложную логику
                # Но для простоты просто распределяем равномерно
                for i, vm_id in enumerate(vms):
                    host_idx = i % len(suitable_hosts)
                    host_id = suitable_hosts[host_idx]
                    
                    # Добавляем VM на хост
                    if host_id not in current_allocations:
                        current_allocations[host_id] = []
                    current_allocations[host_id].append(vm_id)
        
        # Если не группируем по типу, просто оптимизируем размещение
        # в соответствии с приоритетом
        else:
            # Получаем все VM, которые нужно разместить
            all_vms = []
            for host_id in current_allocations:
                all_vms.extend(current_allocations[host_id])
                
            # Очищаем текущие размещения для хостов, которые оптимизируем
            for host_id in hosts_to_optimize:
                if host_id in current_allocations:
                    current_allocations[host_id] = []
                    
            # Сортируем VM в зависимости от приоритета
            if priority == "balanced":
                # Сортируем по размеру (сначала большие)
                all_vms.sort(key=lambda vm_id: max(
                    self.vms.get(vm_id, {}).get('cpu', 0) / 32,
                    self.vms.get(vm_id, {}).get('ram', 0) / 64
                ), reverse=True)
            elif priority == "performance":
                # Сортируем по CPU (сначала с высоким CPU)
                all_vms.sort(key=lambda vm_id: self.vms.get(vm_id, {}).get('cpu', 0), reverse=True)
            elif priority == "memory":
                # Сортируем по RAM (сначала с высоким RAM)
                all_vms.sort(key=lambda vm_id: self.vms.get(vm_id, {}).get('ram', 0), reverse=True)
            else:
                # По умолчанию сортируем по размеру
                all_vms.sort(key=lambda vm_id: max(
                    self.vms.get(vm_id, {}).get('cpu', 0) / 32,
                    self.vms.get(vm_id, {}).get('ram', 0) / 64
                ), reverse=True)
                
            # Размещаем VM на хосты
            # Используем алгоритм First-Fit-Decreasing
            for vm_id in all_vms:
                vm_info = self.vms.get(vm_id, {})
                vm_cpu = vm_info.get('cpu', 0)
                vm_ram = vm_info.get('ram', 0)
                
                # Ищем подходящий хост
                best_host = None
                best_fit = float('inf')
                
                for host_id in hosts_to_optimize:
                    # Проверяем, хватит ли ресурсов на хосте
                    host_capacity = self.calculate_host_capacity(host_id, current_allocations)
                    if host_capacity['cpu'] >= vm_cpu and host_capacity['ram'] >= vm_ram:
                        # Вычисляем, насколько хорошо VM подходит для этого хоста
                        # Используем разные метрики в зависимости от приоритета
                        if priority == "balanced":
                            # Стремимся к целевой утилизации
                            new_utilization = max(
                                (self.hosts[host_id]['cpu'] - host_capacity['cpu'] + vm_cpu) / self.hosts[host_id]['cpu'],
                                (self.hosts[host_id]['ram'] - host_capacity['ram'] + vm_ram) / self.hosts[host_id]['ram']
                            )
                            fit = abs(new_utilization - self.TARGET_UTILIZATION)
                        elif priority == "performance":
                            # Минимизируем утилизацию CPU
                            new_cpu_util = (self.hosts[host_id]['cpu'] - host_capacity['cpu'] + vm_cpu) / self.hosts[host_id]['cpu']
                            fit = new_cpu_util
                        elif priority == "memory":
                            # Минимизируем утилизацию RAM
                            new_ram_util = (self.hosts[host_id]['ram'] - host_capacity['ram'] + vm_ram) / self.hosts[host_id]['ram']
                            fit = new_ram_util
                        else:
                            # По умолчанию стремимся к целевой утилизации
                            new_utilization = max(
                                (self.hosts[host_id]['cpu'] - host_capacity['cpu'] + vm_cpu) / self.hosts[host_id]['cpu'],
                                (self.hosts[host_id]['ram'] - host_capacity['ram'] + vm_ram) / self.hosts[host_id]['ram']
                            )
                            fit = abs(new_utilization - self.TARGET_UTILIZATION)
                            
                        # Если этот хост лучше подходит, запоминаем его
                        if fit < best_fit:
                            best_fit = fit
                            best_host = host_id
                
                # Если нашли подходящий хост, размещаем VM на нем
                if best_host:
                    if best_host not in current_allocations:
                        current_allocations[best_host] = []
                    current_allocations[best_host].append(vm_id)
                    
        # Вычисляем миграции на основе разницы между текущими и новыми размещениями
        self._calculate_migrations(self.previous_allocations, current_allocations)
        
        # Обновляем размещения
        self.previous_allocations = current_allocations
        
        # Обновляем vm_to_host_map
        self.vm_to_host_map.clear()
        for host_id, vms in current_allocations.items():
            for vm_id in vms:
                self.vm_to_host_map[vm_id] = host_id

    def _calculate_migrations(self, old_allocations, new_allocations):
        """Вычисляет миграции на основе разницы между старыми и новыми размещениями.
        
        Параметры:
        - old_allocations: старые размещения {host_id: [vm_id, ...]}
        - new_allocations: новые размещения {host_id: [vm_id, ...]}
        
        Обновляет self.migrations списком миграций.
        """
        # Создаем отображение VM -> хост для старых размещений
        old_vm_to_host = {}
        for host_id, vms in old_allocations.items():
            for vm_id in vms:
                old_vm_to_host[vm_id] = host_id
                
        # Создаем отображение VM -> хост для новых размещений
        new_vm_to_host = {}
        for host_id, vms in new_allocations.items():
            for vm_id in vms:
                new_vm_to_host[vm_id] = host_id
                
        # Находим миграции (VM, которые переместились с одного хоста на другой)
        for vm_id in new_vm_to_host:
            if vm_id in old_vm_to_host and new_vm_to_host[vm_id] != old_vm_to_host[vm_id]:
                # Добавляем миграцию
                self.migrations.append({
                    "vm_id": vm_id,
                    "source": old_vm_to_host[vm_id],
                    "target": new_vm_to_host[vm_id]
                })
                
        # Ограничиваем количество миграций
        if len(self.migrations) > self.MAX_MIGRATIONS:
            # print(f"Too many migrations ({len(self.migrations)}), limiting to {self.MAX_MIGRATIONS}", file=sys.stderr)
            self.migrations = self.migrations[:self.MAX_MIGRATIONS]
            
        # Обновляем множество недавних миграций
        for migration in self.migrations:
            self.recent_migrations.add(migration["vm_id"])

    def estimate_required_hosts(self) -> int:
        """Оценивает необходимое количество хостов для текущей нагрузки."""
        # Считаем общие требования всех VM
        total_cpu = sum(self.vms[vm]['cpu'] for vm in self.vms)
        total_ram = sum(self.vms[vm]['ram'] for vm in self.vms)
        
        # Считаем средние ресурсы хостов
        avg_host_cpu = sum(self.hosts[h]['cpu'] for h in self.hosts) / len(self.hosts)
        avg_host_ram = sum(self.hosts[h]['ram'] for h in self.hosts) / len(self.hosts)
        
        # Оцениваем количество хостов с учетом целевой утилизации (0.807197)
        TARGET_UTILIZATION = 0.807197
        required_by_cpu = math.ceil(total_cpu / (avg_host_cpu * TARGET_UTILIZATION))
        required_by_ram = math.ceil(total_ram / (avg_host_ram * TARGET_UTILIZATION))
        
        # Возвращаем максимальное из двух значений
        return max(required_by_cpu, required_by_ram)

    def update_hosts_to_shutdown(self):
        """Обновляет список хостов, которые следует выключить для получения бонусов."""
        # Оцениваем необходимое количество хостов
        required_hosts = self.estimate_required_hosts()
        # print(f"Estimated required hosts: {required_hosts} out of {len(self.hosts)}", file=sys.stderr)  # Закомментировано
        
        # Хосты, которые уже близки к получению бонуса (4+ раунда с нулевой утилизацией)
        almost_bonus_hosts = {
            host_id for host_id, count in self.host_zero_utilization_count.items()
            if count >= 4 and host_id in self.hosts_with_previous_vms
        }
        # print(f"Hosts close to bonus: {almost_bonus_hosts}", file=sys.stderr)  # Закомментировано
        
        # Хосты с низкой утилизацией, которые можно выключить
        low_utilization_hosts = []
        for host_id in self.hosts:
            if host_id in self.previous_allocations and self.previous_allocations[host_id]:
                state = self.calculate_host_capacity(host_id)
                if state["max_utilization"] < LOWER_THRESHOLD:
                    low_utilization_hosts.append((host_id, state["max_utilization"]))
        
        # Сортируем по утилизации (сначала самые низкие)
        low_utilization_hosts.sort(key=lambda x: x[1])
        
        # Определяем, сколько хостов можно выключить
        hosts_to_shutdown_count = max(0, len(self.hosts) - required_hosts)
        
        # Приоритет для выключения:
        # 1. Хосты, близкие к получению бонуса
        # 2. Хосты с самой низкой утилизацией
        self.hosts_to_shutdown = set()
        
        # Добавляем хосты, близкие к бонусу
        self.hosts_to_shutdown.update(almost_bonus_hosts)
        
        # Добавляем хосты с низкой утилизацией, если нужно больше
        for host_id, util in low_utilization_hosts:
            if len(self.hosts_to_shutdown) < hosts_to_shutdown_count and host_id not in self.hosts_to_shutdown:
                self.hosts_to_shutdown.add(host_id)
        
        # print(f"Hosts to shutdown: {self.hosts_to_shutdown}", file=sys.stderr)  # Закомментировано

def split_json_objects(line):
    """Split concatenated JSON objects into a list of individual JSON objects."""
    objects = []
    depth = 0
    start = 0
    
    for i, char in enumerate(line):
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                try:
                    obj_str = line[start:i+1]
                    obj = json.loads(obj_str)
                    objects.append(obj)
                    start = i + 1
                except json.JSONDecodeError:
                    continue
    
    return objects

def main():
    scheduler = VMScheduler()
    
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        try:
            # Пытаемся разделить слипшиеся JSON объекты
            json_objects = split_json_objects(line)
            
            # Если не удалось разделить, пробуем загрузить как один объект
            if not json_objects:
                try:
                    json_objects = [json.loads(line)]
                except json.JSONDecodeError as e:
                    print(f"Ошибка при разборе JSON: {str(e)}", file=sys.stderr)
                    continue
            
            # Обрабатываем каждый JSON объект
            for data in json_objects:
                result = scheduler.process_input(data)
                print(json.dumps(result, ensure_ascii=False))
                sys.stdout.flush()
                
        except Exception as e:
            print(f"Ошибка при обработке входных данных: {str(e)}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            continue

if __name__ == "__main__":
    main()