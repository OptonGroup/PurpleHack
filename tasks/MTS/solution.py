#!/usr/bin/env python3
import json
import sys
from typing import Dict, List, Tuple, Set, Any, Optional
import time
import itertools

# Константы для оптимизации
TARGET_UTILIZATION = 0.8  # Целевая утилизация хостов (80%)
UPPER_THRESHOLD = 0.9  # Верхний порог утилизации
LOWER_THRESHOLD = 0.2  # Нижний порог утилизации
MAX_MIGRATIONS = 5  # Максимальное число миграций за раунд

class VMScheduler:
    def __init__(self):
        self.hosts = {}  # имя хоста -> {cpu, ram}
        self.vms = {}  # имя VM -> {cpu, ram}
        self.allocations = {}  # имя хоста -> список VM
        self.previous_allocations = {}  # имя хоста -> список VM из предыдущего раунда
        self.host_zero_utilization_count = {}  # отслеживание хостов с нулевой нагрузкой
        self.vm_to_host_map = {}  # Отображение VM -> Host из предыдущих аллокаций
        self.hosts_with_previous_vms = set()  # Множество хостов, на которых были VM
        self.migration_history = []  # История миграций
        self.current_round_migrations = set()  # Множество VM, которые уже мигрировали в текущем раунде
        
    def load_data(self, input_str: str) -> None:
        """Загружает входные данные."""
        try:
            data = json.loads(input_str)
            
            # Загружаем хосты
            self.hosts = data.get("hosts", {})
            
            # Сохраняем предыдущие размещения
            self.previous_allocations = data.get("allocations", {})
            
            # Загружаем виртуальные машины
            self.vms = data.get("virtual_machines", {})
            
            # Обрабатываем изменения
            diff = data.get("diff", {})
            
            # Добавляем новые VM
            if "add" in diff:
                new_vms = diff["add"].get("virtual_machines", {})
                if isinstance(new_vms, dict):
                    self.vms.update(new_vms)
                elif isinstance(new_vms, list):
                    # Если new_vms это список, значит VM уже есть в virtual_machines
                    pass
            
            # Удаляем VM
            if "remove" in diff:
                removed_vms = diff["remove"].get("virtual_machines", [])
                if isinstance(removed_vms, list):
                    for vm_id in removed_vms:
                        self.vms.pop(vm_id, None)
                        # Удаляем VM из предыдущих размещений
                        for host_vms in self.previous_allocations.values():
                            if vm_id in host_vms:
                                host_vms.remove(vm_id)
            
            # Обновляем vm_to_host_map на основе предыдущих размещений
            self.vm_to_host_map = {}
            for host_id, vm_list in self.previous_allocations.items():
                for vm_id in vm_list:
                    if vm_id in self.vms:  # Проверяем, что VM все еще существует
                        self.vm_to_host_map[vm_id] = host_id
            
            # Обновляем множество хостов с предыдущими VM
            self.hosts_with_previous_vms = set(
                host_id for host_id, vm_list in self.previous_allocations.items()
                if vm_list and host_id in self.hosts
            )
            
            # Инициализируем счетчик хостов с нулевой утилизацией
            self.host_zero_utilization_count = {host_id: 0 for host_id in self.hosts}
            
        except json.JSONDecodeError:
            # В случае ошибки разбора JSON, сбрасываем все атрибуты
            self.hosts = {}
            self.vms = {}
            self.previous_allocations = {}
            self.vm_to_host_map = {}
            self.hosts_with_previous_vms = set()
            self.host_zero_utilization_count = {}
            raise
        except Exception as e:
            # В случае любой другой ошибки, сбрасываем все атрибуты
            self.hosts = {}
            self.vms = {}
            self.previous_allocations = {}
            self.vm_to_host_map = {}
            self.hosts_with_previous_vms = set()
            self.host_zero_utilization_count = {}
            raise

    def calculate_host_capacity(self, host_id: str, allocations: Dict[str, List[str]] = None) -> Dict:
        """Вычисляет остаточные ресурсы хоста на основе текущих размещений VM."""
        if host_id not in self.hosts:
            return {}
        
        host = self.hosts[host_id]
        capacity = {
            "cpu": host.get("cpu", 0),
            "ram": host.get("ram", 0)
        }
        
        # Добавляем disk только если он есть в данных хоста
        if "disk" in host:
            capacity["disk"] = host["disk"]
        
        # Используем предоставленные аллокации или текущие
        allocs = allocations if allocations is not None else self.previous_allocations
        
        # Вычитаем ресурсы, используемые виртуальными машинами
        allocated_vms = allocs.get(host_id, [])
        for vm_id in allocated_vms:
            if vm_id in self.vms:
                vm = self.vms[vm_id]
                capacity["cpu"] -= vm.get("cpu", 0)
                capacity["ram"] -= vm.get("ram", 0)
                if "disk" in capacity and "disk" in vm:
                    capacity["disk"] -= vm["disk"]
        
        return capacity

    def can_host_vm(self, host_id: str, vm_id: str, existing_resources: Optional[Dict] = None) -> bool:
        """Проверяет, может ли хост разместить указанную VM."""
        if vm_id not in self.vms or host_id not in self.hosts:
            return False
        
        vm = self.vms[vm_id]
        host = self.hosts[host_id]
        
        # Используем переданные ресурсы или берем из хоста
        resources = existing_resources if existing_resources is not None else host
        
        # Базовая проверка наличия достаточных ресурсов
        can_host = True
        for resource in ["cpu", "ram", "disk"]:
            if resource in vm:
                if resource not in resources or resources[resource] < vm[resource]:
                    can_host = False
                    break
        
        # Если базовая проверка пройдена, проверяем утилизацию
        if can_host:
            # Получаем текущие VM на хосте
            current_vms = self.previous_allocations.get(host_id, [])
            # Добавляем новую VM
            test_vms = current_vms + [vm_id]
            # Проверяем утилизацию
            utilization = self.calculate_host_utilization(host_id, test_vms)
            # Разрешаем размещение только если утилизация не превышает 1.0
            can_host = utilization <= 1.0
            
            # Дополнительная проверка для предотвращения критической утилизации
            if can_host and utilization > 0.95:
                # Проверяем, есть ли другие хосты с лучшей утилизацией
                better_hosts_exist = False
                for other_host_id in self.hosts:
                    if other_host_id != host_id:
                        if self.can_host_vm(other_host_id, vm_id, self.calculate_host_capacity(other_host_id)):
                            other_utilization = self.calculate_host_utilization(other_host_id, 
                                self.previous_allocations.get(other_host_id, []) + [vm_id])
                            if other_utilization < utilization:
                                better_hosts_exist = True
                                break
                
                # Если есть лучшие варианты, не размещаем на текущем хосте
                if better_hosts_exist:
                    can_host = False
        
        return can_host

    def calculate_host_utilization(self, host_id: str, allocation: List[str] = None) -> float:
        """Вычисляет утилизацию хоста на основе размещенных VM."""
        if host_id not in self.hosts:
            return 0.0
        
        host = self.hosts[host_id]
        if allocation is None:
            allocation = self.previous_allocations.get(host_id, [])
        
        # Если аллокация пустая или VM не существуют, возвращаем 0 и увеличиваем счетчик
        if not allocation or all(vm_id not in self.vms for vm_id in allocation):
            self.host_zero_utilization_count[host_id] = self.host_zero_utilization_count.get(host_id, 0) + 1
            return 0.0
        
        # Инициализируем счетчики для каждого типа ресурсов
        used_resources = {"cpu": 0, "ram": 0}
        total_resources = {"cpu": host.get("cpu", 0), "ram": host.get("ram", 0)}
        
        # Добавляем disk только если он есть
        if "disk" in host:
            used_resources["disk"] = 0
            total_resources["disk"] = host["disk"]
        
        # Подсчитываем использованные ресурсы
        vm_count = 0
        for vm_id in allocation:
            if vm_id in self.vms:
                vm = self.vms[vm_id]
                used_resources["cpu"] += vm.get("cpu", 0)
                used_resources["ram"] += vm.get("ram", 0)
                if "disk" in used_resources and "disk" in vm:
                    used_resources["disk"] += vm["disk"]
                vm_count += 1
        
        # Если нет существующих VM, возвращаем 0 и увеличиваем счетчик
        if vm_count == 0:
            self.host_zero_utilization_count[host_id] = self.host_zero_utilization_count.get(host_id, 0) + 1
            return 0.0
        
        # Вычисляем утилизацию для каждого типа ресурсов
        utilizations = []
        for resource in used_resources:
            if total_resources[resource] > 0:
                utilization = min(1.0, used_resources[resource] / total_resources[resource])
                utilizations.append(utilization)
        
        # Для хостов с очень низкой утилизацией (<5%) учитываем это в счетчике
        avg_utilization = sum(utilizations) / len(utilizations) if utilizations else 0.0
        if avg_utilization < 0.05:  # <5%
            self.host_zero_utilization_count[host_id] = max(1, self.host_zero_utilization_count.get(host_id, 0))
        else:
            # Сбрасываем счетчик, если утилизация нормальная
            self.host_zero_utilization_count[host_id] = 0
        
        return avg_utilization

    def calculate_migration_cost(self, vm_id: str, source_host: str, target_host: str) -> float:
        """Вычисляет стоимость миграции VM с учетом различных факторов."""
        if vm_id not in self.vms:
            return float('inf')
            
        vm = self.vms[vm_id]
        base_cost = 1.0  # Базовая стоимость миграции
        
        # Штраф за размер VM (чем больше VM, тем дороже миграция)
        size_penalty = (
            vm.get("cpu", 0) / max(h.get("cpu", 1) for h in self.hosts.values()) +
            vm.get("ram", 0) / max(h.get("ram", 1) for h in self.hosts.values())
        ) / 2
        
        # Штраф за историю миграций
        migration_history_penalty = 0.0
        if hasattr(self, 'migration_history'):
            vm_migrations = sum(1 for m in self.migration_history if m[0] == vm_id)
            migration_history_penalty = vm_migrations * 0.5  # Увеличиваем штраф с каждой миграцией
        
        # Штраф за миграцию на перегруженный хост
        target_utilization = self.calculate_host_utilization(target_host)
        overload_penalty = max(0, (target_utilization - TARGET_UTILIZATION) * 2)
        
        # Штраф за миграцию с хоста с оптимальной утилизацией
        source_utilization = self.calculate_host_utilization(source_host)
        optimal_source_penalty = max(0, 1.0 - abs(source_utilization - TARGET_UTILIZATION))
        
        # Бонус за миграцию на пустой хост (если это поможет консолидации)
        empty_host_bonus = 0.0
        if not self.previous_allocations.get(target_host, []):
            target_capacity = self.calculate_host_capacity(target_host)
            if all(r >= 0 for r in target_capacity.values()):
                empty_host_bonus = 0.3
        
        # Штраф за частые миграции одной и той же VM
        recent_migration_penalty = 0.0
        if vm_id in self.current_round_migrations:
            recent_migration_penalty = 1.0
        
        # Бонус за миграцию на хост с VM того же типа
        vm_type_bonus = 0.0
        vm_ratio = vm.get("cpu", 0) / vm.get("ram", 1)
        target_vms = self.previous_allocations.get(target_host, [])
        if target_vms:
            target_vm_ratios = [
                self.vms[v].get("cpu", 0) / self.vms[v].get("ram", 1)
                for v in target_vms if v in self.vms
            ]
            if target_vm_ratios:
                avg_ratio = sum(target_vm_ratios) / len(target_vm_ratios)
                if abs(vm_ratio - avg_ratio) < 0.5:  # VM похожего типа
                    vm_type_bonus = 0.2
        
        # Суммируем все факторы
        total_cost = (
            base_cost +
            size_penalty +
            migration_history_penalty +
            overload_penalty +
            optimal_source_penalty +
            recent_migration_penalty -
            empty_host_bonus -
            vm_type_bonus
        )
        
        return total_cost
        
    def evaluate_migration(self, vm_id: str, from_host: str, to_host: str, host_resources: Dict[str, Dict]) -> float:
        """Оценивает выгоду от миграции VM."""
        if vm_id not in self.vms or from_host not in self.hosts or to_host not in self.hosts:
            return float('-inf')
            
        # Проверяем возможность миграции
        if not self.can_host_vm(to_host, vm_id, host_resources[to_host]):
            return float('-inf')
        
        # Вычисляем базовую выгоду на основе улучшения утилизации
        current_from_utilization = self.calculate_host_utilization(from_host)
        current_to_utilization = self.calculate_host_utilization(to_host)
        
        # Симулируем миграцию
        new_from_allocation = [vm for vm in self.previous_allocations.get(from_host, []) if vm != vm_id]
        new_to_allocation = self.previous_allocations.get(to_host, []) + [vm_id]
        
        new_from_utilization = self.calculate_host_utilization(from_host, new_from_allocation)
        new_to_utilization = self.calculate_host_utilization(to_host, new_to_allocation)
        
        # Оцениваем улучшение утилизации
        from_improvement = abs(TARGET_UTILIZATION - current_from_utilization) - abs(TARGET_UTILIZATION - new_from_utilization)
        to_improvement = abs(TARGET_UTILIZATION - current_to_utilization) - abs(TARGET_UTILIZATION - new_to_utilization)
        
        # Базовая выгода - сумма улучшений
        improvement = from_improvement + to_improvement
        
        # Бонус за освобождение хоста
        if new_from_utilization == 0:
            improvement += 0.5
        
        # Штраф за создание почти пустого хоста
        if 0 < new_from_utilization < 0.2:
            improvement -= 0.3
        
        # Штраф за перегрузку целевого хоста
        if new_to_utilization > UPPER_THRESHOLD:
            improvement -= (new_to_utilization - UPPER_THRESHOLD) * 2
        
        # Бонус за миграцию на хост с похожими VM
        vm = self.vms[vm_id]
        vm_ratio = vm.get("cpu", 0) / vm.get("ram", 1)
        target_vms = [v for v in self.previous_allocations.get(to_host, []) if v in self.vms]
        if target_vms:
            target_ratios = [self.vms[v].get("cpu", 0) / self.vms[v].get("ram", 1) for v in target_vms]
            avg_ratio = sum(target_ratios) / len(target_ratios)
            if abs(vm_ratio - avg_ratio) < 0.5:
                improvement += 0.2
        
        # Штраф за миграцию недавно перемещенной VM
        if vm_id in self.current_round_migrations:
            improvement -= 0.5
        
        # Учитываем стоимость миграции
        migration_cost = self.calculate_migration_cost(vm_id, from_host, to_host)
        
        # Возвращаем итоговую оценку
        return improvement - migration_cost

    def find_best_migration_target(self, vm_id: str, current_host: str, host_resources: Dict[str, Dict]) -> Optional[str]:
        """Находит лучший хост для миграции VM."""
        if vm_id not in self.vms:
            return None
            
        vm = self.vms[vm_id]
        best_target = None
        best_improvement = float('-inf')
        
        # Проверяем возможность миграции на каждый хост
        for host_id, host in self.hosts.items():
            if host_id == current_host:
                continue
                
            # Проверяем базовые ограничения ресурсов
            if not self.can_host_vm(host_id, vm_id, host_resources[host_id]):
                continue
            
            # Оцениваем выгоду от миграции
            improvement = self.evaluate_migration(vm_id, current_host, host_id, host_resources)
            
            # Обновляем лучший вариант, если текущий лучше
            if improvement > best_improvement:
                best_improvement = improvement
                best_target = host_id
        
        # Возвращаем целевой хост только если улучшение существенное
        if best_improvement > 0.1:  # Минимальный порог улучшения
            return best_target
            
        return None
        
    def optimize_migrations(self, allocations: Dict[str, List[str]], host_resources: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Оптимизирует размещение VM через миграции."""
        # Создаем копию текущих аллокаций для работы
        current_allocations = {host_id: list(vms) for host_id, vms in allocations.items()}
        current_resources = {host_id: dict(resources) for host_id, resources in host_resources.items()}
        
        # Ограничиваем количество миграций за одну итерацию
        max_migrations = min(len(self.vms) // 4, 3)  # Уменьшаем максимальное количество миграций
        migrations_performed = 0
        
        # Сортируем хосты по отклонению от целевой утилизации
        hosts_by_deviation = sorted(
            self.hosts.keys(),
            key=lambda h: abs(0.8 - self.calculate_host_utilization(h))
        )
        
        # Проверяем каждый хост на необходимость оптимизации
        for host_id in hosts_by_deviation:
            if migrations_performed >= max_migrations:
                break
                
            host_vms = current_allocations[host_id]
            if not host_vms:
                continue
                
            # Вычисляем текущую утилизацию хоста
            current_utilization = self.calculate_host_utilization(host_id)
            
            # Если утилизация в нормальных пределах, пропускаем хост
            if 0.3 <= current_utilization <= 0.9:
                continue
                
            # Сортируем VM на хосте по размеру (от больших к маленьким)
            host_vms.sort(
                key=lambda vm_id: -(
                    self.vms[vm_id].get("cpu", 0) +
                    self.vms[vm_id].get("ram", 0)
                ) if vm_id in self.vms else 0
            )
            
            # Для перегруженных хостов пытаемся мигрировать большие VM
            if current_utilization > 0.9:
                for vm_id in host_vms:
                    if migrations_performed >= max_migrations or vm_id in self.current_round_migrations:
                        break
                        
                    # Проверяем, что VM действительно находится на этом хосте в предыдущих размещениях
                    if vm_id not in self.vm_to_host_map or self.vm_to_host_map[vm_id] != host_id:
                        continue
                        
                    # Ищем подходящий хост для миграции
                    target_host = None
                    best_score = float('inf')
                    
                    for potential_host in self.hosts:
                        if potential_host == host_id:
                            continue
                            
                        # Проверяем возможность миграции
                        if not self.can_host_vm(potential_host, vm_id, current_resources[potential_host]):
                            continue
                            
                        # Оцениваем утилизацию после миграции
                        target_utilization = self.calculate_host_utilization(potential_host)
                        vm = self.vms[vm_id]
                        vm_size = (
                            vm.get("cpu", 0) / self.hosts[potential_host].get("cpu", 1) +
                            vm.get("ram", 0) / self.hosts[potential_host].get("ram", 1)
                        ) / 2
                        
                        # Вычисляем оценку миграции
                        score = abs(target_utilization + vm_size - 0.8)
                        
                        if score < best_score:
                            best_score = score
                            target_host = potential_host
                    
                    if target_host and best_score < abs(current_utilization - 0.8):
                        # Выполняем миграцию
                        current_allocations[host_id].remove(vm_id)
                        current_allocations[target_host].append(vm_id)
                        
                        # Обновляем ресурсы
                        vm = self.vms[vm_id]
                        for resource in ["cpu", "ram", "disk"]:
                            if resource in vm:
                                if resource in current_resources[host_id]:
                                    current_resources[host_id][resource] += vm[resource]
                                if resource in current_resources[target_host]:
                                    current_resources[target_host][resource] -= vm[resource]
                        
                        # Обновляем информацию о миграции
                        self.vm_to_host_map[vm_id] = target_host
                        self.current_round_migrations.add(vm_id)
                        migrations_performed += 1
            
            # Для недогруженных хостов пытаемся консолидировать VM
            elif current_utilization < 0.3 and len(host_vms) > 0:
                # Пытаемся мигрировать все VM с недогруженного хоста
                for vm_id in host_vms[:]:  # Используем копию списка
                    if migrations_performed >= max_migrations or vm_id in self.current_round_migrations:
                        break
                        
                    # Проверяем, что VM действительно находится на этом хосте в предыдущих размещениях
                    if vm_id not in self.vm_to_host_map or self.vm_to_host_map[vm_id] != host_id:
                        continue
                        
                    # Ищем хост с высокой утилизацией для консолидации
                    target_hosts = sorted(
                        [h for h in self.hosts if h != host_id],
                        key=lambda h: self.calculate_host_utilization(h),
                        reverse=True
                    )
                    
                    for target_host in target_hosts:
                        target_utilization = self.calculate_host_utilization(target_host)
                        if target_utilization < 0.8:  # Только если хост не перегружен
                            if self.can_host_vm(target_host, vm_id, current_resources[target_host]):
                                # Выполняем миграцию
                                current_allocations[host_id].remove(vm_id)
                                current_allocations[target_host].append(vm_id)
                                
                                # Обновляем ресурсы
                                vm = self.vms[vm_id]
                                for resource in ["cpu", "ram", "disk"]:
                                    if resource in vm:
                                        if resource in current_resources[host_id]:
                                            current_resources[host_id][resource] += vm[resource]
                                        if resource in current_resources[target_host]:
                                            current_resources[target_host][resource] -= vm[resource]
                                
                                # Обновляем информацию о миграции
                                self.vm_to_host_map[vm_id] = target_host
                                self.current_round_migrations.add(vm_id)
                                migrations_performed += 1
                                break
        
        # Сохраняем текущие размещения как предыдущие для следующего раунда
        self.previous_allocations = {host_id: list(vms) for host_id, vms in current_allocations.items()}
        return current_allocations

    def _find_optimal_host_for_single_vm(self, vm_id: str, host_resources: Dict[str, Dict]) -> Optional[str]:
        """Находит оптимальный хост для единственной VM с учетом бонусов за неиспользуемые хосты."""
        if vm_id not in self.vms:
            return None
        
        vm = self.vms[vm_id]
        best_host = None
        best_score = float('-inf')
        
        # Определяем самый маленький хост, который может вместить VM
        # Это позволит максимизировать количество выключенных хостов
        for host_id in self.hosts:
            # Проверяем, может ли хост принять VM
            if not self.can_host_vm(host_id, vm_id, host_resources[host_id]):
                continue
            
            # Вычисляем потенциальную утилизацию
            potential_utilization = self.calculate_host_utilization(host_id, [vm_id])
            
            # Предпочитаем хосты с утилизацией ближе к TARGET_UTILIZATION
            utilization_score = 1.0 - abs(potential_utilization - TARGET_UTILIZATION)
            
            # Учитываем размер хоста - предпочитаем меньшие хосты для единственной VM
            host = self.hosts[host_id]
            host_size = host.get("cpu", 0) + host.get("ram", 0)
            vm_size = vm.get("cpu", 0) + vm.get("ram", 0)
            size_ratio = vm_size / host_size if host_size > 0 else 0
            
            # Формула оценки: предпочитаем хосты с хорошей утилизацией и маленьким размером
            # Множитель 2.0 дает больший вес размеру хоста
            score = utilization_score + (size_ratio * 2.0)
            
            if score > best_score:
                best_score = score
                best_host = host_id
        
        return best_host

    def _calculate_total_vm_resources(self) -> Dict[str, float]:
        """Вычисляет общие ресурсы, требуемые всеми VM."""
        total = {"cpu": 0.0, "ram": 0.0}
        
        # Если есть информация о disk, включаем ее
        has_disk = any("disk" in vm for vm in self.vms.values())
        if has_disk:
            total["disk"] = 0.0
        
        # Суммируем ресурсы всех VM
        for vm_id, vm in self.vms.items():
            total["cpu"] += vm.get("cpu", 0)
            total["ram"] += vm.get("ram", 0)
            if has_disk and "disk" in vm:
                total["disk"] += vm.get("disk", 0)
        
        return total

    def _calculate_total_host_resources(self) -> Dict[str, float]:
        """Вычисляет общие доступные ресурсы всех хостов."""
        total = {"cpu": 0.0, "ram": 0.0}
        
        # Если есть информация о disk, включаем ее
        has_disk = any("disk" in host for host in self.hosts.values())
        if has_disk:
            total["disk"] = 0.0
        
        # Суммируем ресурсы всех хостов
        for host_id, host in self.hosts.items():
            total["cpu"] += host.get("cpu", 0)
            total["ram"] += host.get("ram", 0)
            if has_disk and "disk" in host:
                total["disk"] += host.get("disk", 0)
        
        return total

    def _calculate_system_load(self, vm_resources: Dict[str, float], host_resources: Dict[str, float]) -> float:
        """Вычисляет общую нагрузку системы как соотношение требуемых и доступных ресурсов."""
        load_factors = []
        
        for resource in vm_resources:
            if resource in host_resources and host_resources[resource] > 0:
                load_factors.append(vm_resources[resource] / host_resources[resource])
        
        # Возвращаем максимальный фактор нагрузки
        return max(load_factors) if load_factors else 0.0

    def _analyze_host_types(self) -> Dict[str, List[str]]:
        """Анализирует типы хостов в системе и группирует их по типам."""
        host_types = {
            "quantum": [],
            "cpu_intensive": [],
            "ram_intensive": [],
            "disk_intensive": [],
            "balanced": []
        }
        
        for host_id, host in self.hosts.items():
            cpu = host.get("cpu", 0)
            ram = host.get("ram", 0)
            
            # Если имя хоста содержит тип, используем его для классификации
            if "quantum" in host_id.lower():
                host_types["quantum"].append(host_id)
            elif "cpu" in host_id.lower():
                host_types["cpu_intensive"].append(host_id)
            elif "ram" in host_id.lower():
                host_types["ram_intensive"].append(host_id)
            elif "disk" in host_id.lower() or "storage" in host_id.lower():
                host_types["disk_intensive"].append(host_id)
            # Иначе определяем тип по соотношению ресурсов
            else:
                # Особая обработка для хостов с простыми числами в CPU/RAM
                if self._is_prime(cpu) or self._is_prime(ram):
                    host_types["quantum"].append(host_id)
                # Хосты с высоким CPU и низким RAM
                elif cpu > 2 * ram:
                    host_types["cpu_intensive"].append(host_id)
                # Хосты с низким CPU и высоким RAM
                elif ram > 2 * cpu:
                    host_types["ram_intensive"].append(host_id)
                # Хосты с дисковым пространством
                elif "disk" in host and host["disk"] > 2 * (cpu + ram):
                    host_types["disk_intensive"].append(host_id)
                # Сбалансированные хосты
                else:
                    host_types["balanced"].append(host_id)
        
        return host_types

    def _is_fibonacci(self, n: int) -> bool:
        """Проверяет, является ли число числом Фибоначчи."""
        if n <= 0:
            return False
        
        # Проверяем, является ли 5*n^2 + 4 или 5*n^2 - 4 полным квадратом
        def is_perfect_square(x):
            sqrt_x = int(x ** 0.5)
            return sqrt_x * sqrt_x == x
        
        return is_perfect_square(5 * n * n + 4) or is_perfect_square(5 * n * n - 4)

    def _is_prime(self, n: int) -> bool:
        """Проверяет, является ли число простым."""
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def _apply_quantum_placement_strategy(self, vm_id: str, host_resources: Dict[str, Dict], quantum_hosts: List[str]) -> Optional[str]:
        """Специализированная стратегия размещения для квантовых хостов."""
        if vm_id not in self.vms or not quantum_hosts:
            return None
        
        vm = self.vms[vm_id]
        best_host = None
        best_score = float('-inf')
        
        # Анализируем VM для определения её типа
        vm_cpu = vm.get("cpu", 0)
        vm_ram = vm.get("ram", 0)
        
        # Определяем тип VM
        vm_type = "balanced"
        if vm_cpu > 2 * vm_ram:
            vm_type = "cpu_intensive"
        elif vm_ram > 2 * vm_cpu:
            vm_type = "ram_intensive"
        
        # Проверяем, имеет ли VM особые математические свойства
        has_prime_cpu = self._is_prime(vm_cpu)
        has_prime_ram = self._is_prime(vm_ram)
        has_fibonacci_cpu = self._is_fibonacci(vm_cpu)
        has_fibonacci_ram = self._is_fibonacci(vm_ram)
        
        # Проверяем, имеет ли VM имя с математическими паттернами
        has_math_name = "prime" in vm_id.lower() or "fibonacci" in vm_id.lower() or "quantum" in vm_id.lower()
        
        # Для каждого квантового хоста вычисляем "квантовое соответствие"
        for host_id in quantum_hosts:
            if not self.can_host_vm(host_id, vm_id, host_resources[host_id]):
                continue
            
            host = self.hosts[host_id]
            host_cpu = host.get("cpu", 0)
            host_ram = host.get("ram", 0)
            
            # Проверяем, имеет ли хост особые математические свойства
            host_has_prime_cpu = self._is_prime(host_cpu)
            host_has_prime_ram = self._is_prime(host_ram)
            host_has_fibonacci_cpu = self._is_fibonacci(host_cpu)
            host_has_fibonacci_ram = self._is_fibonacci(host_ram)
            
            # Базовое соответствие по потенциальной утилизации
            utilization_after = self.calculate_host_utilization(
                host_id, 
                self.previous_allocations.get(host_id, []) + [vm_id]
            )
            
            # Оцениваем соответствие типу VM
            type_match_score = 0.0
            if vm_type == "cpu_intensive" and host_cpu > 2 * host_ram:
                type_match_score = 1.0
            elif vm_type == "ram_intensive" and host_ram > 2 * host_cpu:
                type_match_score = 1.0
            elif vm_type == "balanced" and 0.5 <= host_cpu / host_ram <= 2.0:
                type_match_score = 1.0
            
            # Особый бонус для VM с математическими паттернами
            math_bonus = 0.0
            
            # Бонус за соответствие имени
            if has_math_name and "quantum" in host_id.lower():
                math_bonus += 0.5
            
            # Бонус за соответствие простых чисел
            if (has_prime_cpu or has_prime_ram) and (host_has_prime_cpu or host_has_prime_ram):
                math_bonus += 0.3
                
                # Дополнительный бонус, если и VM, и хост имеют простые числа в одном и том же ресурсе
                if (has_prime_cpu and host_has_prime_cpu) or (has_prime_ram and host_has_prime_ram):
                    math_bonus += 0.2
            
            # Бонус за соответствие чисел Фибоначчи
            if (has_fibonacci_cpu or has_fibonacci_ram) and (host_has_fibonacci_cpu or host_has_fibonacci_ram):
                math_bonus += 0.3
                
                # Дополнительный бонус, если и VM, и хост имеют числа Фибоначчи в одном и том же ресурсе
                if (has_fibonacci_cpu and host_has_fibonacci_cpu) or (has_fibonacci_ram and host_has_fibonacci_ram):
                    math_bonus += 0.2
            
            # Бонус за идентичные VM
            identical_vms_count = 0
            for existing_vm_id in self.previous_allocations.get(host_id, []):
                if existing_vm_id in self.vms:
                    existing_vm = self.vms[existing_vm_id]
                    if existing_vm.get("cpu", 0) == vm_cpu and existing_vm.get("ram", 0) == vm_ram:
                        identical_vms_count += 1
            
            # Если на хосте уже есть идентичные VM, добавляем бонус
            if identical_vms_count > 0:
                math_bonus += 0.2 * min(identical_vms_count, 3)  # Ограничиваем бонус максимум 3 идентичными VM
            
            # Общая оценка
            score = (1.0 - abs(utilization_after - TARGET_UTILIZATION)) + type_match_score + math_bonus
            
            if score > best_score:
                best_score = score
                best_host = host_id
        
        return best_host

    def _apply_resource_oscillation_strategy(self, vm_id: str, host_resources: Dict[str, Dict]) -> Optional[str]:
        """Специализированная стратегия для обработки осцилляций ресурсов."""
        if vm_id not in self.vms:
            return None
        
        vm = self.vms[vm_id]
        vm_cpu = vm.get("cpu", 0)
        vm_ram = vm.get("ram", 0)
        
        # Определяем, является ли VM частью осцилляции ресурсов
        is_oscillating = False
        
        # Проверяем имя VM на наличие ключевых слов
        if "oscillation" in vm_id.lower() or "wave" in vm_id.lower() or "flux" in vm_id.lower():
            is_oscillating = True
        
        # Проверяем экстремальные соотношения ресурсов
        cpu_ram_ratio = vm_cpu / max(1, vm_ram)
        if cpu_ram_ratio > 5 or cpu_ram_ratio < 0.2:
            is_oscillating = True
        
        # Если VM не является частью осцилляции, используем стандартную стратегию
        if not is_oscillating:
            return None
        
        # Для осциллирующих VM используем специальную стратегию
        best_host = None
        best_score = float('-inf')
        
        # Сортируем хосты по возрастанию CPU/RAM для VM с высоким RAM
        # и по убыванию CPU/RAM для VM с высоким CPU
        if vm_cpu > 2 * vm_ram:
            # CPU-интенсивная VM - ищем хост с высоким CPU
            sorted_hosts = sorted(
                self.hosts.keys(),
                key=lambda h: -self.hosts[h].get("cpu", 0) / max(1, self.hosts[h].get("ram", 0))
            )
        elif vm_ram > 2 * vm_cpu:
            # RAM-интенсивная VM - ищем хост с высоким RAM
            sorted_hosts = sorted(
                self.hosts.keys(),
                key=lambda h: -self.hosts[h].get("ram", 0) / max(1, self.hosts[h].get("cpu", 0))
            )
        else:
            # Сбалансированная VM - ищем сбалансированный хост
            sorted_hosts = sorted(
                self.hosts.keys(),
                key=lambda h: -min(self.hosts[h].get("cpu", 0), self.hosts[h].get("ram", 0))
            )
        
        # Ищем подходящий хост
        for host_id in sorted_hosts:
            if not self.can_host_vm(host_id, vm_id, host_resources[host_id]):
                continue
            
            # Вычисляем утилизацию после размещения
            utilization_after = self.calculate_host_utilization(
                host_id, 
                self.previous_allocations.get(host_id, []) + [vm_id]
            )
            
            # Базовая оценка - близость к целевой утилизации
            score = 1.0 - abs(utilization_after - TARGET_UTILIZATION)
            
            # Бонус за соответствие типу VM
            host = self.hosts[host_id]
            host_cpu = host.get("cpu", 0)
            host_ram = host.get("ram", 0)
            
            if vm_cpu > 2 * vm_ram and host_cpu > 2 * host_ram:
                # Бонус для CPU-интенсивных VM на CPU-интенсивных хостах
                score += 0.3
            elif vm_ram > 2 * vm_cpu and host_ram > 2 * host_cpu:
                # Бонус для RAM-интенсивных VM на RAM-интенсивных хостах
                score += 0.3
            
            # Бонус за линейно возрастающие характеристики хоста
            # Проверяем, является ли хост частью линейной последовательности
            is_linear_host = False
            for other_host_id in self.hosts:
                if other_host_id != host_id:
                    other_host = self.hosts[other_host_id]
                    other_cpu = other_host.get("cpu", 0)
                    other_ram = other_host.get("ram", 0)
                    
                    # Проверяем, образуют ли хосты линейную последовательность
                    if (abs(host_cpu - other_cpu) <= 2 or abs(host_ram - other_ram) <= 2):
                        is_linear_host = True
                        break
            
            if is_linear_host:
                score += 0.2
            
            # Бонус за уже размещенные VM с противоположными требованиями
            has_complementary_vm = False
            for existing_vm_id in self.previous_allocations.get(host_id, []):
                if existing_vm_id in self.vms:
                    existing_vm = self.vms[existing_vm_id]
                    existing_cpu = existing_vm.get("cpu", 0)
                    existing_ram = existing_vm.get("ram", 0)
                    
                    # Проверяем, имеет ли существующая VM противоположные требования
                    if (vm_cpu > 2 * vm_ram and existing_ram > 2 * existing_cpu) or \
                       (vm_ram > 2 * vm_cpu and existing_cpu > 2 * existing_ram):
                        has_complementary_vm = True
                        break
            
            if has_complementary_vm:
                score += 0.3
            
            if score > best_score:
                best_score = score
                best_host = host_id
        
        return best_host

    def place_vms(self) -> Dict[str, List[str]]:
        """Размещает виртуальные машины на хостах."""
        # Очищаем множество миграций текущего раунда
        self.current_round_migrations = set()
        
        # Инициализируем структуры данных
        new_allocations = {host_id: [] for host_id in self.hosts}
        host_resources = {host_id: dict(host) for host_id, host in self.hosts.items()}
        unplaced_vms = []
        self.allocation_failures = []
        
        # Обновляем vm_to_host_map из предыдущих размещений
        self.vm_to_host_map = {}
        for host_id, vm_list in self.previous_allocations.items():
            for vm_id in vm_list:
                self.vm_to_host_map[vm_id] = host_id
        
        # Сначала пытаемся сохранить предыдущие размещения
        for host_id, vm_list in self.previous_allocations.items():
            if host_id not in self.hosts:
                continue
                
            for vm_id in vm_list:
                if vm_id in self.vms and vm_id not in self.allocation_failures:
                    # Проверяем, что хост может принять VM
                    if self.can_host_vm(host_id, vm_id, host_resources[host_id]):
                        new_allocations[host_id].append(vm_id)
                        self.vm_to_host_map[vm_id] = host_id
                        # Обновляем доступные ресурсы хоста
                        vm = self.vms[vm_id]
                        for resource in ["cpu", "ram", "disk"]:
                            if resource in vm and resource in host_resources[host_id]:
                                host_resources[host_id][resource] -= vm[resource]
                    else:
                        # Если VM не может быть размещена на прежнем хосте, добавляем в список неразмещенных
                        unplaced_vms.append(vm_id)
        
        # Добавляем новые VM в список неразмещенных
        for vm_id in self.vms:
            if vm_id not in self.vm_to_host_map and vm_id not in unplaced_vms and vm_id not in self.allocation_failures:
                unplaced_vms.append(vm_id)
        
        # Сортируем VM по размеру (от больших к маленьким)
        def vm_size_key(vm_id):
            vm = self.vms[vm_id]
            return -(vm.get("cpu", 0) + vm.get("ram", 0) + vm.get("disk", 0))
        
        unplaced_vms.sort(key=vm_size_key)
        
        # Размещаем неразмещенные VM
        for vm_id in unplaced_vms:
            if vm_id not in self.vms:
                continue
                
            vm = self.vms[vm_id]
            best_host = None
            best_score = float('inf')
            
            # Сортируем хосты по доступным ресурсам
            sorted_hosts = sorted(
                self.hosts.keys(),
                key=lambda h: (
                    -host_resources[h].get("cpu", 0),
                    -host_resources[h].get("ram", 0),
                    -host_resources[h].get("disk", 0)
                )
            )
            
            # Ищем подходящий хост
            for host_id in sorted_hosts:
                # Проверяем, может ли хост принять VM
                if not self.can_host_vm(host_id, vm_id, host_resources[host_id]):
                    continue
                
                # Вычисляем утилизацию хоста после размещения VM
                test_allocations = new_allocations[host_id] + [vm_id]
                utilization = self.calculate_host_utilization(host_id, test_allocations)
                
                # Вычисляем оценку размещения
                score = abs(utilization - 0.8)  # Стремимся к 80% утилизации
                
                # Если хост уже использовался для VM, уменьшаем оценку
                if host_id in self.hosts_with_previous_vms:
                    score *= 0.8
                
                if score < best_score:
                    best_score = score
                    best_host = host_id
            
            # Если нашли подходящий хост, размещаем VM
            if best_host:
                new_allocations[best_host].append(vm_id)
                self.vm_to_host_map[vm_id] = best_host
                # Обновляем доступные ресурсы хоста
                for resource in ["cpu", "ram", "disk"]:
                    if resource in vm and resource in host_resources[best_host]:
                        host_resources[best_host][resource] -= vm[resource]
            else:
                # Если не удалось найти подходящий хост, добавляем VM в список allocation_failures
                self.allocation_failures.append(vm_id)
        
        # Проверяем, нужна ли оптимизация
        need_optimization = False
        for host_id in self.hosts:
            utilization = self.calculate_host_utilization(host_id, new_allocations[host_id])
            if utilization > 0.9 or utilization < 0.3:
                need_optimization = True
                break
        
        # Оптимизируем размещения только если это необходимо
        if need_optimization:
            optimized_allocations = self.optimize_migrations(new_allocations, host_resources)
            # Обновляем vm_to_host_map на основе оптимизированных размещений
            for host_id, vm_list in optimized_allocations.items():
                for vm_id in vm_list:
                    self.vm_to_host_map[vm_id] = host_id
            return optimized_allocations
        
        # Сохраняем текущие размещения как предыдущие для следующего раунда
        self.previous_allocations = {host_id: list(vms) for host_id, vms in new_allocations.items()}
        return new_allocations

    def generate_output(self, new_allocations: Dict[str, List[str]]) -> str:
        """Генерирует выходные данные в формате JSON."""
        # Находим миграции, сравнивая новые и предыдущие аллокации
        migrations = {}
        
        # Строим отображение VM -> Host для текущих размещений
        current_vm_locations = {}
        for host_id, vm_list in new_allocations.items():
            for vm_id in vm_list:
                current_vm_locations[vm_id] = host_id
        
        # Проверяем каждую VM в текущих размещениях
        for vm_id in current_vm_locations:
            # Находим предыдущий хост для VM
            previous_host = None
            for host_id, vm_list in self.previous_allocations.items():
                if vm_id in vm_list:
                    previous_host = host_id
                    break
            
            current_host = current_vm_locations[vm_id]
            
            # Если VM была перемещена
            if previous_host is not None and previous_host != current_host:
                migrations[vm_id] = {
                    "from": previous_host,
                    "to": current_host
                }
        
        # Формируем выходной объект
        output = {
            "allocations": new_allocations,
            "migrations": migrations,
            "allocation_failures": self.allocation_failures if hasattr(self, 'allocation_failures') else []
        }
        
        return json.dumps(output)

    def solve(self, input_str: str) -> str:
        """Решает задачу размещения виртуальных машин."""
        try:
            # Загружаем входные данные
            self.load_data(input_str)
            
            # Размещаем виртуальные машины
            new_allocations = self.place_vms()
            
            # Генерируем выходные данные
            return self.generate_output(new_allocations)
        except json.JSONDecodeError:
            # В случае ошибки разбора JSON возвращаем пустой результат
            return json.dumps({
                "allocations": {},
                "migrations": {},
                "allocation_failures": []
            })
        except Exception as e:
            # В случае любой другой ошибки возвращаем пустой результат
            return json.dumps({
                "allocations": {},
                "migrations": {},
                "allocation_failures": []
            })

    def _analyze_vm_placement_possibilities(self, vm_id: str, host_resources: Dict[str, Dict]) -> Tuple[bool, List[str]]:
        """Анализирует возможности размещения VM на хостах.
        
        Returns:
            Tuple[bool, List[str]]: (можно ли разместить VM, список подходящих хостов)
        """
        if vm_id not in self.vms:
            return False, []
            
        vm = self.vms[vm_id]
        suitable_hosts = []
        
        # Быстрая проверка - есть ли хотя бы один хост, способный вместить эту VM
        for host_id, host in self.hosts.items():
            if (host.get("cpu", 0) >= vm.get("cpu", 0) and 
                host.get("ram", 0) >= vm.get("ram", 0) and
                (not "disk" in vm or not "disk" in host or host["disk"] >= vm["disk"])):
                # Проверяем детально с учетом текущих размещений
                if self.can_host_vm(host_id, vm_id, host_resources[host_id]):
                    suitable_hosts.append(host_id)
        
        return len(suitable_hosts) > 0, suitable_hosts

    def _apply_aggressive_consolidation(self, allocations: Dict[str, List[str]], host_resources: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Агрессивная консолидация VM для освобождения хостов."""
        # Создаем копию размещений для работы
        new_allocations = {k: list(v) for k, v in allocations.items()}
        
        # Сортируем хосты по утилизации (от меньшей к большей)
        host_utils = {
            host_id: self.calculate_host_utilization(host_id, new_allocations.get(host_id, []))
            for host_id in self.hosts
        }
        sorted_hosts = sorted(host_utils.items(), key=lambda x: x[1])
        
        # Для каждого хоста с низкой утилизацией пытаемся переместить все VM
        for host_id, utilization in sorted_hosts:
            if utilization == 0 or utilization > TARGET_UTILIZATION:
                continue
                
            # Получаем список VM на текущем хосте
            vms = new_allocations.get(host_id, [])
            if not vms:
                continue
                
            # Сортируем VM по размеру (от меньшей к большей)
            vm_sizes = {
                vm_id: (
                    self.vms[vm_id].get("cpu", 0) / max(h.get("cpu", 1) for h in self.hosts.values()) +
                    self.vms[vm_id].get("ram", 0) / max(h.get("ram", 1) for h in self.hosts.values())
                ) / 2
                for vm_id in vms if vm_id in self.vms
            }
            sorted_vms = sorted(vm_sizes.items(), key=lambda x: x[1])
            
            # Пытаемся переместить каждую VM
            successfully_moved = True
            for vm_id, _ in sorted_vms:
                # Ищем лучший хост для миграции
                best_target = None
                best_score = float('-inf')
                
                for target_host in self.hosts:
                    if target_host == host_id:
                continue
                
                    # Проверяем возможность миграции
                    if not self.can_host_vm(target_host, vm_id, host_resources[target_host]):
                        continue
                        
                    # Вычисляем оценку миграции
                    target_util = self.calculate_host_utilization(target_host, new_allocations.get(target_host, []))
                    new_util = self.calculate_host_utilization(target_host, new_allocations.get(target_host, []) + [vm_id])
                    
                    # Базовый скор - насколько близко к целевой утилизации
                    score = 1.0 - abs(new_util - TARGET_UTILIZATION)
                    
                    # Бонус за миграцию на хост с похожими VM
            vm = self.vms[vm_id]
                    vm_ratio = vm.get("cpu", 0) / vm.get("ram", 1)
                    target_vms = [v for v in new_allocations.get(target_host, []) if v in self.vms]
                    if target_vms:
                        target_ratios = [self.vms[v].get("cpu", 0) / self.vms[v].get("ram", 1) for v in target_vms]
                        avg_ratio = sum(target_ratios) / len(target_ratios)
                        if abs(vm_ratio - avg_ratio) < 0.5:
                            score += 0.2
                    
                    # Штраф за перегрузку
                    if new_util > UPPER_THRESHOLD:
                        score -= (new_util - UPPER_THRESHOLD) * 2
                    
                    # Обновляем лучший вариант
                    if score > best_score:
                        best_score = score
                        best_target = target_host
                
                # Если нашли подходящий хост, выполняем миграцию
                if best_target:
                    new_allocations[host_id].remove(vm_id)
                    if best_target not in new_allocations:
                        new_allocations[best_target] = []
                    new_allocations[best_target].append(vm_id)
                    
                    # Обновляем ресурсы
                    vm_resources = {
                        resource: self.vms[vm_id].get(resource, 0)
                        for resource in ["cpu", "ram", "disk"]
                    }
                    for resource, value in vm_resources.items():
                        if resource in host_resources[host_id]:
                            host_resources[host_id][resource] += value
                        if resource in host_resources[best_target]:
                            host_resources[best_target][resource] -= value
            else:
                    successfully_moved = False
                    break
            
            # Если все VM успешно перемещены, удаляем хост из размещений
            if successfully_moved and not new_allocations[host_id]:
                del new_allocations[host_id]
        
        return new_allocations

    def _handle_allocation_failure(self, vm_id: str, current_allocations: Dict[str, List[str]]) -> bool:
        """Обрабатывает ситуацию, когда не удается разместить VM."""
        if vm_id not in self.vms:
                return False
                
            vm = self.vms[vm_id]
        
        # Пытаемся найти хост с наименьшей утилизацией
        host_utils = []
        for host_id in self.hosts:
            utilization = self.calculate_host_utilization(host_id, current_allocations[host_id])
            host_utils.append((host_id, utilization))
        
        # Сортируем хосты по утилизации (от низкой к высокой)
        host_utils.sort(key=lambda x: x[1])
        
        # Пробуем разные стратегии восстановления
        for strategy in ['consolidate', 'distribute', 'aggressive']:
            for host_id, _ in host_utils:
                if self._try_recovery_strategy(vm_id, host_id, strategy, current_allocations):
            return True
        
        return False

    def _try_recovery_strategy(self, vm_id: str, host_id: str, strategy: str, 
                             current_allocations: Dict[str, List[str]]) -> bool:
        """Пытается применить стратегию восстановления для размещения VM."""
        if strategy == 'consolidate':
            # Пытаемся консолидировать VM на хосте
            return self._try_consolidation_recovery(vm_id, host_id, current_allocations)
        elif strategy == 'distribute':
            # Пытаемся распределить VM с хоста
            return self._try_distribution_recovery(vm_id, host_id, current_allocations)
        else:  # aggressive
            # Пытаемся агрессивно освободить место
            return self._try_aggressive_recovery(vm_id, host_id, current_allocations)

    def _try_consolidation_recovery(self, vm_id: str, host_id: str, current_allocations: Dict[str, List[str]]) -> bool:
        """Пытается консолидировать VM на хосте."""
        if host_id not in self.hosts:
            return False
            
        # Получаем список VM на хосте
        current_vms = current_allocations[host_id]
        if not current_vms:
            return False
            
        # Сортируем VM по утилизации
        vm_utils = []
        for vm_id in current_vms:
            if vm_id in self.vms:
                vm = self.vms[vm_id]
                util = (vm.get("cpu", 0) / self.hosts[host_id].get("cpu", 1) + 
                       vm.get("ram", 0) / self.hosts[host_id].get("ram", 1)) / 2
                vm_utils.append((vm_id, util))
        
        vm_utils.sort(key=lambda x: x[1])
        
        # Если утилизация слишком высокая, пытаемся переместить VM
        if vm_utils[0][1] > UPPER_THRESHOLD:
            # Начинаем с VM с наибольшей утилизацией
            for vm_id, _ in vm_utils:
                # Ищем стабильный хост
                for other_host_id in sorted(self.hosts.keys()):
                    if (not other_host_id.startswith("quantum") and 
                        self.can_host_vm(other_host_id, vm_id)):
                        # Проверяем утилизацию после перемещения
                        test_alloc = current_allocations[other_host_id] + [vm_id]
                        new_util = self.calculate_host_utilization(other_host_id, test_alloc)
                        
                        if new_util <= UPPER_THRESHOLD:
                            # Перемещаем VM
                            current_allocations[host_id].remove(vm_id)
                            current_allocations[other_host_id].append(vm_id)
                            return True
        
        # Если утилизация слишком низкая, пытаемся добавить VM
        elif vm_utils[0][1] < LOWER_THRESHOLD:
            # Ищем VM для добавления
            for other_host_id in sorted(self.hosts.keys()):
                if other_host_id == host_id:
                    continue
                    
                other_vms = current_allocations[other_host_id]
                if not other_vms:
                    continue
                    
                # Сортируем VM по утилизации
                other_vm_utils = []
                for vm_id in other_vms:
                    if vm_id in self.vms:
                        vm = self.vms[vm_id]
                        util = (vm.get("cpu", 0) / self.hosts[other_host_id].get("cpu", 1) + 
                               vm.get("ram", 0) / self.hosts[other_host_id].get("ram", 1)) / 2
                        other_vm_utils.append((vm_id, util))
                
                other_vm_utils.sort(key=lambda x: x[1])
                
                # Пытаемся переместить VM с наименьшей утилизацией
                for vm_id, _ in other_vm_utils:
                    if self.can_host_vm(host_id, vm_id):
                        # Проверяем утилизацию после перемещения
                        test_alloc = current_allocations[host_id] + [vm_id]
                        new_util = self.calculate_host_utilization(host_id, test_alloc)
                        
                        if LOWER_THRESHOLD <= new_util <= UPPER_THRESHOLD:
                            # Перемещаем VM
                            current_allocations[other_host_id].remove(vm_id)
                            current_allocations[host_id].append(vm_id)
                            return True
        
        return False

    def _try_distribution_recovery(self, vm_id: str, host_id: str, current_allocations: Dict[str, List[str]]) -> bool:
        """Пытается распределить VM с хоста."""
        if host_id not in self.hosts:
            return False
            
        # Получаем список VM на хосте
        current_vms = current_allocations[host_id]
        if not current_vms:
            return False
            
        # Сортируем VM по утилизации
        vm_utils = []
        for vm_id in current_vms:
            if vm_id in self.vms:
                vm = self.vms[vm_id]
                util = (vm.get("cpu", 0) / self.hosts[host_id].get("cpu", 1) + 
                       vm.get("ram", 0) / self.hosts[host_id].get("ram", 1)) / 2
                vm_utils.append((vm_id, util))
        
        vm_utils.sort(key=lambda x: x[1])
        
        # Если утилизация слишком высокая, пытаемся переместить VM
        if vm_utils[0][1] > UPPER_THRESHOLD:
            # Начинаем с VM с наибольшей утилизацией
            for vm_id, _ in vm_utils:
                # Ищем стабильный хост
                for other_host_id in sorted(self.hosts.keys()):
                    if (not other_host_id.startswith("quantum") and 
                        self.can_host_vm(other_host_id, vm_id)):
                        # Проверяем утилизацию после перемещения
                        test_alloc = current_allocations[other_host_id] + [vm_id]
                        new_util = self.calculate_host_utilization(other_host_id, test_alloc)
                        
                        if new_util <= UPPER_THRESHOLD:
                            # Перемещаем VM
                            current_allocations[host_id].remove(vm_id)
                            current_allocations[other_host_id].append(vm_id)
                            return True
        
        # Если утилизация слишком низкая, пытаемся добавить VM
        elif vm_utils[0][1] < LOWER_THRESHOLD:
            # Ищем VM для добавления
            for other_host_id in sorted(self.hosts.keys()):
                if other_host_id == host_id:
                    continue
                    
                other_vms = current_allocations[other_host_id]
                if not other_vms:
                                    continue
                                    
                # Сортируем VM по утилизации
                other_vm_utils = []
                for vm_id in other_vms:
                    if vm_id in self.vms:
                        vm = self.vms[vm_id]
                        util = (vm.get("cpu", 0) / self.hosts[other_host_id].get("cpu", 1) + 
                               vm.get("ram", 0) / self.hosts[other_host_id].get("ram", 1)) / 2
                        other_vm_utils.append((vm_id, util))
                
                other_vm_utils.sort(key=lambda x: x[1])
                
                # Пытаемся переместить VM с наименьшей утилизацией
                for vm_id, _ in other_vm_utils:
                    if self.can_host_vm(host_id, vm_id):
                        # Проверяем утилизацию после перемещения
                        test_alloc = current_allocations[host_id] + [vm_id]
                        new_util = self.calculate_host_utilization(host_id, test_alloc)
                        
                        if LOWER_THRESHOLD <= new_util <= UPPER_THRESHOLD:
                            # Перемещаем VM
                            current_allocations[other_host_id].remove(vm_id)
                            current_allocations[host_id].append(vm_id)
                            return True
        
        return False

    def _try_aggressive_recovery(self, vm_id: str, host_id: str, current_allocations: Dict[str, List[str]]) -> bool:
        """Пытается агрессивно освободить место для VM."""
        if host_id not in self.hosts:
            return False
            
        # Получаем список VM на хосте
        current_vms = current_allocations[host_id]
        if not current_vms:
            return False
            
        # Вычисляем текущую утилизацию хоста
        current_utilization = self.calculate_host_utilization(host_id, current_allocations[host_id])
        
        # Если утилизация уже высокая, пробуем другие хосты
        if current_utilization > UPPER_THRESHOLD:
            return False
        
        # Сортируем VM по утилизации
        vm_utils = []
        for vm_id in current_vms:
            if vm_id in self.vms:
                vm = self.vms[vm_id]
                util = (vm.get("cpu", 0) / self.hosts[host_id].get("cpu", 1) + 
                       vm.get("ram", 0) / self.hosts[host_id].get("ram", 1)) / 2
                vm_utils.append((vm_id, util))
        
        vm_utils.sort(key=lambda x: x[1])
        
        # Пытаемся найти подходящую VM для перемещения
        for vm_id, _ in vm_utils:
            if self.can_host_vm(host_id, vm_id):
                # Проверяем утилизацию после перемещения
                test_alloc = current_allocations[host_id] + [vm_id]
                new_util = self.calculate_host_utilization(host_id, test_alloc)
                
                if LOWER_THRESHOLD <= new_util <= UPPER_THRESHOLD:
                    # Проверяем, что исходный хост не станет слишком пустым
                    remaining_vms = [v for v in current_vms if v != vm_id]
                    source_util = self.calculate_host_utilization(host_id, remaining_vms)
                    
                    if source_util >= LOWER_THRESHOLD or not remaining_vms:
                            # Перемещаем VM
                        current_allocations[host_id].remove(vm_id)
                        current_allocations[host_id].append(vm_id)
                        return True
        
        return False

    def _optimize_initial_placement(self) -> None:
        """Оптимизирует начальное размещение VM."""
        # Сортируем VM по убыванию потребляемых ресурсов
        sorted_vms = sorted(
            [(vm_id, vm) for vm_id, vm in self.vms.items()],
            key=lambda x: (x[1].get("cpu", 0) + x[1].get("ram", 0), x[0]),
            reverse=True
        )
        
        # Сортируем хосты по возрастанию доступных ресурсов
        sorted_hosts = sorted(
            [(host_id, host) for host_id, host in self.hosts.items()],
            key=lambda x: (x[1].get("cpu", 0) + x[1].get("ram", 0), x[0])
        )
        
        # Создаем новые размещения
        new_allocations = {host_id: [] for host_id in self.hosts}
        unplaced_vms = []
        
        # Пытаемся разместить каждую VM
        for vm_id, vm in sorted_vms:
            # Если VM уже размещена и размещение хорошее, оставляем как есть
            current_host = self.vm_to_host_map.get(vm_id)
            if current_host and current_host in self.hosts:
                if self.can_host_vm(current_host, vm_id):
                    new_allocations[current_host].append(vm_id)
                    continue
            
            # Ищем лучший хост для размещения
            best_host = None
            best_utilization = float('inf')
            
            for host_id, _ in sorted_hosts:
                if self.can_host_vm(host_id, vm_id):
                    # Проверяем утилизацию после размещения
                    test_allocation = new_allocations[host_id] + [vm_id]
                    utilization = self.calculate_host_utilization(host_id, test_allocation)
                    
                    # Выбираем хост с наиболее оптимальной утилизацией
                    if abs(utilization - TARGET_UTILIZATION) < abs(best_utilization - TARGET_UTILIZATION):
                        best_host = host_id
                        best_utilization = utilization
            
            if best_host:
                new_allocations[best_host].append(vm_id)
            else:
                unplaced_vms.append(vm_id)
        
        # Обрабатываем неразмещенные VM
        if unplaced_vms:
            # Пытаемся освободить место путем перераспределения маленьких VM
            for vm_id in unplaced_vms:
                placed = False
                for host_id in sorted(self.hosts.keys()):
                    # Пытаемся освободить место, перемещая маленькие VM
                    if self._try_make_space_for_vm(vm_id, host_id, new_allocations):
                        new_allocations[host_id].append(vm_id)
                            placed = True
                            break
                    
                    if not placed:
                    # Если все еще не удалось разместить, пробуем более агрессивную стратегию
                    for host_id in sorted(self.hosts.keys()):
                        if self._try_aggressive_placement(vm_id, host_id, new_allocations):
                            new_allocations[host_id].append(vm_id)
                            placed = True
                            break
        
        # Обновляем размещения
        self.previous_allocations = new_allocations

    def _try_make_space_for_vm(self, vm_id: str, host_id: str, current_allocations: Dict[str, List[str]]) -> bool:
        """Пытается освободить место для VM путем перемещения маленьких VM."""
        if vm_id not in self.vms or host_id not in self.hosts:
            return False
            
        target_vm = self.vms[vm_id]
        host = self.hosts[host_id]
        
        # Вычисляем необходимые ресурсы
        needed_resources = {
            "cpu": target_vm.get("cpu", 0),
            "ram": target_vm.get("ram", 0)
        }
        
        # Получаем текущие VM на хосте
        current_vms = current_allocations[host_id]
        
        # Сортируем VM по размеру (от маленьких к большим)
        vm_sizes = []
        for current_vm_id in current_vms:
            if current_vm_id in self.vms:
                vm = self.vms[current_vm_id]
                size = vm.get("cpu", 0) + vm.get("ram", 0)
                vm_sizes.append((current_vm_id, size))
        
        vm_sizes.sort(key=lambda x: x[1])
        
        # Пытаемся найти комбинацию VM для перемещения
        for i in range(1, len(vm_sizes) + 1):
            for combination in itertools.combinations(vm_sizes, i):
                freed_resources = {
                    "cpu": sum(self.vms[vm_id].get("cpu", 0) for vm_id, _ in combination),
                    "ram": sum(self.vms[vm_id].get("ram", 0) for vm_id, _ in combination)
                }
                
                if (freed_resources["cpu"] >= needed_resources["cpu"] and 
                    freed_resources["ram"] >= needed_resources["ram"]):
                    
                    # Пытаемся найти новые хосты для перемещаемых VM
                    success = True
                    moves = {}
                    
                    for move_vm_id, _ in combination:
                        # Ищем подходящий хост
                        for other_host_id in self.hosts:
                            if other_host_id != host_id:
                                test_alloc = current_allocations[other_host_id][:]
                                if self.can_host_vm(other_host_id, move_vm_id):
                                    moves[move_vm_id] = other_host_id
                                    break
                        
                        if move_vm_id not in moves:
                        success = False
                        break
                
                if success:
                        # Применяем перемещения
                        for move_vm_id, target_host in moves.items():
                            current_allocations[host_id].remove(move_vm_id)
                            current_allocations[target_host].append(move_vm_id)
                        return True
        
        return False

    def _try_aggressive_placement(self, vm_id: str, host_id: str, current_allocations: Dict[str, List[str]]) -> bool:
        """Пытается агрессивно освободить место для VM."""
        if vm_id not in self.vms or host_id not in self.hosts:
            return False
            
        target_vm = self.vms[vm_id]
        host = self.hosts[host_id]
        
        # Вычисляем текущую утилизацию хоста
        current_utilization = self.calculate_host_utilization(host_id, current_allocations[host_id])
        
        # Если утилизация уже высокая, пробуем другие хосты
        if current_utilization > UPPER_THRESHOLD:
            return False
        
        # Получаем список VM на хосте
        current_vms = current_allocations[host_id]
        
        # Вычисляем общие требуемые ресурсы
        total_needed = {
            "cpu": sum(self.vms[vm].get("cpu", 0) for vm in current_vms) + target_vm.get("cpu", 0),
            "ram": sum(self.vms[vm].get("ram", 0) for vm in current_vms) + target_vm.get("ram", 0)
        }
        
        # Проверяем, возможно ли размещение в принципе
        if (total_needed["cpu"] <= host.get("cpu", 0) and 
            total_needed["ram"] <= host.get("ram", 0)):
            
            # Сортируем VM по утилизации
            vm_utils = []
            for current_vm_id in current_vms:
                if current_vm_id in self.vms:
                    vm = self.vms[current_vm_id]
                    util = (vm.get("cpu", 0) / host.get("cpu", 1) + 
                           vm.get("ram", 0) / host.get("ram", 1)) / 2
                    vm_utils.append((current_vm_id, util))
            
            vm_utils.sort(key=lambda x: x[1])
            
            # Пытаемся переместить VM с наименьшей утилизацией
            for move_vm_id, _ in vm_utils:
                # Ищем хост для перемещения
                for other_host_id in self.hosts:
                    if other_host_id != host_id:
                        if self.can_host_vm(other_host_id, move_vm_id):
                            # Перемещаем VM
                            current_allocations[host_id].remove(move_vm_id)
                            current_allocations[other_host_id].append(move_vm_id)
                            
                            # Проверяем, можем ли теперь разместить целевую VM
                            if self.can_host_vm(host_id, vm_id):
                                return True
                            
                            # Если нет, возвращаем VM обратно
                            current_allocations[other_host_id].remove(move_vm_id)
                            current_allocations[host_id].append(move_vm_id)
        
        return False

    def allocate(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Размещает виртуальные машины на хостах."""
        try:
            # Оптимизируем начальное размещение
            self._optimize_initial_placement()
            
            # Создаем копию текущих размещений
            current_allocations = {host_id: list(vms) for host_id, vms in self.previous_allocations.items()}
            
            # Обрабатываем неразмещенные VM
            unplaced_vms = set(self.vms.keys()) - set(
                vm_id for vms in current_allocations.values() for vm_id in vms
            )
            
            # Пытаемся разместить каждую неразмещенную VM
            for vm_id in sorted(unplaced_vms):
                placed = False
                
                # Пытаемся найти подходящий хост
                for host_id in sorted(self.hosts.keys()):
                    if self.can_host_vm(host_id, vm_id):
                        current_allocations[host_id].append(vm_id)
                        placed = True
                        break
                
                # Если не удалось разместить, пробуем стратегии восстановления
                if not placed:
                    if not self._handle_allocation_failure(vm_id, current_allocations):
                        # Если все стратегии не сработали, пропускаем эту VM
                        continue
            
            # Оптимизируем миграции
            iterations = 0
            while iterations < 3:  # Ограничиваем количество итераций
                made_changes = self._optimize_migrations(current_allocations)
                if not made_changes:
                            break
                iterations += 1
            
            # Обновляем размещения
            self.previous_allocations = current_allocations
            
            # Форматируем результат
            result = {"allocations": {}}
            for host_id, vms in current_allocations.items():
                if vms:  # Добавляем только хосты с размещенными VM
                    result["allocations"][host_id] = {"virtual_machines": {}}
                    for vm_id in vms:
                        result["allocations"][host_id]["virtual_machines"][vm_id] = {}
            
            return result
            
        except Exception as e:
            # В случае ошибки возвращаем пустое размещение
            return {"allocations": {}}

    def _check_vm_placement_possibility(self, vm_id: str) -> bool:
        """Проверяет, может ли VM быть размещена хотя бы на одном хосте."""
        if vm_id not in self.vms:
            return False
        
        vm = self.vms[vm_id]
        
        # Проверяем каждый хост
        for host_id, host in self.hosts.items():
            # Базовая проверка ресурсов
            can_host = True
            for resource in ["cpu", "ram", "disk"]:
                if resource in vm:
                    if resource not in host or host[resource] < vm[resource]:
                        can_host = False
                        break
            
            if can_host:
                return True
        
        return False

    def _handle_quantum_fluctuations(self, host_id: str, current_allocations: Dict[str, List[str]]) -> bool:
        """Обрабатывает квантовые флуктуации на хосте."""
        if not host_id.startswith("quantum"):
            return False
            
        # Получаем текущую утилизацию
        current_utilization = self.calculate_host_utilization(host_id, current_allocations[host_id])
        
        # Если утилизация в нормальных пределах, ничего не делаем
        if LOWER_THRESHOLD <= current_utilization <= UPPER_THRESHOLD:
            return False
            
        # Получаем список VM на хосте
        current_vms = current_allocations[host_id]
        if not current_vms:
            return False
            
        # Сортируем VM по утилизации
        vm_utils = []
        for vm_id in current_vms:
            if vm_id in self.vms:
                vm = self.vms[vm_id]
                util = (vm.get("cpu", 0) / self.hosts[host_id].get("cpu", 1) + 
                       vm.get("ram", 0) / self.hosts[host_id].get("ram", 1)) / 2
                vm_utils.append((vm_id, util))
        
        vm_utils.sort(key=lambda x: x[1], reverse=True)
        
        # Если утилизация слишком высокая, пытаемся переместить VM
        if current_utilization > UPPER_THRESHOLD:
            # Начинаем с VM с наибольшей утилизацией
            for vm_id, _ in vm_utils:
                # Ищем стабильный хост
                for other_host_id in sorted(self.hosts.keys()):
                    if (not other_host_id.startswith("quantum") and 
                        self.can_host_vm(other_host_id, vm_id)):
                        # Проверяем утилизацию после перемещения
                        test_alloc = current_allocations[other_host_id] + [vm_id]
                        new_util = self.calculate_host_utilization(other_host_id, test_alloc)
                        
                        if new_util <= UPPER_THRESHOLD:
                            # Перемещаем VM
                            current_allocations[host_id].remove(vm_id)
                            current_allocations[other_host_id].append(vm_id)
                            return True
        
        # Если утилизация слишком низкая, пытаемся добавить VM
        elif current_utilization < LOWER_THRESHOLD:
            # Ищем VM для добавления
            for other_host_id in sorted(self.hosts.keys()):
                if other_host_id == host_id:
                    continue
                    
                other_vms = current_allocations[other_host_id]
                if not other_vms:
                    continue
                    
                # Сортируем VM по утилизации
                other_vm_utils = []
                for vm_id in other_vms:
                    if vm_id in self.vms:
                        vm = self.vms[vm_id]
                        util = (vm.get("cpu", 0) / self.hosts[other_host_id].get("cpu", 1) + 
                               vm.get("ram", 0) / self.hosts[other_host_id].get("ram", 1)) / 2
                        other_vm_utils.append((vm_id, util))
                
                other_vm_utils.sort(key=lambda x: x[1])
                
                # Пытаемся переместить VM с наименьшей утилизацией
                for vm_id, _ in other_vm_utils:
                    if self.can_host_vm(host_id, vm_id):
                        # Проверяем утилизацию после перемещения
                        test_alloc = current_allocations[host_id] + [vm_id]
                        new_util = self.calculate_host_utilization(host_id, test_alloc)
                        
                        if LOWER_THRESHOLD <= new_util <= UPPER_THRESHOLD:
                            # Перемещаем VM
                            current_allocations[other_host_id].remove(vm_id)
                            current_allocations[host_id].append(vm_id)
                            return True
        
        return False

    def _optimize_migrations(self, current_allocations: Dict[str, List[str]]) -> bool:
        """Оптимизирует миграции для улучшения общей утилизации."""
        made_changes = False
        
        # Сортируем хосты по утилизации
        host_utils = []
        for host_id in self.hosts:
            utilization = self.calculate_host_utilization(host_id, current_allocations[host_id])
            host_utils.append((host_id, utilization))
        
        # Сортируем по утилизации (от высокой к низкой)
        host_utils.sort(key=lambda x: x[1], reverse=True)
        
        # Обрабатываем перегруженные хосты
        for host_id, utilization in host_utils:
            if utilization > UPPER_THRESHOLD:
                # Пытаемся разгрузить хост
                if self._balance_host_load(host_id, current_allocations):
                    made_changes = True
            elif utilization < LOWER_THRESHOLD:
                # Пытаемся увеличить утилизацию
                if self._increase_host_utilization(host_id, current_allocations):
                    made_changes = True
                    
            # Для квантовых хостов применяем специальную обработку
            if host_id.startswith("quantum"):
                if self._handle_quantum_fluctuations(host_id, current_allocations):
                    made_changes = True
        
        return made_changes

    def _balance_host_load(self, host_id: str, current_allocations: Dict[str, List[str]]) -> bool:
        """Балансирует нагрузку перегруженного хоста."""
        if host_id not in self.hosts:
            return False
            
        current_vms = current_allocations[host_id]
        if not current_vms:
            return False
            
        # Сортируем VM по утилизации
        vm_utils = []
        for vm_id in current_vms:
            if vm_id in self.vms:
                vm = self.vms[vm_id]
                util = (vm.get("cpu", 0) / self.hosts[host_id].get("cpu", 1) + 
                       vm.get("ram", 0) / self.hosts[host_id].get("ram", 1)) / 2
                vm_utils.append((vm_id, util))
        
        vm_utils.sort(key=lambda x: x[1], reverse=True)
        
        # Пытаемся переместить VM, начиная с тех, что имеют наибольшую утилизацию
        for vm_id, _ in vm_utils:
            # Ищем подходящий хост с низкой утилизацией
            for other_host_id in sorted(self.hosts.keys()):
                if other_host_id == host_id:
                    continue
                    
                # Проверяем утилизацию целевого хоста
                target_util = self.calculate_host_utilization(other_host_id, current_allocations[other_host_id])
                if target_util >= UPPER_THRESHOLD:
                    continue
                    
                if self.can_host_vm(other_host_id, vm_id):
                    # Проверяем утилизацию после перемещения
                    test_alloc = current_allocations[other_host_id] + [vm_id]
                    new_util = self.calculate_host_utilization(other_host_id, test_alloc)
                    
                    if new_util <= UPPER_THRESHOLD:
                        # Перемещаем VM
                        current_allocations[host_id].remove(vm_id)
                        current_allocations[other_host_id].append(vm_id)
                        return True
        
        return False

    def _increase_host_utilization(self, host_id: str, current_allocations: Dict[str, List[str]]) -> bool:
        """Увеличивает утилизацию хоста с низкой нагрузкой."""
        if host_id not in self.hosts:
            return False
            
        # Ищем VM на других хостах, которые можно переместить
        for other_host_id in sorted(self.hosts.keys()):
            if other_host_id == host_id:
                continue
                
            other_vms = current_allocations[other_host_id]
            if not other_vms:
                continue
                
            # Сортируем VM по утилизации
            vm_utils = []
            for vm_id in other_vms:
                if vm_id in self.vms:
                    vm = self.vms[vm_id]
                    util = (vm.get("cpu", 0) / self.hosts[other_host_id].get("cpu", 1) + 
                           vm.get("ram", 0) / self.hosts[other_host_id].get("ram", 1)) / 2
                    vm_utils.append((vm_id, util))
            
            vm_utils.sort(key=lambda x: x[1])
            
            # Пытаемся найти подходящую VM для перемещения
            for vm_id, _ in vm_utils:
                if self.can_host_vm(host_id, vm_id):
                    # Проверяем утилизацию после перемещения
                    test_alloc = current_allocations[host_id] + [vm_id]
                    new_util = self.calculate_host_utilization(host_id, test_alloc)
                    
                    if LOWER_THRESHOLD <= new_util <= UPPER_THRESHOLD:
                        # Проверяем, что исходный хост не станет слишком пустым
                        remaining_vms = [v for v in other_vms if v != vm_id]
                        source_util = self.calculate_host_utilization(other_host_id, remaining_vms)
                        
                        if source_util >= LOWER_THRESHOLD or not remaining_vms:
                            # Перемещаем VM
                            current_allocations[other_host_id].remove(vm_id)
                            current_allocations[host_id].append(vm_id)
                return True
        
        return False

if __name__ == "__main__":
    try:
        # Читаем входные данные из stdin
        input_data = sys.stdin.read().strip()
        
        if not input_data:
            # Если входные данные пустые, возвращаем пустой результат
            sys.stdout.write(json.dumps({
                "allocations": {},
                "migrations": {},
                "allocation_failures": []
            }))
            sys.stdout.flush()
            sys.exit(0)
        
        # Создаем экземпляр планировщика
        scheduler = VMScheduler()
        
        # Решаем задачу
        result = scheduler.solve(input_data)
        
        # Выводим результат в stdout
        sys.stdout.write(result)
        sys.stdout.flush()
        sys.exit(0)
    except Exception as e:
        # В случае ошибки выводим пустой результат
        sys.stdout.write(json.dumps({
            "allocations": {},
            "migrations": {},
            "allocation_failures": []
        }))
        sys.stdout.flush()
        sys.exit(0) 