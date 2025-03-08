#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess
import time
import math
import glob
import psutil
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
import argparse
import sys
from dataclasses import dataclass

@dataclass
class Host:
    cpu: int
    ram: int

@dataclass
class VirtualMachine:
    cpu: int
    ram: int

@dataclass
class Allocation:
    host_name: str
    vm_names: List[str]

@dataclass
class Migration:
    vm_name: str
    from_host: str
    to_host: str

@dataclass
class TestResult:
    test_name: str
    round_number: int
    score: float
    execution_time: float
    memory_usage: float
    utilization_scores: Dict[str, float]
    allocation_failure_penalty: float
    migration_penalty: float
    zero_utilization_bonus: float
    details: Dict[str, Any]

class VMSchedulerTester:
    def __init__(self, solution_path: str, tests_dir: str, verbose: bool = False):
        self.solution_path = solution_path
        self.tests_dir = tests_dir
        self.verbose = verbose
        self.test_results: List[TestResult] = []
        
        # Состояние для отслеживания хостов с нулевой утилизацией
        self.zero_utilization_rounds: Dict[str, Dict[str, int]] = {}  # test_name -> {host_name -> count}
        self.hosts_with_previous_vms: Dict[str, Set[str]] = {}  # test_name -> {host_names}
        
    def calculate_utilization_score(self, utilization: float) -> float:
        """
        Вычисляет баллы за утилизацию по формуле:
        f(x) = -0.67466 + (42.385/(-2.5x+5.96)) * exp(-2*ln(-2.5x+2.96))
        """
        if utilization < 0 or utilization > 1:
            raise ValueError(f"Utilization must be between 0 and 1, got {utilization}")
        
        if utilization == 0:
            return 0
        
        x = utilization
        try:
            inner_term = -2.5 * x + 2.96
            if inner_term <= 0:
                return 0  # Защита от отрицательного логарифма
            
            result = -0.67466 + (42.385 / (-2.5 * x + 5.96)) * math.exp(-2 * math.log(-2.5 * x + 2.96))
            return max(0, result)  # Защита от отрицательного результата
        except Exception as e:
            print(f"Error calculating utilization score for {utilization}: {e}")
            return 0
    
    def run_test(self, test_dir: str) -> List[TestResult]:
        """Запускает тест для указанного каталога с тестами"""
        test_name = os.path.basename(test_dir)
        print(f"\n=== Running test: {test_name} ===")
        
        # Инициализация для отслеживания хостов с нулевой утилизацией
        self.zero_utilization_rounds[test_name] = {}
        self.hosts_with_previous_vms[test_name] = set()
        
        results = []
        round_files = sorted(glob.glob(os.path.join(test_dir, "round_*.json")))
        
        # Текущее состояние размещения ВМ
        current_allocations: Dict[str, List[str]] = {}  # host_name -> [vm_names]
        
        for round_idx, round_file in enumerate(round_files, 1):
            print(f"  Round {round_idx}: {os.path.basename(round_file)}")
            
            # Чтение входных данных
            with open(round_file, 'r') as f:
                input_data = f.read().strip()
            
            # Запуск решения и измерение времени и памяти
            result = self.run_solution(input_data)
            
            # Анализ результатов
            test_result = self.analyze_result(
                test_name, 
                round_idx, 
                input_data, 
                result['output'], 
                result['execution_time'], 
                result['memory_usage'],
                current_allocations
            )
            
            # Обновление текущего состояния размещения
            input_json = json.loads(input_data)
            output_json = json.loads(result['output'])
            
            # Обновляем текущие размещения на основе выходных данных
            current_allocations = output_json.get('allocations', {})
            
            # Сохраняем результат
            results.append(test_result)
            
            # Вывод подробной информации, если включен режим verbose
            if self.verbose:
                print(f"    Score: {test_result.score:.4f}")
                print(f"    Execution time: {test_result.execution_time:.4f} seconds")
                print(f"    Memory usage: {test_result.memory_usage:.2f} MB")
                print(f"    Utilization scores: {test_result.utilization_scores}")
                if test_result.allocation_failure_penalty != 0:
                    print(f"    Allocation failure penalty: {test_result.allocation_failure_penalty:.4f}")
                if test_result.migration_penalty != 0:
                    print(f"    Migration penalty: {test_result.migration_penalty:.4f}")
                if test_result.zero_utilization_bonus != 0:
                    print(f"    Zero utilization bonus: {test_result.zero_utilization_bonus:.4f}")
        
        return results
    
    def run_solution(self, input_data: str) -> Dict[str, Any]:
        """Запускает решение с заданными входными данными и возвращает результат"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        start_time = time.time()
        
        # Запуск решения
        try:
            proc = subprocess.Popen(
                self.solution_path,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = proc.communicate(input=input_data, timeout=5)
            
            if proc.returncode != 0:
                print(f"Error running solution: {stderr}")
                return {
                    'output': '{"allocations": {}, "allocation_failures": [], "migrations": {}}',
                    'execution_time': 0,
                    'memory_usage': 0
                }
            
            execution_time = time.time() - start_time
            
            # Измерение использования памяти
            current_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_usage = current_memory - initial_memory
            
            return {
                'output': stdout.strip(),
                'execution_time': execution_time,
                'memory_usage': memory_usage
            }
        except subprocess.TimeoutExpired:
            print("Solution execution timed out (5 seconds)")
            return {
                'output': '{"allocations": {}, "allocation_failures": [], "migrations": {}}',
                'execution_time': 5,
                'memory_usage': 0
            }
        except Exception as e:
            print(f"Error running solution: {e}")
            return {
                'output': '{"allocations": {}, "allocation_failures": [], "migrations": {}}',
                'execution_time': 0,
                'memory_usage': 0
            }
    
    def analyze_result(
        self, 
        test_name: str, 
        round_idx: int, 
        input_data: str, 
        output_data: str, 
        execution_time: float, 
        memory_usage: float,
        previous_allocations: Dict[str, List[str]]
    ) -> TestResult:
        """Анализирует результат выполнения решения и вычисляет оценку"""
        try:
            input_json = json.loads(input_data)
            output_json = json.loads(output_data)
            
            hosts = {name: Host(data['cpu'], data['ram']) 
                    for name, data in input_json.get('hosts', {}).items()}
            
            vms = {name: VirtualMachine(data['cpu'], data['ram']) 
                  for name, data in input_json.get('virtual_machines', {}).items()}
            
            allocations = output_json.get('allocations', {})
            allocation_failures = output_json.get('allocation_failures', [])
            migrations = output_json.get('migrations', {})
            
            # Вычисление утилизации для каждого хоста
            host_utilizations: Dict[str, Dict[str, float]] = {}  # host_name -> {'cpu': float, 'ram': float}
            
            for host_name, host_data in hosts.items():
                host_utilizations[host_name] = {'cpu': 0.0, 'ram': 0.0}
                
                # Если хост есть в размещениях, считаем его утилизацию
                if host_name in allocations:
                    vm_names = allocations[host_name]
                    
                    # Обновляем множество хостов, на которых были ВМ
                    if vm_names:
                        self.hosts_with_previous_vms[test_name].add(host_name)
                    
                    # Суммируем ресурсы всех ВМ на хосте
                    total_cpu = sum(vms[vm_name].cpu for vm_name in vm_names if vm_name in vms)
                    total_ram = sum(vms[vm_name].ram for vm_name in vm_names if vm_name in vms)
                    
                    # Вычисляем утилизацию как отношение используемых ресурсов к доступным
                    host_utilizations[host_name]['cpu'] = total_cpu / host_data.cpu if host_data.cpu > 0 else 0
                    host_utilizations[host_name]['ram'] = total_ram / host_data.ram if host_data.ram > 0 else 0
            
            # Вычисление общей утилизации для каждого хоста (среднее между CPU и RAM)
            host_total_utilizations = {
                host_name: (util['cpu'] + util['ram']) / 2 
                for host_name, util in host_utilizations.items()
            }
            
            # Вычисление баллов за утилизацию
            utilization_scores = {
                host_name: self.calculate_utilization_score(utilization)
                for host_name, utilization in host_total_utilizations.items()
            }
            
            # Обработка хостов с нулевой утилизацией
            zero_utilization_bonus = 0.0
            for host_name, utilization in host_total_utilizations.items():
                if utilization == 0:
                    # Увеличиваем счетчик раундов с нулевой утилизацией
                    if host_name not in self.zero_utilization_rounds[test_name]:
                        self.zero_utilization_rounds[test_name][host_name] = 0
                    self.zero_utilization_rounds[test_name][host_name] += 1
                    
                    # Если хост имел нулевую утилизацию 5+ раундов подряд И на нем ранее были ВМ
                    if (self.zero_utilization_rounds[test_name][host_name] >= 5 and 
                        host_name in self.hosts_with_previous_vms[test_name]):
                        zero_utilization_bonus += 8.0
                else:
                    # Сбрасываем счетчик, если утилизация не нулевая
                    self.zero_utilization_rounds[test_name][host_name] = 0
            
            # Вычисление штрафа за невозможность размещения ВМ
            allocation_failure_penalty = 0.0
            if allocation_failures:
                allocation_failure_penalty = -5.0 * len(hosts)
            
            # Вычисление штрафа за миграцию ВМ
            migration_penalty = 0.0
            if migrations:
                migration_penalty = -1.0 * (len(migrations) ** 2)
            
            # Вычисление общей оценки
            total_score = (
                sum(utilization_scores.values()) + 
                allocation_failure_penalty + 
                migration_penalty + 
                zero_utilization_bonus
            )
            
            # Проверка корректности размещения
            is_valid = self.validate_placement(hosts, vms, allocations, previous_allocations, migrations)
            if not is_valid:
                total_score = -100  # Штраф за некорректное размещение
            
            return TestResult(
                test_name=test_name,
                round_number=round_idx,
                score=total_score,
                execution_time=execution_time,
                memory_usage=memory_usage,
                utilization_scores=utilization_scores,
                allocation_failure_penalty=allocation_failure_penalty,
                migration_penalty=migration_penalty,
                zero_utilization_bonus=zero_utilization_bonus,
                details={
                    'host_utilizations': host_utilizations,
                    'host_total_utilizations': host_total_utilizations,
                    'is_valid_placement': is_valid
                }
            )
        except Exception as e:
            print(f"Error analyzing result: {e}")
            return TestResult(
                test_name=test_name,
                round_number=round_idx,
                score=-100,
                execution_time=execution_time,
                memory_usage=memory_usage,
                utilization_scores={},
                allocation_failure_penalty=0,
                migration_penalty=0,
                zero_utilization_bonus=0,
                details={'error': str(e)}
            )
    
    def validate_placement(
        self, 
        hosts: Dict[str, Host], 
        vms: Dict[str, VirtualMachine], 
        allocations: Dict[str, List[str]],
        previous_allocations: Dict[str, List[str]],
        migrations: Dict[str, Dict[str, str]]
    ) -> bool:
        """Проверяет корректность размещения ВМ на хостах"""
        # Проверка 1: Все ВМ размещены на существующих хостах
        for host_name, vm_names in allocations.items():
            if host_name not in hosts:
                print(f"Error: Host {host_name} does not exist")
                return False
            
            for vm_name in vm_names:
                if vm_name not in vms:
                    print(f"Error: VM {vm_name} does not exist")
                    return False
        
        # Проверка 2: Утилизация хоста не превышает его характеристики
        for host_name, vm_names in allocations.items():
            host = hosts[host_name]
            
            total_cpu = sum(vms[vm_name].cpu for vm_name in vm_names if vm_name in vms)
            total_ram = sum(vms[vm_name].ram for vm_name in vm_names if vm_name in vms)
            
            if total_cpu > host.cpu:
                print(f"Error: Host {host_name} CPU overutilized: {total_cpu} > {host.cpu}")
                return False
            
            if total_ram > host.ram:
                print(f"Error: Host {host_name} RAM overutilized: {total_ram} > {host.ram}")
                return False
        
        # Проверка 3: Каждая ВМ размещена только на одном хосте
        all_placed_vms = []
        for vm_names in allocations.values():
            all_placed_vms.extend(vm_names)
        
        if len(all_placed_vms) != len(set(all_placed_vms)):
            print("Error: Some VMs are placed on multiple hosts")
            return False
        
        # Проверка 4: Проверка миграций
        for vm_name, migration_data in migrations.items():
            from_host = migration_data.get('from')
            to_host = migration_data.get('to')
            
            if from_host not in hosts:
                print(f"Error: Migration source host {from_host} does not exist")
                return False
            
            if to_host not in hosts:
                print(f"Error: Migration target host {to_host} does not exist")
                return False
            
            # Проверяем, что ВМ была на исходном хосте в предыдущем размещении
            if from_host in previous_allocations and vm_name not in previous_allocations[from_host]:
                print(f"Error: VM {vm_name} was not on host {from_host} before migration")
                return False
            
            # Проверяем, что ВМ находится на целевом хосте в текущем размещении
            if to_host in allocations and vm_name not in allocations[to_host]:
                print(f"Error: VM {vm_name} is not on host {to_host} after migration")
                return False
        
        return True
    
    def run_all_tests(self) -> None:
        """Запускает все тесты в указанном каталоге"""
        test_dirs = [d for d in glob.glob(os.path.join(self.tests_dir, "*")) if os.path.isdir(d)]
        
        for test_dir in test_dirs:
            results = self.run_test(test_dir)
            self.test_results.extend(results)
    
    def print_summary(self) -> None:
        """Выводит сводную информацию о результатах тестирования"""
        if not self.test_results:
            print("\nNo test results available.")
            return
        
        print("\n=== Test Summary ===")
        
        # Группировка результатов по тестам
        test_groups = {}
        for result in self.test_results:
            if result.test_name not in test_groups:
                test_groups[result.test_name] = []
            test_groups[result.test_name].append(result)
        
        # Вывод результатов по каждому тесту
        total_score = 0.0
        total_execution_time = 0.0
        total_memory_usage = 0.0
        
        for test_name, results in test_groups.items():
            test_score = sum(r.score for r in results)
            avg_execution_time = sum(r.execution_time for r in results) / len(results)
            avg_memory_usage = sum(r.memory_usage for r in results) / len(results)
            
            print(f"\nTest: {test_name}")
            print(f"  Total score: {test_score:.4f}")
            print(f"  Average execution time: {avg_execution_time:.4f} seconds")
            print(f"  Average memory usage: {avg_memory_usage:.2f} MB")
            
            # Детализация по раундам
            for result in results:
                print(f"  Round {result.round_number}: Score = {result.score:.4f}, "
                      f"Time = {result.execution_time:.4f}s, Memory = {result.memory_usage:.2f}MB")
            
            total_score += test_score
            total_execution_time += sum(r.execution_time for r in results)
            total_memory_usage = max(total_memory_usage, avg_memory_usage)
        
        # Общий итог
        print("\n=== Overall Results ===")
        print(f"Total score across all tests: {total_score:.4f}")
        print(f"Total execution time: {total_execution_time:.4f} seconds")
        print(f"Peak memory usage: {total_memory_usage:.2f} MB")
        
        # Проверка соответствия ограничениям
        time_limit_exceeded = any(r.execution_time > 2.0 for r in self.test_results)
        if time_limit_exceeded:
            print("\nWARNING: Time limit (2 seconds per round) exceeded in some tests!")
        
        print("\nPerformance rating:")
        if total_score > 0:
            if time_limit_exceeded:
                print("⚠️ Solution works but exceeds time limits")
            else:
                if total_score > 500:
                    print("🌟 Excellent solution with high optimization")
                elif total_score > 300:
                    print("✅ Good solution with decent optimization")
                elif total_score > 100:
                    print("👍 Basic solution that works correctly")
                else:
                    print("🔄 Solution works but needs optimization")
        else:
            print("❌ Solution has critical issues")

def main():
    parser = argparse.ArgumentParser(description='Test VM Scheduler Solution')
    parser.add_argument('--solution', type=str, default='./run',
                        help='Path to the solution executable')
    parser.add_argument('--tests', type=str, default='./tests',
                        help='Path to the tests directory')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    tester = VMSchedulerTester(args.solution, args.tests, args.verbose)
    tester.run_all_tests()
    tester.print_summary()

if __name__ == "__main__":
    main() 