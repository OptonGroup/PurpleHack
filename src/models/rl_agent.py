"""
Агент обучения с подкреплением для оптимизации календарного плана.

Этот модуль содержит реализацию агента для обучения с подкреплением,
который принимает решения о порядке выполнения задач и назначении ресурсов.
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import deque
import random
from datetime import datetime
import logging

# Пытаемся импортировать torch, если не получается - используем заглушку
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    logging.info("PyTorch успешно импортирован")
except ImportError:
    logging.warning("PyTorch не найден, используется заглушка")
    import src.models.mock_torch as torch
    from src.models.mock_torch import nn
    from src.models.mock_torch import functional as F
    from src.models.mock_torch import optim
    from src.models.mock_torch import FloatTensor, LongTensor, save, load, no_grad

from src.models.rl_environment import ProjectSchedulingEnvironment


class FeatureExtractor:
    """
    Класс для извлечения признаков из наблюдения среды.
    
    Преобразует сложную структуру наблюдения в вектор признаков
    для использования в нейронной сети.
    """
    
    def __init__(self, env: ProjectSchedulingEnvironment):
        """
        Инициализирует экстрактор признаков.
        
        Args:
            env: Среда для оптимизации календарного плана
        """
        self.env = env
        self.task_ids = list(env.tasks.keys())
        self.resource_ids = list(env.resources.keys())
        
        # Словари для преобразования идентификаторов в индексы
        self.task_to_idx = {task_id: idx for idx, task_id in enumerate(self.task_ids)}
        self.resource_to_idx = {res_id: idx for idx, res_id in enumerate(self.resource_ids)}
        
        # Размерность вектора признаков
        self.n_tasks = len(self.task_ids)
        self.n_resources = len(self.resource_ids)
        
        # Всего признаков: задачи + ресурсы + их комбинации
        self.feature_dim = (
            # Признаки для задач (статус выполнения, приоритет, длительность)
            self.n_tasks * 3 +
            # Признаки для ресурсов (доступность)
            self.n_resources * 1 +
            # Признаки для пар (задача, ресурс) (соответствие ролей)
            self.n_tasks * self.n_resources
        )
    
    def extract_features(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Извлекает признаки из наблюдения.
        
        Args:
            observation: Наблюдение состояния среды
            
        Returns:
            Вектор признаков
        """
        features = np.zeros(self.feature_dim)
        
        # Текущее время
        current_time = observation["current_time"]
        
        # Извлечение признаков для задач
        completed_tasks = set(observation["completed_tasks"])
        for task_id in self.task_ids:
            task_idx = self.task_to_idx[task_id]
            
            # Статус выполнения (1, если задача выполнена)
            features[task_idx] = 1 if task_id in completed_tasks else 0
            
            # Приоритет задачи
            task = self.env.tasks[task_id]
            features[self.n_tasks + task_idx] = task.priority / 10.0  # Нормализация
            
            # Длительность задачи
            features[2 * self.n_tasks + task_idx] = task.duration / 20.0  # Нормализация
        
        # Извлечение признаков для ресурсов
        for resource_id in self.resource_ids:
            resource_idx = self.resource_to_idx[resource_id]
            
            # Доступность ресурса (время до освобождения в днях)
            # Ищем информацию о ресурсе в списке resources
            resource_info = None
            for res_info in observation.get("resources", []):
                if res_info.get("id") == resource_id:
                    resource_info = res_info
                    break
            
            if resource_info and "available_at" in resource_info:
                available_at = resource_info["available_at"]
                days_until_available = (available_at - current_time).total_seconds() / (24 * 60 * 60)
                features[3 * self.n_tasks + resource_idx] = min(days_until_available, 10) / 10.0  # Нормализация
        
        # Извлечение признаков для пар (задача, ресурс)
        for task_id in self.task_ids:
            task = self.env.tasks[task_id]
            task_idx = self.task_to_idx[task_id]
            
            for resource_id in self.resource_ids:
                resource = self.env.resources[resource_id]
                resource_idx = self.resource_to_idx[resource_id]
                
                # Соответствие ролей (1, если роли соответствуют)
                role_match = (task.projectRole is None) or (task.projectRole == resource.projectRole)
                features[3 * self.n_tasks + self.n_resources + task_idx * self.n_resources + resource_idx] = 1 if role_match else 0
        
        return features


class DQNNetwork(nn.Module):
    """
    Нейронная сеть для аппроксимации Q-функции.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Инициализирует нейронную сеть.
        
        Args:
            input_dim: Размерность входного вектора
            hidden_dim: Размерность скрытого слоя
            output_dim: Размерность выходного вектора
        """
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть.
        
        Args:
            x: Входной тензор
            
        Returns:
            Выходной тензор
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RLAgent:
    """
    Агент обучения с подкреплением для оптимизации календарного плана.
    
    Использует алгоритм Deep Q-Learning для принятия решений
    о порядке выполнения задач и назначении ресурсов.
    """
    
    def __init__(
        self, 
        env: ProjectSchedulingEnvironment,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        memory_capacity: int = 10000,
        target_update_frequency: int = 10
    ):
        """
        Инициализирует агента обучения с подкреплением.
        
        Args:
            env: Среда для оптимизации календарного плана
            hidden_dim: Размерность скрытого слоя нейронной сети
            gamma: Коэффициент дисконтирования
            epsilon_start: Начальное значение эпсилон для epsilon-greedy политики
            epsilon_end: Конечное значение эпсилон
            epsilon_decay: Коэффициент убывания эпсилон
            learning_rate: Скорость обучения
            batch_size: Размер батча для обучения
            memory_capacity: Емкость памяти для опыта
            target_update_frequency: Частота обновления целевой сети
        """
        self.env = env
        self.feature_extractor = FeatureExtractor(env)
        
        # Размерность входа и выхода нейронной сети
        self.input_dim = self.feature_extractor.feature_dim
        self.output_dim = len(env.tasks) * len(env.resources)
        
        # Создание основной и целевой сетей
        self.policy_net = DQNNetwork(self.input_dim, hidden_dim, self.output_dim)
        self.target_net = DQNNetwork(self.input_dim, hidden_dim, self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Целевая сеть всегда в режиме вывода
        
        # Оптимизатор
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Память для хранения опыта
        self.memory = deque(maxlen=memory_capacity)
        
        # Гиперпараметры
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        
        # Счетчик шагов
        self.steps_done = 0
        
        # Словари для преобразования индексов в действия и обратно
        self.task_ids = list(env.tasks.keys())
        self.resource_ids = list(env.resources.keys())
        self.action_to_idx = {}
        self.idx_to_action = {}
        
        # Создаем отображение между действиями и индексами
        idx = 0
        for task_id in self.task_ids:
            for resource_id in self.resource_ids:
                action = (task_id, resource_id)
                self.action_to_idx[action] = idx
                self.idx_to_action[idx] = action
                idx += 1
        
        # Логгер
        self.logger = logging.getLogger(__name__)
    
    def select_action(self, observation: Dict[str, Any]) -> Tuple[str, str]:
        """
        Выбирает действие на основе текущего наблюдения.
        
        Args:
            observation: Наблюдение состояния среды
            
        Returns:
            Кортеж (task_id, resource_id)
        """
        # Извлекаем признаки из наблюдения
        features = self.feature_extractor.extract_features(observation)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)  # Добавляем размерность батча
        
        # Получаем доступные действия
        available_actions = self.env.get_available_actions()
        
        # Если нет доступных действий, возвращаем None
        if not available_actions:
            return None
        
        # Индексы доступных действий
        available_indices = [self.action_to_idx[action] for action in available_actions if action in self.action_to_idx]
        
        # Epsilon-greedy политика
        if random.random() < self.epsilon:
            # Выбор случайного действия из доступных
            action_idx = random.choice(available_indices)
            return self.idx_to_action[action_idx]
        else:
            # Выбор наилучшего действия из доступных
            self.policy_net.eval()
            with torch.no_grad():
                q_values = self.policy_net(features_tensor).squeeze()
                
                # Маскируем недоступные действия
                mask = torch.ones_like(q_values) * float('-inf')
                mask[available_indices] = 0
                masked_q_values = q_values + mask
                
                # Выбираем действие с максимальным Q-значением
                action_idx = torch.argmax(masked_q_values).item()
                
            self.policy_net.train()
            return self.idx_to_action[action_idx]
    
    def store_transition(
        self, 
        observation: Dict[str, Any], 
        action: Tuple[str, str], 
        reward: float, 
        next_observation: Dict[str, Any], 
        done: bool
    ):
        """
        Сохраняет переход в память.
        
        Args:
            observation: Текущее наблюдение
            action: Выбранное действие
            reward: Полученная награда
            next_observation: Следующее наблюдение
            done: Флаг завершения эпизода
        """
        # Извлекаем признаки
        features = self.feature_extractor.extract_features(observation)
        next_features = self.feature_extractor.extract_features(next_observation)
        
        # Получаем индекс действия
        action_idx = self.action_to_idx[action]
        
        # Сохраняем переход в память
        self.memory.append((features, action_idx, reward, next_features, done))
    
    def update_model(self):
        """
        Обновляет модель на основе опыта из памяти.
        """
        # Если недостаточно опыта, пропускаем обновление
        if len(self.memory) < self.batch_size:
            return
        
        # Выбираем случайный батч из памяти
        batch = random.sample(self.memory, self.batch_size)
        features, action_idxs, rewards, next_features, dones = zip(*batch)
        
        # Преобразуем в тензоры
        features = torch.FloatTensor(features)
        action_idxs = torch.LongTensor(action_idxs)
        rewards = torch.FloatTensor(rewards)
        next_features = torch.FloatTensor(next_features)
        dones = torch.FloatTensor(dones)
        
        # Получаем текущие Q-значения для выбранных действий
        current_q_values = self.policy_net(features).gather(1, action_idxs.unsqueeze(1)).squeeze(1)
        
        # Получаем максимальные Q-значения для следующих состояний от целевой сети
        next_q_values = self.target_net(next_features).max(1)[0]
        
        # Маскируем Q-значения для терминальных состояний
        next_q_values = next_q_values * (1 - dones)
        
        # Вычисляем целевые Q-значения
        target_q_values = rewards + self.gamma * next_q_values
        
        # Вычисляем функцию потерь
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Градиентный спуск
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Обновляем целевую сеть
        self.steps_done += 1
        if self.steps_done % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Обновляем эпсилон
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def train(
        self, 
        num_episodes: int, 
        max_steps_per_episode: int = 1000,
        verbose: bool = True
    ) -> List[float]:
        """
        Обучает агента на заданном количестве эпизодов.
        
        Args:
            num_episodes: Количество эпизодов для обучения
            max_steps_per_episode: Максимальное количество шагов в эпизоде
            verbose: Выводить ли информацию о процессе обучения
            
        Returns:
            Список наград за каждый эпизод
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            # Сбрасываем среду
            observation = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps_per_episode):
                # Выбор действия
                action = self.select_action(observation)
                
                # Если нет доступных действий, завершаем эпизод
                if action is None:
                    break
                
                # Выполнение действия
                next_observation, reward, done, _ = self.env.step(action)
                
                # Сохранение перехода в память
                self.store_transition(observation, action, reward, next_observation, done)
                
                # Обновление модели
                loss = self.update_model()
                
                # Обновление наблюдения и награды
                observation = next_observation
                episode_reward += reward
                
                if verbose and step % 10 == 0:
                    self.logger.info(f"Episode {episode+1}/{num_episodes}, Step {step+1}, Loss: {loss}, Epsilon: {self.epsilon:.4f}")
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            if verbose:
                self.logger.info(f"Episode {episode+1}/{num_episodes} completed with reward {episode_reward:.4f}")
        
        return episode_rewards
    
    def optimize_schedule(self) -> Dict[str, Tuple[datetime, datetime, str]]:
        """
        Оптимизирует календарный план, используя обученную модель.
        
        Returns:
            Оптимизированный план в виде словаря {task_id: (start_date, end_date, resource_id)}
        """
        # Сбрасываем среду
        observation = self.env.reset()
        
        while True:
            # Выбор действия
            action = self.select_action(observation)
            
            # Если нет доступных действий, завершаем планирование
            if action is None:
                break
            
            # Выполнение действия
            observation, _, done, _ = self.env.step(action)
            
            if done:
                break
        
        # Возвращаем оптимизированный план
        return self.env.get_schedule()
    
    def save_model(self, path: str):
        """
        Сохраняет модель в файл.
        
        Args:
            path: Путь для сохранения модели
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
    
    def load_model(self, path: str):
        """
        Загружает модель из файла.
        
        Args:
            path: Путь к сохраненной модели
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done'] 