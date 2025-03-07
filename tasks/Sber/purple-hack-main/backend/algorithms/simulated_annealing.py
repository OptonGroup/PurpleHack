from copy import deepcopy
from typing import Callable
import math
import random

from .data_matching_functions import create_project_role_ids, match_task_id_to_resourse_id
from .cost_functions import cost_fn, cost_len_fn, cost_resource_fn, combined_cost_fn
from .graph_dependencies.topological_sort import top_sort
from .tasks import assign_time
from .common import data_lists


class SimulatedAnnealing:
    def __init__(self,
                 cost_fn: Callable,
                 temperature: float,
                 temperature_decay: float = 0.99,
                 num_iterations: int = 1):
        self.temp = temperature
        self.temp_decay = temperature_decay
        self.num_iterations = num_iterations
        self.cost_fn = cost_fn

    def fit(self, *args) -> None:
        pass

    def predict_sequence(self, tasks_to_resources):
        task_sequence = top_sort.create_sequence()
        blockers = top_sort.blokers
        assigned_time_list = assign_time(deepcopy(task_sequence), deepcopy(tasks_to_resources), deepcopy(blockers))
        ans = self.cost_fn(deepcopy(assigned_time_list))

        for i in range(self.num_iterations):
            self.temp *= self.temp_decay
            first_id = random.randint(0, len(task_sequence) - 1)
            second_id = random.randint(0, len(task_sequence) - 1)

            if top_sort.can_swap(task_sequence[first_id], task_sequence[second_id], first_id, second_id, task_sequence):
                annealed_task_sequence = task_sequence.copy()
                annealed_task_sequence[first_id], annealed_task_sequence[second_id] = annealed_task_sequence[second_id], \
                    annealed_task_sequence[first_id]
                annealed_assigned_time_list = assign_time(deepcopy(annealed_task_sequence),
                                                          deepcopy(tasks_to_resources), deepcopy(blockers))
                annealed_ans = self.cost_fn(deepcopy(annealed_assigned_time_list))

                if annealed_ans < ans or random.uniform(0, 1) < math.exp((ans - annealed_ans) / self.temp):
                    task_sequence = annealed_task_sequence
                    assigned_time_list = annealed_assigned_time_list
                    ans = annealed_ans

        return ans, assigned_time_list

    def predict_resource(self, data: dict, role_ids_local_count: dict, optimize_type="time"):
        resources_by_project_roles = create_project_role_ids(data.get("resources"))
        tasks_to_resources = match_task_id_to_resourse_id(data, resources_by_project_roles, role_ids_local_count, optimize_type)
        tasks_by_project_roles = create_project_role_ids(data.get("tasks"))
        project_roles = list(tasks_by_project_roles.keys())

        sequence_simulated_annealing = SimulatedAnnealing(self.cost_fn, 1.0, 0.99, 100)
        ans, assigned_time_list = sequence_simulated_annealing.predict_sequence(tasks_to_resources)

        for i in range(self.num_iterations):
            self.temp *= self.temp_decay
            annealed_tasks_to_resources = deepcopy(tasks_to_resources)
            project_role_id = random.choice(project_roles)
            first_id = tasks_by_project_roles[project_role_id][
                random.randint(0, len(tasks_by_project_roles[project_role_id]) - 1)]  # task id
            second_id = tasks_by_project_roles[project_role_id][
                random.randint(0, len(tasks_by_project_roles[project_role_id]) - 1)]
            annealed_tasks_to_resources[first_id], annealed_tasks_to_resources[second_id] = \
                annealed_tasks_to_resources[second_id], annealed_tasks_to_resources[first_id]

            annealed_ans, annealed_assigned_time_list = sequence_simulated_annealing.predict_sequence(
                annealed_tasks_to_resources)
            if annealed_ans < ans or random.uniform(0, 1) < math.exp((ans - annealed_ans - 1) / self.temp):
                tasks_to_resources = annealed_tasks_to_resources
                ans, assigned_time_list = annealed_ans, annealed_assigned_time_list
        return ans, assigned_time_list
