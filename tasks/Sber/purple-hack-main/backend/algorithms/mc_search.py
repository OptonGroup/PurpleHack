import json
from typing import Callable, Dict, Any, List
import random

from .data_matching_functions import create_project_role_ids, match_task_id_to_resourse_id
from .graph_dependencies.topological_sort import top_sort
from .tasks import assign_time
from .common import data_lists, get_output


class MonteCarloSearch:
    def __init__(self,
                 cost_fn: Callable[[List[Dict[str, Any]]], int],
                 num_iterations: int) -> None:
        self.cost_fn = cost_fn
        self.num_iterations = num_iterations

        self.task_sequence = top_sort.task_sequence
        self.blockers = top_sort.blokers
    
    def assign_time(self, tasks2resources) -> List[Dict[str, Any]]:
        return assign_time(self.task_sequence, tasks2resources, self.blockers)
    
    def fit(self, *args) -> None:
        pass

    def predict(self, data_lists) -> List[Dict[str, Any]]:
        tasks_by_project_roles = create_project_role_ids(data_lists["tasks"])
        tasks_to_resourses = match_task_id_to_resourse_id(data_lists, tasks_by_project_roles)
        assigned_time = self.assign_time(tasks_to_resourses)

        best_combination = assigned_time
        best_cost = self.cost_fn(assigned_time)

        for _ in range(self.num_iterations):
            
            curr_tasks_to_resourses = match_task_id_to_resourse_id(data_lists, tasks_by_project_roles)
            curr_assigned_time = self.assign_time(curr_tasks_to_resourses)

            current_cost = self.cost_fn(curr_assigned_time)

            if current_cost < best_cost:
                best_combination = curr_assigned_time
                best_cost = current_cost
        
        return best_combination
    
    def save_result(self,
                    data: Dict[str, Any],
                    output_path: str) -> None:
        if not output_path.endswith(".json"):
            output_path = output_path + ".json"
        
        with open(output_path, "w") as f:
            json.dump(get_output(self.predict(data)), f)
