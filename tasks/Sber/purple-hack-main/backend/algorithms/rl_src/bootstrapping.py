import random
from uuid import uuid4
# from algorithms.cost_functions import cost_len_fn, cost_resource_fn, cost_fn


class DatasetBoostrapping:
    def __init__(self, data_lists) -> None:
        self.original = data_lists


        self.resourses_id = list(set(r["id"] for r in self.original["resources"]))
        self.resourses_price = list(set(r["price"] for r in self.original["resources"]))
        self.resourses_role = list(set(r["project_role_id"] for r in self.original["resources"]))

        self.tasks_id = list(set(r["id"] for r in self.original["tasks"]))
        self.tasks_effort = list(set(r["effort"] for r in self.original["tasks"]))
        self.tasks_role = list(set(r["project_role_id"] for r in self.original["tasks"]))

    def perturb(self, seed, num_resources=6, num_tasks=9, num_dependencies=6):
        tasks_id = []

        output = {'start_date': '2024-02-05',
                  'resources': [],
                  'tasks': [],
                  'dependencies': []
                  }
        
        # random.seed(seed)

        for _ in range(num_resources):
            output["resources"].append({
                'id': str(id(random.choice(range(1000)))),
                'price': random.choice(self.resourses_price),
                'project_role_id': random.choice(self.resourses_role),
                'hours': 8
            })
        
        for _ in range(num_tasks):
            task_id = str(id(random.choice(range(1000))))
            tasks_id.append(task_id)

            output["tasks"].append({
                'id': task_id,
                'effort': random.choice(self.tasks_effort),
                'project_role_id': random.choice(self.tasks_role)
            })
        
        for _ in range(num_dependencies):
            from_task = random.choice(tasks_id)
            # idx = tasks_id.index(from_task)
            to_task = random.choice([id_ for id_ in tasks_id if id_ != from_task])

            output["dependencies"].append({
                'id': str(id(random.choice(range(1000)))),
                'from': from_task,
                'to': to_task
            })
        
        return output
