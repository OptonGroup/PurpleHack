"""Synthetic data generator for project scheduling."""

import random
from datetime import datetime, timedelta
from typing import List, Tuple

import networkx as nx
import numpy as np
from omegaconf import DictConfig

from ..models.types import ProjectCalendar, ProjectData, Resource, Task


class SyntheticDataGenerator:
    """Generator for synthetic project scheduling data."""
    
    def __init__(self, config: DictConfig):
        """Initialize the data generator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.roles = [
            "developer",
            "designer",
            "tester",
            "analyst",
            "manager"
        ]
        
    def generate_tasks(
        self,
        num_tasks: int,
        start_date: datetime
    ) -> List[Task]:
        """Generate a list of tasks.
        
        Args:
            num_tasks: Number of tasks to generate
            start_date: Project start date
            
        Returns:
            List of generated tasks
        """
        tasks = []
        for i in range(num_tasks):
            duration = random.randint(1, 10)
            role = random.choice(self.roles)
            
            # Some tasks have time constraints
            has_constraint = random.random() < 0.3
            constraint_type = "startnoearlierthan" if has_constraint else None
            constraint_date = (
                start_date + timedelta(days=random.randint(0, 30))
                if has_constraint else None
            )
            
            task = Task(
                id=i,
                duration=duration,
                role=role,
                constraint_type=constraint_type,
                constraint_date=constraint_date,
                dependencies=[]
            )
            tasks.append(task)
            
        return tasks
    
    def generate_dependencies(
        self,
        tasks: List[Task],
        density: float
    ) -> List[Task]:
        """Generate task dependencies.
        
        Args:
            tasks: List of tasks
            density: Dependency graph density
            
        Returns:
            Tasks with generated dependencies
        """
        # Create a DAG
        G = nx.DiGraph()
        G.add_nodes_from(range(len(tasks)))
        
        # Add random edges while maintaining DAG property
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                if random.random() < density:
                    G.add_edge(i, j)
                    
        # Ensure the graph is a DAG by removing cycles
        while not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            for cycle in cycles:
                G.remove_edge(cycle[0], cycle[1])
                
        # Update task dependencies
        for i, task in enumerate(tasks):
            task.dependencies = list(G.predecessors(i))
            
        return tasks
    
    def generate_resources(
        self,
        num_resources: int,
        start_date: datetime,
        duration_days: int
    ) -> List[Resource]:
        """Generate project resources.
        
        Args:
            num_resources: Number of resources to generate
            start_date: Project start date
            duration_days: Project duration in days
            
        Returns:
            List of generated resources
        """
        resources = []
        for i in range(num_resources):
            # Each resource has 1-2 roles
            num_roles = random.randint(1, 2)
            roles = random.sample(self.roles, num_roles)
            
            # Generate non-working days
            calendar = []
            for day in range(duration_days):
                date = start_date + timedelta(days=day)
                # 20% chance of being unavailable
                if random.random() < 0.2:
                    calendar.append(date)
                    
            resource = Resource(
                id=i,
                role=roles[0],  # Primary role
                calendar=calendar
            )
            resources.append(resource)
            
        return resources
    
    def generate_project_calendar(
        self,
        start_date: datetime,
        duration_days: int
    ) -> ProjectCalendar:
        """Generate project calendar.
        
        Args:
            start_date: Project start date
            duration_days: Project duration in days
            
        Returns:
            Generated project calendar
        """
        working_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        
        # Generate holidays (5% of days are holidays)
        holidays = []
        for day in range(duration_days):
            date = start_date + timedelta(days=day)
            if random.random() < 0.05:
                holidays.append(date)
                
        return ProjectCalendar(
            working_days=working_days,
            holidays=holidays
        )
    
    def generate_optimal_schedule(
        self,
        project_data: ProjectData
    ) -> Tuple[List[int], List[int], List[datetime], List[datetime]]:
        """Generate optimal schedule for training.
        
        Args:
            project_data: Project data
            
        Returns:
            Tuple containing:
            - Task sequence
            - Resource assignments
            - Start times
            - End times
        """
        # Create a priority queue based on dependencies and constraints
        G = nx.DiGraph()
        for task in project_data.tasks:
            G.add_node(task.id)
            for dep in task.dependencies:
                G.add_edge(dep, task.id)
                
        # Topological sort for initial sequence
        task_sequence = list(nx.topological_sort(G))
        
        # Assign resources based on roles and availability
        resource_assignments = [-1] * len(project_data.tasks)
        start_times = [None] * len(project_data.tasks)
        end_times = [None] * len(project_data.tasks)
        
        current_time = datetime.min
        resource_availability = [datetime.min] * len(project_data.resources)
        
        for task_id in task_sequence:
            task = project_data.tasks[task_id]
            
            # Find earliest possible start time
            earliest_start = current_time
            for dep in task.dependencies:
                if end_times[dep] > earliest_start:
                    earliest_start = end_times[dep]
                    
            if task.constraint_type == "startnoearlierthan":
                if task.constraint_date > earliest_start:
                    earliest_start = task.constraint_date
                    
            # Find available resource
            best_resource = None
            best_start_time = datetime.max
            
            for res_id, resource in enumerate(project_data.resources):
                if resource.role == task.role:
                    available_time = resource_availability[res_id]
                    if available_time < best_start_time:
                        best_resource = res_id
                        best_start_time = available_time
                        
            # Assign task to resource
            resource_assignments[task_id] = best_resource
            start_time = max(earliest_start, best_start_time)
            end_time = start_time + timedelta(days=task.duration)
            
            start_times[task_id] = start_time
            end_times[task_id] = end_time
            resource_availability[best_resource] = end_time
            
        return task_sequence, resource_assignments, start_times, end_times
    
    def generate_dataset(
        self,
        num_samples: int,
        min_tasks: int,
        max_tasks: int,
        min_resources: int,
        max_resources: int,
        dependency_density: float
    ) -> List[Tuple[ProjectData, Tuple]]:
        """Generate a dataset of project scheduling problems.
        
        Args:
            num_samples: Number of samples to generate
            min_tasks: Minimum number of tasks per project
            max_tasks: Maximum number of tasks per project
            min_resources: Minimum number of resources per project
            max_resources: Maximum number of resources per project
            dependency_density: Density of task dependencies
            
        Returns:
            List of (project_data, optimal_schedule) pairs
        """
        dataset = []
        for _ in range(num_samples):
            # Generate random project size
            num_tasks = random.randint(min_tasks, max_tasks)
            num_resources = random.randint(min_resources, max_resources)
            
            # Generate project components
            start_date = datetime.now()
            duration_days = num_tasks * 2  # Rough estimate
            
            tasks = self.generate_tasks(num_tasks, start_date)
            tasks = self.generate_dependencies(tasks, dependency_density)
            resources = self.generate_resources(
                num_resources,
                start_date,
                duration_days
            )
            calendar = self.generate_project_calendar(
                start_date,
                duration_days
            )
            
            project_data = ProjectData(
                tasks=tasks,
                resources=resources,
                project_calendar=calendar
            )
            
            # Generate optimal schedule
            optimal_schedule = self.generate_optimal_schedule(project_data)
            
            dataset.append((project_data, optimal_schedule))
            
        return dataset 