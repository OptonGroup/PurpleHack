import random
from collections import defaultdict
from copy import copy
from pprint import pprint
import queue

from .graph import create_graph, get_all_roots, get_all_leaves, dp_roots_and_leaves, extend_graph
from ..common import data_lists
from ..common import data as common_data


class TopSort:
    def __init__(self, data: dict):
        if not data:
            return
        self.graph, self.blokers = create_graph(data.get("dependencies"), data.get("is_task"))
        self.extended_graph = {}
        self.roots = get_all_roots(data.get("dependencies"), data.get("is_task"))
        self.leaves = get_all_leaves(data.get("dependencies"), data.get("is_task"))
        self.task_sequence = []
        self.tin = {}
        self.tout = {}
        self.time = 0
        self.roots_gr = {}
        self.leaves_gr = {}
        self.init_extended_graph(data)

        for k, v in self.roots_gr.items():
            self.roots_gr[k] = list(set(v))

        for k, v in self.leaves_gr.items():
            self.leaves_gr[k] = list(set(v))

        for k, v in self.graph.items():
            self.graph[k] = list(set(v))
        for k, v in self.blokers.items():
            self.blokers[k] = list(set(v))

        self.vis = defaultdict(bool)
        self.roots = [task_id for task_id in common_data["tasks"].keys() if len(self.blokers[task_id]) == 0]

        self.mat = defaultdict(lambda: defaultdict(bool))
        self.top_sort()
        pprint(self.graph)
        print("TASK TREE")
        pprint(data.get("task_tree"))
        self.create_mat()

    def create_mat(self):
        for task_id in common_data["tasks"].keys():
            self.mat_dfs(task_id, "*", task_id)

    def mat_dfs(self, v, par, w):
        if v == par:
            return
        if v != w:
            self.mat[w][v] = True
        for t in self.graph[v]:
            self.mat_dfs(t, v, w)

    def init_extended_graph(self, data: dict):
        self.roots_gr, self.leaves_gr = dp_roots_and_leaves(data.get("task_tree"), data.get("is_task"), self.roots,
                                                            self.leaves)

        self.graph = extend_graph(self.graph, data.get("dependencies"), self.roots_gr, self.leaves_gr, self.blokers)

    def dfs(self, vertex: str, par: str) -> None:
        self.tin[vertex] = self.time
        self.time += 1
        self.task_sequence.append(vertex)

        for to in self.graph.get(vertex, []):
            if to != par:
                self.dfs(to, vertex)

        self.tout[vertex] = self.time
        self.time += 1

    def top_sort(self) -> None:
        for root in self.roots:
            self.dfs(root, "-1")

    def gen_sequence(self) -> list:
        return self.task_sequence

    def is_parent(self, upper_vertex: str, lower_vertex: str) -> bool:
        # if upper_vertex[-2:] == '12' and lower_vertex[-2:] == '10':
        #     x = 0
        # return self.tin[upper_vertex] < self.tin[lower_vertex] and self.tout[upper_vertex] > self.tout[lower_vertex]

        return self.mat[upper_vertex][lower_vertex]

    def can_swap(self, upper_vertex: str, lower_vertex: str, upper_idx: int, lower_idx: int,
                 task_sequence: list) -> bool:
        if upper_vertex[-2:] == '10' and lower_vertex[-2:] == '12':
            x = 0
        res = True
        task_sequence = copy(task_sequence)
        task_sequence[lower_idx], task_sequence[upper_idx] = task_sequence[upper_idx], task_sequence[lower_idx]

        is_c = True

        d = {0: 8, 1: 12, 2: 16}

        for t in task_sequence:
            n = int(t[-2:])
            if n in [8, 9, 10, 12, 13, 14]:
                if d[0] == n:
                    d[0] += 1
                    if d[0] == 11:
                        d[0] = 12
                else:
                    is_c = False
                    break
            if n in [16, 17, 18]:
                if d[2] == n:
                    d[2] += 1
                else:
                    is_c = False
                    break

        # lower_idx, upper_idx = upper_idx, lower_idx
        lower_vertex, upper_vertex = upper_vertex, lower_vertex
        for i in range(upper_idx):
            tt = self.is_parent(upper_vertex, task_sequence[i])
            res &= not self.is_parent(upper_vertex, task_sequence[i])
        for i in range(upper_idx, len(task_sequence)):
            res &= not self.is_parent(task_sequence[i], upper_vertex)

        for i in range(lower_idx):
            res &= not self.is_parent(lower_vertex, task_sequence[i])
        for i in range(lower_idx, len(task_sequence)):
            res &= not self.is_parent(task_sequence[i], lower_vertex)
        x = 0

        return res

    def dfs_shuffled(self, vertex, par, sequence):
        if self.vis[vertex]:
            return
        self.vis[vertex] = True
        sequence.append(vertex)
        for to in self.graph.get(vertex, []):
            if to != par:
                self.dfs_shuffled(to, vertex, sequence)

    def bfs_shuffled(self):
        list_q = []
        self.vis = defaultdict(bool)
        for root in self.roots:
            list_q.append(root)
            self.vis[root] = True

        sequence = []
        while list_q:
            vertex = list_q[0]
            list_q.pop(0)
            if all(self.vis[t] for t in self.blokers[vertex]):
                self.vis[vertex] = True
                sequence.append(vertex)
            else:
                list_q.append(vertex)
                continue
            for to in self.graph.get(vertex, []):
                if not self.vis[to]:
                    list_q.append(to)

        return sequence

    def create_sequence(self) -> list:
        shuffled_roots = list(self.roots)
        random.shuffle(shuffled_roots)
        sequence = self.bfs_shuffled()

        return sequence


top_sort = TopSort({})


def init():
    top_sort.__init__(data_lists)
