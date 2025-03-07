import json

from algorithms.common import init, get_output, data_lists
from algorithms.cost_functions import combined_cost_fn, cost_len_fn, cost_resource_fn, cost_fn
from algorithms.data_matching_functions import count_role_ids
from algorithms.simulated_annealing import SimulatedAnnealing
from algorithms.graph_dependencies import topological_sort
from algorithms import cost_functions


def run_simulated_annealing(duration, price, resource, analyst_cnt, dev_cnt, tester_cnt) -> [int, dict]:
    num_iterations = 300
    role_ids_local_count = {
        "analyst": analyst_cnt,
        "developer": dev_cnt,
        "tester": tester_cnt,
    }

    if duration == 1 and price == 0 and resource == 0:
        simulated_annealing = SimulatedAnnealing(cost_len_fn, 1.0, 0.99, num_iterations)
        ans, assigned_time_list = simulated_annealing.predict_resource(data_lists, role_ids_local_count, "time")
    elif duration == 0 and price == 1 and resource == 0:
        simulated_annealing = SimulatedAnnealing(cost_len_fn, 1.0, 0.99, num_iterations)
        ans, assigned_time_list = simulated_annealing.predict_resource(data_lists, role_ids_local_count, "price")
    elif duration == 0 and price == 0 and resource == 1:
        print("INN")
        simulated_annealing = SimulatedAnnealing(cost_len_fn, 1.0, 0.99, num_iterations)
        ans, assigned_time_list = simulated_annealing.predict_resource(data_lists, role_ids_local_count, "resource")
    else:
        simulated_annealing = SimulatedAnnealing(
            combined_cost_fn(cost_len_fn, cost_fn, cost_resource_fn, duration, price, resource), 1.0, 0.99,
            num_iterations)
        ans, assigned_time_list = simulated_annealing.predict_resource(data_lists, role_ids_local_count, "time")

    return ans, assigned_time_list


def process_json(data: dict, duration: float, price: float, resource: float) -> dict:
    init(data)
    topological_sort.init()
    cost_functions.init()

    print(duration, price, resource)

    role_ids_count = count_role_ids(data_lists.get("resources"))
    print(role_ids_count)
    min_sum = 1000
    ans = 1000
    assigned_time_list = []

    if duration == 1 or (price == 0 and resource == 0):
        ans, assigned_time_list = run_simulated_annealing(duration, price, resource, role_ids_count["analyst"],
                                                          role_ids_count["developer"], role_ids_count["tester"])
        return get_output(assigned_time_list)

    for analysts in range(1, role_ids_count["analyst"]):
        for devs in range(1, role_ids_count["developer"]):
            for testers in range(1, role_ids_count["tester"]):
                current_sum = analysts + devs + testers
                current_ans, current_assigned_time_list = run_simulated_annealing(duration, price, resource, analysts, devs, testers)
                if resource == 1 and current_sum < min_sum:
                    min_sum = current_sum
                    ans, assigned_time_list = current_ans, current_assigned_time_list
                else:
                    ans, assigned_time_list = current_ans, current_assigned_time_list

    return get_output(assigned_time_list)
