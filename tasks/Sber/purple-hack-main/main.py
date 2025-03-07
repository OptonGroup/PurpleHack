from backend.algorithms import SimulatedAnnealing
from backend.algorithms.common import data_lists
from backend.algorithms import cost_len_fn, cost_resource_fn, cost_fn, combined_cost_fn
from backend.algorithms.common import get_output
import json

if __name__ == "__main__":
    min_ans = None
    min_assigned_time_list = []

    for _ in range(1):
        simulated_annealing = SimulatedAnnealing(combined_cost_fn(cost_len_fn, cost_resource_fn, cost_fn), 1.0, 0.99, 1000)
        ans, assigned_time_list = simulated_annealing.predict_resource(data_lists)
        if min_ans is None or ans < min_ans:
            min_ans = ans
            min_assigned_time_list = assigned_time_list

    print(min_ans)
    res = get_output(min_assigned_time_list)
    with open("result.json", "w") as result_file:
        try:
            json.dump(res, result_file)
        except Exception as err:
            print("result ", err, res)
