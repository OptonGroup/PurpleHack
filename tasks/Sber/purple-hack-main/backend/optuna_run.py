import json
from algorithms.common import init, get_output, data_lists
from algorithms.cost_functions import combined_cost_fn, cost_len_fn, cost_resource_fn, cost_fn
from algorithms.simulated_annealing import SimulatedAnnealing
from algorithms.graph_dependencies import topological_sort
from algorithms import cost_functions
import random


import optuna
# pip install optuna


def objective_len(trial):
    temperature = trial.suggest_float("temperature", 0.1, 5)
    temperature_decay = trial.suggest_float("temperature_decay", 0.5, 0.99)
    num_iterations = trial.suggest_int("num_iterations", 5, 1000)

    simulated_annealing = SimulatedAnnealing(cost_len_fn, temperature, temperature_decay, num_iterations)
    ans, assigned_time_list = simulated_annealing.predict_resource(data_lists)

    cost = cost_len_fn(assigned_time_list)
    return cost


def objective_price(trial):
    temperature = trial.suggest_float("temperature", -5, 5)
    temperature_decay = trial.suggest_float("temperature_decay", 0.5, 0.99)
    num_iterations = trial.suggest_int("num_iterations", 5, 1000)

    simulated_annealing = SimulatedAnnealing(cost_len_fn, temperature, temperature_decay, num_iterations)
    ans, assigned_time_list = simulated_annealing.predict_resource(data_lists, "price")

    cost = cost_len_fn(assigned_time_list)
    return cost


def objective_resource(trial):
    temperature = trial.suggest_float("temperature", 0.1, 5)
    temperature_decay = trial.suggest_float("temperature_decay", 0.90, 0.99)
    num_iterations = trial.suggest_int("num_iterations", 5, 1000)

    simulated_annealing = SimulatedAnnealing(cost_len_fn, temperature, temperature_decay, num_iterations)
    ans, assigned_time_list = simulated_annealing.predict_resource(data_lists, "resource")

    cost = cost_len_fn(assigned_time_list)
    return cost


def process_json2(data: dict, duration: float, price: float, resource: float) -> dict:
    # print(data)
    random.seed(42)
    init(data)
    topological_sort.init()
    cost_functions.init()

    study = optuna.create_study(direction="minimize")
    # study.optimize()

    print(duration, price, resource)
    num_iterations = 300
    
    if duration == 1 and price == 0 and resource == 0:
        study.optimize(objective_len, n_trials=100)
        params = study.best_params

        simulated_annealing = SimulatedAnnealing(cost_len_fn, **params)
        ans, assigned_time_list = simulated_annealing.predict_resource(data_lists)
    
    elif duration == 0 and price == 1 and resource == 0:
        study.optimize(objective_price, n_trials=100)
        params = study.best_params

        simulated_annealing = SimulatedAnnealing(cost_len_fn, **params)
        ans, assigned_time_list = simulated_annealing.predict_resource(data_lists, "price")
    
    elif duration == 0 and price == 0 and resource == 1:
        print("INN")
        study.optimize(objective_resource, n_trials=100)
        params = study.best_params

        simulated_annealing = SimulatedAnnealing(cost_len_fn, **params)
        ans, assigned_time_list = simulated_annealing.predict_resource(data_lists, "resource")

    else:
        cost_fn_ = combined_cost_fn(cost_len_fn, cost_fn, cost_resource_fn, duration, price, resource)

        def _obj(trial):
            temperature = trial.suggest_float("temperature", -5, 5)
            temperature_decay = trial.suggest_float("temperature_decay", 0.5, 0.99)
            num_iterations = trial.suggest_int("num_iterations", 5, 1000)

            simulated_annealing = SimulatedAnnealing(cost_fn_, temperature, temperature_decay, num_iterations)
            ans, assigned_time_list = simulated_annealing.predict_resource(data_lists)

            cost = cost_fn_(assigned_time_list)
            return cost
        
        study.optimize(_obj, n_trials=100)
        params = study.best_params

        simulated_annealing = SimulatedAnnealing(cost_fn_, **params)
        ans, assigned_time_list = simulated_annealing.predict_resource(data_lists)

    return get_output(assigned_time_list)


if __name__ == "__main__":
    with open("тестовое задание.json") as f:
        d = json.load(f)

    res = process_json2(d, 1, 0, 0)
    
    with open("result.json", "w") as output_file:
        json.dump(res, output_file)