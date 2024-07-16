"""
Compare the performance of a high_multiplicity_fair_allocation

Programmer: Naor Ladani & Elor Israeli
Since: 2024-06
"""

import experiments_csv
from pandas import read_csv
import matplotlib.pyplot as plt
from fairpyx import divide, AgentBundleValueMatrix, Instance
import fairpyx.algorithms.high_multiplicity_fair_allocation as high
import fairpyx.algorithms.improved_high_multiplicity as imp
import fairpyx.algorithms.second_improved_high_multiplicity as sec
from typing import *
import numpy as np
from eefpy import Objective, EnvyNotion
from eefpy import solve as solve
from experiments_csv import single_plot_results, multi_plot_results

max_value = 1000
normalized_sum_of_values = 100
TIME_LIMIT = 60

algorithms_plot = [
    "high_multiplicity_fair_allocation",
    "solve",
    "improved_high_multiplicity_fair_allocation",
    "second_improved_high_multiplicity_fair_allocation"
]

# Define the specific algorithm you want to check
algorithms = [
    high.high_multiplicity_fair_allocation,
    solve,
    imp.improved_high_multiplicity_fair_allocation,
    sec.second_improved_high_multiplicity_fair_allocation
]

######### EXPERIMENT WITH UNIFORMLY-RANDOM DATA ##########

def evaluate_algorithm_on_instance(algorithm, instance):
    if algorithm is solve:
        agent_valuations = [[int(instance.agent_item_value(agent, item)) for item in instance.items] for agent in instance.agents]
        print(f' agent valuations:  {agent_valuations}')
        items = [instance.item_capacity(item) for item in instance.items]

        alloc = solve(num_agents=instance.num_of_agents, num_types=instance.num_of_items, agent_utils=agent_valuations, items=items, envy=EnvyNotion.EF, obj=Objective.NONE)
        allocation = {}
        for i, agent in enumerate(instance.agents):
            allocation[agent] = []
            for j, item in enumerate(instance.items):
                if alloc == []:
                    allocation[agent] = []
                else:
                    for sum in range(alloc[i][j]):
                        allocation[agent].append(item)
    else:
        allocation = divide(algorithm, instance)
    matrix = AgentBundleValueMatrix(instance, allocation)
    matrix.use_normalized_values()
    return {
        "utilitarian_value": matrix.utilitarian_value(),
        "egalitarian_value": matrix.egalitarian_value(),
    }

def course_allocation_with_random_instance_uniform(
        num_of_agents: int, num_of_items: int,
        value_noise_ratio: float,
        algorithm: Callable,
        random_seed: int):
    agent_capacity_bounds = [1000, 1000]
    item_capacity_bounds = [2, 10]
    np.random.seed(random_seed)
    instance = Instance.random_uniform(
        num_of_agents=num_of_agents, num_of_items=num_of_items,
        normalized_sum_of_values=normalized_sum_of_values,
        agent_capacity_bounds=agent_capacity_bounds,
        item_capacity_bounds=item_capacity_bounds,
        item_base_value_bounds=[1, max_value],
        item_subjective_ratio_bounds=[1 - value_noise_ratio, 1 + value_noise_ratio]
    )
    return evaluate_algorithm_on_instance(algorithm, instance)

def run_uniform_experiment():
    # Run on uniformly-random data:
    experiment = experiments_csv.Experiment("results/", "high_multi.csv", backup_folder="results/backup/")
    input_ranges = {
        "num_of_agents": [2, 3, 4, 5],
        "num_of_items": [2, 3, 5, 6],
        "value_noise_ratio": [0, 0.2, 0.5, 0.8, 1],
        "algorithm": algorithms,
        "random_seed": range(2),
    }
    experiment.run_with_time_limit(course_allocation_with_random_instance_uniform, input_ranges, time_limit=TIME_LIMIT)

######### EXPERIMENT WITH DATA SAMPLED FROM naor input DATA ##########

import json

filename = "data/naor_input.json"
with open(filename, "r", encoding="utf-8") as file:
    naor_input = json.load(file)

def course_allocation_with_random_instance_sample(
        max_total_agent_capacity: int,
        algorithm: Callable,
        random_seed: int):
    np.random.seed(random_seed)

    (agent_capacities, item_capacities, valuations) = (naor_input["agent_capacities"], naor_input["item_capacities"], naor_input["valuations"])
    instance = Instance.random_sample(
        max_num_of_agents=max_total_agent_capacity,
        max_total_agent_capacity=max_total_agent_capacity,
        prototype_agent_conflicts=[],
        prototype_agent_capacities=agent_capacities,
        prototype_valuations=valuations,
        item_capacities=item_capacities,
        item_conflicts=[]
    )
    return evaluate_algorithm_on_instance(algorithm, instance)

def run_naor_experiment():
    # Run on Ariel sample data:
    experiment = experiments_csv.Experiment("results/", "course_allocation_naor.csv", backup_folder="results/backup/")
    input_ranges = {
        "max_total_agent_capacity": [12],  # in reality: 1115
        "algorithm": algorithms,
        "random_seed": range(2),
    }
    experiment.run_with_time_limit(course_allocation_with_random_instance_sample, input_ranges, time_limit=TIME_LIMIT)

def create_plot_naor_experiment():
    csv_file = 'results/course_allocation_naor.csv'
    filter_dict = {}
    x_field = 'algorithm'
    y_fields = ['utilitarian_value', 'egalitarian_value', 'runtime']
    subplot_field = 'algorithm'
    single_plot_results(csv_file, filter_dict, x_field, y_fields[0], subplot_field, mean=True, save_to_file='results/naor_utilitarian_value_plot.png')
    single_plot_results(csv_file, filter_dict, x_field, y_fields[1], subplot_field, mean=True, save_to_file='results/naor_egalitarian_value_plot.png')
    single_plot_results(csv_file, filter_dict, x_field, y_fields[2], subplot_field, mean=True, save_to_file='results/naor_runtime_plot.png')

def create_plot_uniform():
    csv_file = 'results/high_multi.csv'
    filter_dict = {'algorithm': 'high_multiplicity_fair_allocation', 'random_seed': 0}
    x_field = 'value_noise_ratio'
    y_fields = ['utilitarian_value', 'egalitarian_value', 'runtime']
    subplot_field = 'num_of_items'  # Ensure this matches your CSV column name

    multi_plot_results(csv_file, filter_dict, x_field, y_fields[0], subplot_field=subplot_field, mean=True,
                       subplot_rows=1, subplot_cols=3, save_to_file='results/high_multiplicity_utilitarian_plot.png')


######### MAIN PROGRAM ##########

if __name__ == "__main__":
    import logging

    experiments_csv.logger.setLevel(logging.DEBUG)
    # run_naor_experiment()
    # create_plot_naor_experiment()
    # run_uniform_experiment()
    create_plot_uniform()
