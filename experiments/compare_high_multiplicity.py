"""
Compare the performance of a specific algorithm for fair course allocation.

Programmer: Erel Segal-Halevi
Since: 2023-07
"""
import experiments_csv

from fairpyx import divide, AgentBundleValueMatrix, Instance
import fairpyx.algorithms.high_multiplicity_fair_allocation as high
import json
from typing import *
import numpy as np

max_value = 60
normalized_sum_of_values = 60
TIME_LIMIT = 60

# Define the specific algorithm you want to check
algorithms = [high.high_multiplicity_fair_allocation]


def evaluate_algorithm_on_instance(algorithm, instance):
    allocation = divide(algorithm, instance)
    matrix = AgentBundleValueMatrix(instance, allocation)
    matrix.use_normalized_values()
    return {
        "utilitarian_value": matrix.utilitarian_value(),
        "egalitarian_value": matrix.egalitarian_value(),

    }

######### EXPERIMENT WITH UNIFORMLY-RANDOM DATA ##########

def course_allocation_with_random_instance_uniform(
    num_of_agents:int, num_of_items:int,
    value_noise_ratio:float,
    algorithm:Callable,
    random_seed: int):
    agent_capacity_bounds =  [2, 8, 12]
    item_capacity_bounds = [2, 4, 10]
    np.random.seed(random_seed)
    instance = Instance.random_uniform(
        num_of_agents=num_of_agents, num_of_items=num_of_items,
        normalized_sum_of_values=normalized_sum_of_values,
        agent_capacity_bounds=agent_capacity_bounds,
        item_capacity_bounds=item_capacity_bounds,
        item_base_value_bounds=[1,max_value],
        item_subjective_ratio_bounds=[1-value_noise_ratio, 1+value_noise_ratio]
        )
    return evaluate_algorithm_on_instance(algorithm, instance)


def run_uniform_experiment():
    # Run on uniformly-random data:
    experiment = experiments_csv.Experiment("results/", "high_multi.csv", backup_folder="results/backup/")
    input_ranges = {
        "num_of_agents": [2,3,4,5],
        "num_of_items":  [2,3,5,6],
        "value_noise_ratio": [0, 0.5, 1.5],
        "algorithm": algorithms,
        "random_seed": range(3),
    }
    experiment.run_with_time_limit(course_allocation_with_random_instance_uniform, input_ranges, time_limit=TIME_LIMIT)


######### EXPERIMENT WITH DATA SAMPLED FROM ARIEL 5783 DATA ##########

import json
filename = "data/naor_input.json"
with open(filename, "r", encoding="utf-8") as file:
    naor_input = json.load(file)

def course_allocation_with_random_instance_sample(
    max_total_agent_capacity:int,
    algorithm:Callable,
    random_seed: int,):
    np.random.seed(random_seed)

    ( agent_capacities, item_capacities, valuations) = \
        (naor_input["agent_capacities"], naor_input["item_capacities"],naor_input["valuations"])
    instance = Instance.random_sample(
        max_num_of_agents = max_total_agent_capacity,
        max_total_agent_capacity = max_total_agent_capacity,
        prototype_agent_conflicts=[],
        prototype_agent_capacities=agent_capacities,
        prototype_valuations=valuations,
        item_capacities=item_capacities,
        item_conflicts=[])

    return evaluate_algorithm_on_instance(algorithm, instance)

def run_naor_experiment():
    # Run on Ariel sample data:z
    experiment = experiments_csv.Experiment("results/", "course_allocation_naorl.csv", backup_folder="results/backup/")
    input_ranges = {
        "max_total_agent_capacity": [12], # in reality: 1115
        "algorithm": algorithms,
        "random_seed": range(13),
    }
    experiment.run_with_time_limit(course_allocation_with_random_instance_sample, input_ranges, time_limit=TIME_LIMIT)



######### MAIN PROGRAM ##########


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    # run_naor_experiment()
    run_uniform_experiment()
