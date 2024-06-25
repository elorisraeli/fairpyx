"""
Tests for the high-multiplicity-fair-allocation algorithm.

Article Title: High-Multiplicity Fair Allocation Made More Practical
Article URL: https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p260.pdf

Algorithm Name: High Multiplicity Fair Allocation
Algorithm Description: This algorithm finds an allocation maximizing the sum of utilities
                         for given instance with envy-freeness and Pareto-optimality constraints if exists.

Programmers: Naor Ladani and Elor Israeli
Since : 2024-05
"""

import pytest

import fairpyx
import numpy as np

from fairpyx.algorithms.high_multiplicity_fair_allocation import high_multiplicity_fair_allocation

NUM_OF_RANDOM_INSTANCES = 10


def test_feasibility():
    for i in range(NUM_OF_RANDOM_INSTANCES):
        np.random.seed(i)
        instance = fairpyx.Instance.random_uniform(
            num_of_agents=70, num_of_items=10, normalized_sum_of_values=1000,
            agent_capacity_bounds=[2, 6],
            item_capacity_bounds=[20, 40],
            item_base_value_bounds=[1, 1000],
            item_subjective_ratio_bounds=[0.5, 1.5]
        )
        allocation = fairpyx.divide(fairpyx.algorithms.high_multiplicity_fair_allocation.high_multiplicity_fair_allocation, instance=instance)
        print(allocation)
        fairpyx.validate_allocation(instance, allocation, title=f"Seed {i}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
