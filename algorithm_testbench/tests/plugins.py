"""Small adversarial algorithms for testing process isolation and verification."""

import time

from algorithm_testbench.model import AlgorithmResult


def mutate(scenario, limits, config):
    scenario.graph.nodes[scenario.entry]["resources"]["cpu"] = 0
    limits["cpu"] = 1e9
    return AlgorithmResult({scenario.entry: list(scenario.graph)})


def slow(scenario, limits, config):
    time.sleep(5)
    return AlgorithmResult()


def false_optimum(scenario, limits, config):
    return AlgorithmResult({n: [n] for n in scenario.graph}, "optimal")
