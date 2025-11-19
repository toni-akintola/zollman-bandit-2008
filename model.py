import random
import numpy as np
from emergent.main import AgentModel
from typing import Tuple, List
import networkx as nx
import matplotlib.pyplot as plt


# --- Unified Initial Data Generation Function ---
def generateInitialData(model: AgentModel):
    max_prior = model["max_prior_value"]
    # Ensure max_prior is positive, default to a small value if not.
    if max_prior <= 0:
        max_prior = 1.0
    initial_data = {
        "a_alpha": random.uniform(1e-5, max_prior),
        "a_beta": random.uniform(1e-5, max_prior),
        "b_alpha": random.uniform(1e-5, max_prior),
        "b_beta": random.uniform(1e-5, max_prior),
    }
    # Ensure denominators are not zero
    a_denom = initial_data["a_alpha"] + initial_data["a_beta"]
    b_denom = initial_data["b_alpha"] + initial_data["b_beta"]

    expectations = {
        "a_expectation": initial_data["a_alpha"] / a_denom if a_denom > 0 else 0,
        "b_expectation": initial_data["b_alpha"] / b_denom if b_denom > 0 else 0,
    }
    initial_data.update(expectations)
    return initial_data


# --- Unified Timestep Data Generation Function ---
def generateTimestepData(model: AgentModel):

    graph = model.get_graph()

    num_trials = model["num_trials_per_step"]
    a_objective = model["a_objective"]
    b_objective = model["b_objective"]

    for _node, node_data in graph.nodes(data=True):
        if node_data["a_expectation"] > node_data["b_expectation"]:
            successes = int(np.random.binomial(num_trials, a_objective, size=None))
            node_data["a_alpha"] += successes
            node_data["a_beta"] += max(0, num_trials - successes)
            a_denom = node_data["a_alpha"] + node_data["a_beta"]
            node_data["a_expectation"] = (
                node_data["a_alpha"] / a_denom if a_denom > 0 else 0
            )
        else:
            successes = int(np.random.binomial(num_trials, b_objective, size=None))
            node_data["b_alpha"] += successes
            node_data["b_beta"] += max(0, num_trials - successes)
            b_denom = node_data["b_alpha"] + node_data["b_beta"]
            node_data["b_expectation"] = (
                node_data["b_alpha"] / b_denom if b_denom > 0 else 0
            )

    model.set_graph(graph)


# --- Model Construction ---
def constructModel() -> AgentModel:
    model = AgentModel()

    model.update_parameters(
        {
            # Common parameters editable in UI, used for graph setup by Emergent
            "num_nodes": 10,
            "graph_type": "complete",  # Options: "complete", "wheel", "cycle"
            # Parameters for "2008" variation
            "a_objective": 0.19,
            "b_objective": 0.71,
            "num_trials_per_step": 5,
            "max_prior_value": 4.0,
        }
    )

    # Set the unified functions
    model.set_initial_data_function(generateInitialData)
    model.set_timestep_function(generateTimestepData)

    return model


if __name__ == "__main__":
    model = constructModel()
    model.initialize_graph()