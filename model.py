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
    
    return {**initial_data, **expectations}


# --- Unified Timestep Data Generation Function (With Social Learning) ---
def generateTimestepData(model: AgentModel):
    graph = model.get_graph()

    num_trials = model["num_trials_per_step"]
    a_objective = model["a_objective"]
    b_objective = model["b_objective"]

    # Temporary storage for what happened this turn
    # Structure: { node_id: { "arm": "a" or "b", "successes": int, "failures": int } }
    step_results = {}

    # --- PHASE 1: ACTION (Everyone acts based on current beliefs) ---
    for node_id, node_data in graph.nodes(data=True):
        # Greedy Decision
        if node_data["a_expectation"] > node_data["b_expectation"]:
            chosen_arm = "a"
            objective = a_objective
        else:
            chosen_arm = "b"
            objective = b_objective

        # Run Trials
        successes = int(np.random.binomial(num_trials, objective))
        failures = max(0, num_trials - successes)

        # Record the visible outcome
        step_results[node_id] = {
            "arm": chosen_arm,
            "successes": successes,
            "failures": failures
        }

    # --- PHASE 2: SOCIAL LEARNING (Everyone updates based on self + neighbors) ---
    for node_id, node_data in graph.nodes(data=True):
        
        # 1. Identify who to learn from: Self + Neighbors
        # We use list(graph.neighbors(node_id)) to get connected nodes
        sources = [node_id] + list(graph.neighbors(node_id))

        # 2. Absorb data from all sources
        for source_id in sources:
            result = step_results[source_id]
            
            if result["arm"] == "a":
                node_data["a_alpha"] += result["successes"]
                node_data["a_beta"]  += result["failures"]
            else:
                # Result was on arm B
                node_data["b_alpha"] += result["successes"]
                node_data["b_beta"]  += result["failures"]

        # 3. Recalculate Expectations based on new total data
        a_denom = node_data["a_alpha"] + node_data["a_beta"]
        node_data["a_expectation"] = (
            node_data["a_alpha"] / a_denom if a_denom > 0 else 0
        )

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