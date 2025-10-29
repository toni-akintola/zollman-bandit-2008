# Zollman Bandit Model (2008 Variation)
A bandit model by [Kevin Zollman](https://www.kevinzollman.com/) exploring how network structure affects scientific communities' ability to reach accurate consensus. This is the **2008 variation** focusing on division of labor and cognitive diversity. View the full paper [here](https://www.kevinzollman.com/uploads/5/0/3/6/50361245/zollman_-_communication_structure.pdf).

## Abstract
> There is growing interest in understanding and eliciting division of labor
within groups of scientists. This paper illustrates the need for this division of labor
through a historical example, and a formal model is presented to better analyze
situations of this type. Analysis of this model reveals that a division of labor can
be maintained in two different ways: by limiting information or by endowing
the scientists with extreme beliefs. If both features are present however, cognitive
diversity is maintained indefinitely, and as a result agents fail to converge to the
truth. Beyond the mechanisms for creating diversity suggested here, this shows that
the real epistemic goal is not diversity but transient diversity.

## Model Implementation (2008)
This variation implements a multi-armed bandit problem where scientists:
- Choose between two methodologies (A or B) based on expected values
- Update their beliefs using Beta distributions after observing outcomes
- Share information within their network structure

### Key Features
- **Learning Mechanism**: Bayesian updating with Beta-Binomial conjugate priors
- **Decision Rule**: Choose methodology with higher expected value
- **Information Sharing**: Agents observe neighbors' experimental outcomes
- **Network Effects**: Different structures (complete, cycle, wheel) affect information flow

## Model Parameters
* `num_nodes`: Size of network (default: 10)
* `a_objective_2008`: True success probability of methodology A (default: 0.19)
* `b_objective_2008`: True success probability of methodology B (default: 0.71)
* `num_trials_per_step_2008`: Number of trials per experiment (default: 5)
* `max_prior_value_2008`: Maximum value for initial Beta distribution parameters (default: 4.0)
* `graph_type`: Network structure - "complete", "cycle", or "wheel"

## Agent Attributes
Each scientist maintains:
- `a_alpha`, `a_beta`: Beta distribution parameters for methodology A
- `b_alpha`, `b_beta`: Beta distribution parameters for methodology B  
- `a_expectation`, `b_expectation`: Expected success rates for each methodology