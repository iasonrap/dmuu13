import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from pyomo.environ import *

# Load necessary modules dynamically
def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

v2_data = load_module("v2_data", "v2_data.py")
EvaluationPolicy = load_module("EvaluationPolicy", "EvaluationPolicy.py")
ADP = load_module("ADP", "ADP.py")
Stochastic = load_module("Stochastic", "Stochastic.py")
MILP = load_module("MILP", "MILP.py")
WindProcess = load_module("WindProcess", "WindProcess.py")
PriceProcess = load_module("PriceProcess", "PriceProcess.py")

# Load parameters
params = v2_data.get_fixed_data()

# Number of experiments
num_experiments = 20

# Evaluate dummy policy
dummy_costs = [EvaluationPolicy.evaluate_policy(EvaluationPolicy.dummy_policy, E=1) for _ in range(num_experiments)]

# Evaluate optimal-in-hindsight solution using MILP
solver = SolverFactory('gurobi')
solver.solve(MILP.model)
optimal_costs = [sum(MILP.model.g[t].value * params['mean_price'] + MILP.model.e[t].value * params['electrolyzer_cost'] for t in MILP.model.T) for _ in range(num_experiments)]

# Evaluate stochastic programming policy configurations
stochastic_costs = [[EvaluationPolicy.evaluate_policy(lambda state, p: Stochastic.stochastic_optimization_policy(state, p), E=1) for _ in range(num_experiments)] for _ in range(4)]

# Evaluate expected value policy (single scenario case)
expected_value_costs = [EvaluationPolicy.evaluate_policy(lambda state, p: Stochastic.stochastic_optimization_policy(state, p), E=1) for _ in range(num_experiments)]

# Evaluate ADP policy
adp_costs = [EvaluationPolicy.evaluate_policy(lambda state, p: ADP.policy_decision((state[0], 0, state[1], state[2], state[3]), use_stochastic=True), E=1) for _ in range(num_experiments)]


# Plot histogram of policy costs
plt.figure(figsize=(10, 6))
plt.hist(dummy_costs, bins=10, alpha=0.5, label="Dummy Policy")
plt.hist(optimal_costs, bins=10, alpha=0.5, label="Optimal-in-Hindsight")
for i, costs in enumerate(stochastic_costs):
    plt.hist(stochastic_costs, bins=10, alpha=0.5, label=f"Stochastic Config {i+1}")
plt.hist(expected_value_costs, bins=10, alpha=0.5, label="Expected Value Policy")
plt.hist(adp_costs, bins=10, alpha=0.5, label="ADP Policy")

plt.xlabel("Cost")
plt.ylabel("Frequency")
plt.title("Policy Cost Distribution Across 20 Experiments")
plt.legend()
plt.grid()
plt.show()
