import numpy as np
import matplotlib.pyplot as plt
import data
import WindProcess
import PriceProcess
from pyomo.environ import *

# Load fixed data
params = data.get_fixed_data()
T = params['num_timeslots']

# Initialize Pyomo model
model = ConcreteModel()

# Sets
model.T = RangeSet(0, T-1)

# Decision variables
model.e = Var(model.T, within=Binary)
model.p2h = Var(model.T, within=NonNegativeReals, bounds=(0, params['p2h_rate']))
model.h = Var(model.T, within=NonNegativeReals, bounds=(0, params['hydrogen_capacity']))
model.h2p = Var(model.T, within=NonNegativeReals, bounds=(0, params['h2p_rate']))
model.g = Var(model.T, within=NonNegativeReals, bounds=(0, max(params['demand_schedule'])))  # Upper bound to avoid infeasibility

# Initialize wind and price time series
wind_trajectory = [params['target_mean_wind']]
price_trajectory = [params['mean_price']]

for t in range(1, T):
    wind_trajectory.append(WindProcess.wind_model(wind_trajectory[-1], wind_trajectory[-2] if t > 1 else wind_trajectory[-1], params))
    price_trajectory.append(PriceProcess.price_model(price_trajectory[-1], price_trajectory[-2] if t > 1 else price_trajectory[-1], wind_trajectory[-1], params))

# Objective function: Minimize cost
def objective_rule(m):
    return sum(m.g[t] * price_trajectory[t] + m.e[t] * params['electrolyzer_cost'] for t in m.T)

model.obj = Objective(rule=objective_rule, sense=minimize)

# Constraints
def energy_balance_rule(m, t):
    demand = params['demand_schedule'][t]
    wind_power = wind_trajectory[t]
    return demand <= wind_power + m.h2p[t] + m.g[t] - m.p2h[t]  # Relaxed constraint to ensure feasibility

model.energy_balance = Constraint(model.T, rule=energy_balance_rule)

# Hydrogen storage dynamics
def hydrogen_storage_rule(m, t):
    if t == 0:
        return m.h[t] == params['conversion_p2h'] * m.p2h[t] - params['conversion_h2p'] * m.h2p[t]
    else:
        return m.h[t] == m.h[t-1] + params['conversion_p2h'] * m.p2h[t] - params['conversion_h2p'] * m.h2p[t]

model.hydrogen_storage = Constraint(model.T, rule=hydrogen_storage_rule)

# Electrolyzer ON constraint
def electrolyzer_constraint_rule(m, t):
    return m.p2h[t] <= m.e[t] * params['p2h_rate']

model.electrolyzer_constraint = Constraint(model.T, rule=electrolyzer_constraint_rule)

# Ensure hydrogen storage remains non-negative
def hydrogen_non_negative_rule(m, t):
    return m.h[t] >= 0

model.hydrogen_non_negative = Constraint(model.T, rule=hydrogen_non_negative_rule)

# Solve model
solver = SolverFactory('gurobi')
solver.solve(model)

# Extract results
results = {
    'grid_power': [model.g[t].value for t in model.T],
    'power_to_hydrogen': [model.p2h[t].value for t in model.T],
    'hydrogen_to_power': [model.h2p[t].value for t in model.T],
    'hydrogen_storage_level': [model.h[t].value for t in model.T],
    'electrolyzer_status': [model.e[t].value for t in model.T],
}

# Plot results using Plots.py structure
times = range(T)
plt.figure(figsize=(14, 10))

plt.subplot(8, 1, 1)
plt.plot(times, wind_trajectory, label="Wind Power", color="blue")
plt.ylabel("Wind Power")
plt.legend()

plt.subplot(8, 1, 2)
plt.plot(times, params['demand_schedule'], label="Demand Schedule", color="orange")
plt.ylabel("Demand")
plt.legend()

plt.subplot(8, 1, 3)
plt.step(times, results['electrolyzer_status'], label="Electrolyzer Status", color="red", where="post")
plt.ylabel("El. Status")
plt.legend()

plt.subplot(8, 1, 4)
plt.plot(times, results['hydrogen_storage_level'], label="Hydrogen Level", color="green")
plt.ylabel("Hydr. Level")
plt.legend()

plt.subplot(8, 1, 5)
plt.plot(times, results['power_to_hydrogen'], label="p2h", color="orange")
plt.ylabel("p2h")
plt.legend()

plt.subplot(8, 1, 6)
plt.plot(times, results['hydrogen_to_power'], label="h2p", color="blue")
plt.ylabel("h2p")
plt.legend()

plt.subplot(8, 1, 7)
plt.plot(times, results['grid_power'], label="Grid Power", color="green")
plt.ylabel("Grid Power")
plt.legend()

plt.subplot(8, 1, 8)
plt.plot(times, price_trajectory, label="price", color="red")
plt.ylabel("Price")
plt.xlabel("Time")
plt.legend()

plt.tight_layout()
plt.show()
