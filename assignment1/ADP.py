import numpy as np
from WindProcess import wind_model
from PriceProcess import price_model
from v2_data import get_fixed_data
from Stochastic import stochastic_optimization_policy

data = get_fixed_data()


def cost_function(state, action):

    wt, dt, et, ht, lambdat = state
    et_next, p2ht, h2pt, gt = action

    return lambdat * gt + data['electrolyzer_cost'] * p2ht + 0.05 * h2pt


def get_feasible_actions(state):

    wt, dt, et, ht, lambdat = state
    actions = []

    for et_next in [0, 1]:
        for p2ht in np.linspace(0, min(data['p2h_max_rate'], max(0, wt - dt)), 5):
            for h2pt in np.linspace(0, min(data['h2p_max_rate'], min(ht, dt - wt)), 5):
                for gt in np.linspace(0, max(0, dt - wt - h2pt), 5):
                    actions.append((et_next, p2ht, h2pt, gt))

    return actions


def policy_decision(state, use_stochastic=False):
    """
    Compute the here-and-now decision purely based on the current state.
    """
    if use_stochastic:
        # Adjust state format to match stochastic_optimization_policy expectations
        wt, dt, et, ht, lambdat = state
        stochastic_state = (wt, et, ht, lambdat)  # Remove demand (dt)
        return stochastic_optimization_policy(stochastic_state, data)

    best_action = None
    min_cost = float('inf')

    for action in get_feasible_actions(state):
        cost = cost_function(state, action)

        if cost < min_cost:
            min_cost = cost
            best_action = action

    return best_action


# Example usage
state = (5, 10, 1, 20, 35)  # Example initial state
optimal_action = policy_decision(state, use_stochastic=True)
print("Optimal Here-and-Now Action:", optimal_action)