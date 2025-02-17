import numpy as np
import matplotlib.pyplot as plt
import data
import WindProcess
import PriceProcess
from pyomo.environ import *


def evaluate_policy(policy, E=100, T=24):
    """
    Parameters:
        E (int): Number of independent experiments (days).
        T (int): Number of time steps per experiment (hours in a day).
    """
    params = data.get_fixed_data()
    cost_results = []

    for _ in range(E):
        wind_power = [params['target_mean_wind']]
        price_series = [params['mean_price']]

        # Generate wind and price time series
        for t in range(1, T):
            wind_power.append(
                WindProcess.wind_model(wind_power[-1], wind_power[-2] if t > 1 else wind_power[-1], params))
            price_series.append(
                PriceProcess.price_model(price_series[-1], price_series[-2] if t > 1 else price_series[-1],
                                         wind_power[-1], params))

        hydrogen_storage = 0
        electrolyzer_status = 0
        total_cost = 0

        for t in range(T):
            state = (wind_power[t], electrolyzer_status, hydrogen_storage, price_series[t])
            action = policy(t, state, params)

            e_next, p2h, h2p, grid_power = action

            # Update hydrogen storage based on conversion rates
            hydrogen_storage = max(0,
                                   hydrogen_storage + params['conversion_p2h'] * p2h - params['conversion_h2p'] * h2p)

            # Compute cost using the reward function
            cost = price_series[t] * grid_power + params['electrolyzer_cost'] * e_next
            total_cost += cost

            # Update electrolyzer status
            electrolyzer_status = e_next

        cost_results.append(total_cost)

    return np.mean(cost_results)


def dummy_policy(t, state, params):
    """Dummy policy that never uses the electrolyzer."""
    return 0, 0, 0, params['demand_schedule'][t]  # Only draws from the grid


if __name__ == "__main__":
    avg_cost = evaluate_policy(dummy_policy, E=100, T=24)
    print(f"Average cost with dummy policy: {avg_cost:.2f}")
