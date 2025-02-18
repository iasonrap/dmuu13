import numpy as np
import matplotlib.pyplot as plt
import data
import v2_data
import WindProcess
import PriceProcess
from pyomo.environ import *

def evaluate_policy(policy, E=20, H=24):
    """
        E (int): Number of independent experiments (days).
        H (int): Number of time steps per experiment (hours in a day).
    """
    params = v2_data.get_fixed_data()
    cost_results = []

    for _ in range(E):
        # Initialize stochastic variables
        wind_power = [params['target_mean_wind']]
        price_series = [params['mean_price']]

        # Generate wind and price time series
        for t in range(1, H):
            wind_power.append(
                WindProcess.wind_model(wind_power[-1], wind_power[-2] if t > 1 else wind_power[-1], params))
            price_series.append(
                PriceProcess.price_model(price_series[-1], price_series[-2] if t > 1 else price_series[-1],
                                         wind_power[-1], params))

        # Initialize state variables
        hydrogen_storage = 0
        electrolyzer_status = 0
        total_cost = 0

        for t in range(H):
            state = (wind_power[t], electrolyzer_status, hydrogen_storage, price_series[t])

            # Get decisions from policy
            decisions = policy(state, params)

            # Unpack policy decisions
            e_next, p2h, h2p, grid_power = decisions

            # Check and correct decisions if inconsistent
            if p2h < 0: p2h = 0
            if h2p < 0: h2p = 0
            if hydrogen_storage + params['conversion_p2h'] * p2h - params['conversion_h2p'] * h2p < 0:
                h2p = 0  # Prevent negative hydrogen storage

            # Calculate cost for this stage
            stage_cost = price_series[t] * grid_power + params['electrolyzer_cost'] * e_next
            total_cost += stage_cost

            # Update state for next stage
            hydrogen_storage = max(0,
                                   hydrogen_storage + params['conversion_p2h'] * p2h - params['conversion_h2p'] * h2p)
            electrolyzer_status = e_next

        # Store total cost for this experiment
        cost_results.append(total_cost)

    # Return expected policy cost
    return np.mean(cost_results)


def dummy_policy(state, params):
    """Dummy policy that never uses the electrolyzer."""
    return 0, 0, 0, params['demand_schedule'][0]  # Only draws from the grid


if __name__ == "__main__":
    avg_cost = evaluate_policy(dummy_policy, E=20)
    print(f"Expected policy cost with dummy policy: {avg_cost:.2f}")
