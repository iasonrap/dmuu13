import numpy as np
import v2_data as data
import WindProcess as wind
import PriceProcess as price


def stochastic_optimization_policy(state, params):
    """
    Decision-making policy using stochastic optimization.

    """
    wind_power, electrolyzer_status, hydrogen_storage, electricity_price = state

    # Initialize decision variables
    e_next = electrolyzer_status  # Default to keeping the same status
    p2h = 0  # Power allocated to hydrogen production
    h2p = 0  # Hydrogen used for power generation
    g = params['demand_schedule'][0]  # Default: draw from the grid

    # Stochastic scenario generation (simplified)
    future_wind = [wind.wind_model(wind_power, wind_power, params) for _ in range(2)]  # Two branches per stage
    future_prices = [price.price_model(electricity_price, electricity_price + np.random.normal(0, 2), w, params) for w
                     in future_wind]

    # Compute expected future conditions
    expected_price = np.mean(future_prices)
    expected_wind = np.mean(future_wind)

    # Power balance constraint: Ensure total power meets demand
    demand = params['demand_schedule'][0]
    if wind_power + h2p < demand:
        g = demand - (wind_power + h2p)  # Draw remaining power from the grid

    # Hydrogen balance constraint: Ensure hydrogen storage remains non-negative
    new_hydrogen_storage = hydrogen_storage + params['conversion_p2h'] * p2h - h2p
    if new_hydrogen_storage < 0:
        h2p = hydrogen_storage  # Restrict hydrogen use to available storage

    # Electrolyzer activation decision based on price and wind power availability
    if expected_price < params['mean_price'] and expected_wind > wind_power:
        e_next = 1  # Turn ON electrolyzer
        p2h = min(params['p2h_max_rate'], wind_power)  # Convert wind power to hydrogen

    # Hydrogen-to-power conversion decision based on electricity price
    if electricity_price > params['mean_price'] and hydrogen_storage > 0:
        h2p = min(hydrogen_storage, params['h2p_max_rate'])  # Convert stored hydrogen to power

    # Ensure decision variables remain non-negative
    p2h = max(0, p2h)
    h2p = max(0, h2p)
    g = max(0, g)

    # Provide detailed output explanation
    print(f"\nDecision Breakdown:")
    print(f"  - Wind Availability: {wind_power} MW (Wind power available for use)")
    print(
        f"  - Electrolyzer Status: {'ON' if e_next else 'OFF'} (Electrolyzer is {'active' if e_next else 'inactive'})")
    print(
        f"  - Power to Hydrogen: {p2h} MW (Power allocated to produce hydrogen, depends on wind availability and electrolyzer status)")
    print(
        f"  - Hydrogen to Power: {h2p} MW (Hydrogen used for power generation, depends on storage and electricity price)")
    print(f"  - Grid Power Draw: {g} MW (Power drawn from the grid to meet remaining demand if necessary)")
    return wind_power,e_next, p2h, h2p, g


if __name__ == "__main__":
    # Example  state
    state = (2.5, 0, 10, 30)
    params = data.get_fixed_data()
    decision = stochastic_optimization_policy(state, params)
    print(f"\nFinal Decision Output (Tuple Format): {decision}")
