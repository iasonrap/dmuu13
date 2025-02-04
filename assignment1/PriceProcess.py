# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:39:11 2025

@author: geots
"""

import numpy as np

def price_model(current_price, previous_price, projected_wind, data):
    """
    Price process with dependence on previous prices and projected wind generation.

    Args:
        current_price (float): Current electricity price.
        previous_price (float): Electricity price at the previous time step.
        projected_wind (float): Projected wind generation for the next time step.
        data (dict): Fixed data containing model parameters.

    Returns:
        float: Next price.
    """
    mean_price = data['mean_price']
    reversion_strength = data['price_reversion_strength']
    wind_influence = data['wind_influence_on_price']
    price_cap = data['price_cap']
    price_floor = data['price_floor']

    mean_reversion = reversion_strength * (mean_price - current_price)
    wind_effect = wind_influence * projected_wind
    noise = np.random.normal(0, 1)

    next_price = current_price + 0.6 * (current_price - previous_price) + mean_reversion + wind_effect + noise

    if next_price < 0:
        if np.random.rand() > 0.2:
            next_price = np.random.uniform(0, mean_price * 0.3)

    return max(min(next_price, price_cap), price_floor)