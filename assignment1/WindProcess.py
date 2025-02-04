# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:37:57 2025

@author: geots
"""

import numpy as np

def wind_model(current, previous, data):
    """
    Wind model to simulate realistic stochastic transitions.

    Args:
        current (float): Current wind generation.
        previous (float): Wind generation at the previous time step.
        data (dict): Fixed data containing model parameters.

    Returns:
        float: Next wind generation.
    """
    target_mean = data['target_mean_wind']
    reversion_strength = data['wind_reversion_strength']
    extreme_event_prob = data['extreme_event_prob_wind']

    correlated_noise = np.random.normal(0, 1) + 0.8 * (current - previous)
    mean_reversion = reversion_strength * (target_mean - current)

    if np.random.rand() < extreme_event_prob:
        extreme_event = np.random.choice([np.random.uniform(10, 15), np.random.uniform(0, 2)])
    else:
        extreme_event = 0

    next_wind = current + mean_reversion + correlated_noise + extreme_event
    return max(next_wind, 0)