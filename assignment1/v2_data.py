# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 16:18:27 2024

@author: geots
"""

import numpy as np

def get_fixed_data():
    """
    Returns the fixed data for the energy hub simulation.
    """
    num_timeslots = 24
    return {
        # Conversion efficiencies
        'conversion_p2h': 0.9,
        'conversion_h2p': 0.8,

        # Hydrogen storage capacity
        'hydrogen_capacity': 15,
        
        'p2h_max_rate': 5,
        'h2p_max_rate': 5,

        # Electrolyzer cost
        'electrolyzer_cost': 1, 

        # Wind model parameters
        'target_mean_wind': 4.5,
        'wind_reversion_strength': 0.15,
        'extreme_event_prob_wind': 0.03,

        # Price model parameters
        'mean_price': 35,
        'price_reversion_strength': 0.12,
        'wind_influence_on_price': -0.6,
        'price_cap': 90,  
        'price_floor': 0,
        
        
        'num_timeslots': num_timeslots,
        'demand_schedule': [5 + 2 * np.sin(2 * np.pi * t / 24) for t in range(num_timeslots)]

        
    }