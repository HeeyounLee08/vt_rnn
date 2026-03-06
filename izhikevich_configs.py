"""
izhikevich_configs.py - Configuration for Izhikevich neuron types
"""

# Cell IDs for each neuron type
cids = [
    'RS',   # 1: tonic spiking (Regular Spiking)
    'PS',   # 2: phasic spiking
    'TB',   # 3: tonic bursting
    'PB',   # 4: phasic bursting
    'MM',   # 5: mixed mode
    'FA',   # 6: spike frequency adaptation
    'E1',   # 7: Class 1 (Excitability type 1)
    'E2',   # 8: Class 2 (Excitability type 2)
    'SL',   # 9: spike latency
    'SO',   # 10: subthreshold oscillations (not available)
    'R',    # 11: resonator
    'IN',   # 12: integrator
    'RBS',  # 13: rebound spike
    'RBB',  # 14: rebound burst
    'TV',   # 15: threshold variability
    'B1',   # 16: bistability
    'DA',   # 17: depolarizing after-potential (not available)
    'A',    # 18: accomodation
    'IS',   # 19: inhibition-induced spiking
    'IB',   # 20: inhibition-induced bursting
    'B2'    # 21: bistability 2
]

# Full names for each neuron type
index_to_name = {
    1: 'Tonic Spiking',
    2: 'Phasic Spiking',
    3: 'Tonic Bursting',
    4: 'Phasic Bursting',
    5: 'Mixed Mode',
    6: 'Spike Frequency Adaptation',
    7: 'Class 1 Excitability',
    8: 'Class 2 Excitability',
    9: 'Spike Latency',
    10: 'Subthreshold Oscillations',
    11: 'Resonator',
    12: 'Integrator',
    13: 'Rebound Spike',
    14: 'Rebound Burst',
    15: 'Threshold Variability',
    16: 'Bistability',
    17: 'Depolarizing After-Potential',
    18: 'Accommodation',
    19: 'Inhibition-Induced Spiking',
    20: 'Inhibition-Induced Bursting',
    21: 'Bistability 2'
}
