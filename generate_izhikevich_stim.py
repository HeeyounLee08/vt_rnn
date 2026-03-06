"""
generate_izhikevich_stim.py - Faithful translation of generate_izhikevich_stim.m

This code generates a stimulus appropriate to the specified type of
Izhikevich neuron. Defaults for each cell type are those used in
"Capturing the dynamical repertoire of single neurons with GLMs" (Weber
& Pillow 2017).
"""

import numpy as np


def generate_izhikevich_stim(cellType, T=None):
    """
    [I, dt] = generate_izhikevich_stim(cellType, T)

    This code generates a stimulus appropriate to the specified type of
    Izhikevich neuron.

    Parameters:
    -----------
    cellType : int
        Type of Izhikevich neuron. Choose from:
            1. tonic spiking
            2. phasic spiking
            3. tonic bursting
            4. phasic bursting
            5. mixed mode
            6. spike frequency adaptation
            7. Class 1
            8. Class 2
            9. spike latency
            10. subthreshold oscillations -- not available
            11. resonator
            12. integrator
            13. rebound spike
            14. rebound burst
            15. threshold variability
            16. bistability
            17. depolarizing after-potential -- not available
            18. accomodation
            19. inhibition-induced spiking
            20. inhibition-induced bursting
            21. bistability 2 (Not in original Izhikevich paper)
    T : float, optional
        Max time of stimulus, in ms (default: 10000)

    Returns:
    --------
    I : ndarray
        Stimulus (current input)
    dt : float
        Time step, in ms
    """
    # check for unavailable behaviors
    if cellType in [10, 17]:
        print("This is a subthreshold behavior, which can't be captured by a GLM. "
              "No example stimulus has been designed for this cell type.")
        return None, None

    # set defaults
    if T is None:
        T = 10000

    # parameters
    #       I        dt
    pars = np.array([
        [14,       0.1],     # 1. tonic spiking
        [0.5,      0.1],     # 2. phasic spiking
        [10,       0.1],     # 3. tonic bursting
        [0.6,      0.1],     # 4. phasic bursting
        [10,       0.1],     # 5. mixed mode
        [20,       0.1],     # 6. spike frequency adaptation
        [25,       0.1],     # 7. Class 1
        [0.5,      0.1],     # 8. Class 2
        [3.49,     0.1],     # 9. spike latency
        [0,        1],       # 10. subthreshold oscillations
        [0.3,      0.5],     # 11. resonator
        [27.4,     0.5],     # 12. integrator
        [-5,       0.1],     # 13. rebound spike
        [-5,       0.1],     # 14. rebound burst
        [2.3,      1],       # 15. threshold variability
        [26.1,     0.05],    # 16. bistability
        [0,        0.1],     # 17. depolarizing after-potential
        [20,       0.1],     # 18. accomodation
        [70,       0.1],     # 19. inhibition-induced spiking
        [70,       0.1],     # 20. inhibition-induced bursting
        [26.1,     0.05]     # 21. bistability 2
    ])

    Ival = pars[cellType - 1, 0]
    dt = pars[cellType - 1, 1]

    t = np.arange(dt, T + dt, dt)

    # generate stimulus
    I = np.zeros(len(t))

    stepLength = 500  # in units of ms
    nStepsUp = int(np.floor(T / stepLength / 2))

    if cellType in [1, 2, 3, 4, 5, 6, 10, 19, 20]:
        if cellType in [19, 20]:
            I = 80 * np.ones(len(t))
        for i in range(1, nStepsUp + 1):
            idx = (t > (stepLength + stepLength * 2 * (i - 1))) & (t < (stepLength * 2 * i + 1))
            I[idx] = Ival

    elif cellType in [7, 8]:
        if cellType == 7:
            stepSizes = np.arange(15, 31)
        elif cellType == 8:
            stepSizes = np.arange(0.1, 0.725, 0.025)
        for i in range(len(stepSizes)):
            idx = (t > (stepLength + stepLength * 2 * i)) & (t < (stepLength * 2 * (i + 1) + 1))
            I[idx] = stepSizes[i]

    elif cellType == 9:
        stepLength = 150
        nStepsUp = int(np.floor(T / stepLength / 2))

        for i in range(1, nStepsUp + 1):
            idx = (t > (stepLength * 1.94 + stepLength * 2 * (i - 1))) & (t < (stepLength * 2 * i + 1))
            I[idx] = Ival

    elif cellType == 11:  # resonator
        stepLength = 150
        nStepsUp = int(np.floor(T / stepLength / 2))

        for i in range(2, nStepsUp + 1):
            pulseLength = round(5 / dt)
            idx = (t > (stepLength + stepLength * 2 * (i - 1))) & \
                  (t < (stepLength + stepLength * 2 * (i - 1) + pulseLength))
            I[idx] = Ival
            # second pulse
            idx = (t > (stepLength + stepLength * 2 * (i - 1) + pulseLength + 2 * i + pulseLength / 2)) & \
                  (t < (stepLength + stepLength * 2 * (i - 1) + 2 * pulseLength + 2 * i + pulseLength / 2))
            I[idx] = Ival

    elif cellType == 12:  # integrator
        stepLength = 250
        nStepsUp = int(np.floor(T / stepLength / 2))

        for i in range(3, nStepsUp + 1):
            pulseLength = round(4 / dt)
            idx = (t > (stepLength + stepLength * 2 * (i - 1))) & \
                  (t < (stepLength + stepLength * 2 * (i - 1) + pulseLength))
            I[idx] = Ival
            # second pulse
            idx = (t > (stepLength + stepLength * 2 * (i - 1) + pulseLength + 6 * i + pulseLength / 2)) & \
                  (t < (stepLength + stepLength * 2 * (i - 1) + 2 * pulseLength + 6 * i + pulseLength / 2))
            I[idx] = Ival

    elif cellType in [13, 14]:  # rebound spike, rebound burst
        for i in range(1, nStepsUp + 1):
            idx = (t > (stepLength * 1.6 + stepLength * 2 * (i - 1))) & (t < (stepLength * 2 * i + 1))
            I[idx] = Ival

    elif cellType == 15:  # threshold variability, steps at random times
        dur = int(1 / dt)  # duration of step in ms
        for i in range(1, nStepsUp * 2 + 1):
            idx_start = int(stepLength * i / dt) - dur
            idx_end = int(stepLength * i / dt)
            if idx_end <= len(I):
                I[idx_start:idx_end] = Ival
                if i % 2 == 1:
                    neg_idx_start = idx_start - 25
                    neg_idx_end = idx_start - 25 + dur
                    if neg_idx_start >= 0 and neg_idx_end <= len(I):
                        I[neg_idx_start:neg_idx_end] = -Ival

    elif cellType in [16, 21]:
        if cellType == 16:
            pulsePolarity = 1
        elif cellType == 21:
            pulsePolarity = -1

        stepLength = 50
        nStepsUp = int(np.floor(T / stepLength))
        I = I - 65
        pulseDir = 2
        delay = -3
        for i in range(1, nStepsUp + 1):
            if i % 2 == 1:
                idx = (t > (stepLength + stepLength * (i - 1))) & \
                      (t < (stepLength + stepLength * (i - 1) + pulseDir))
                I[idx] = I[idx] + Ival
            else:
                idx = (t > (delay + stepLength + stepLength * (i - 1))) & \
                      (t < (delay + stepLength + stepLength * (i - 1) + pulseDir))
                I[idx] = I[idx] + Ival * pulsePolarity

    elif cellType == 18:
        baseline = -70
        I = baseline * np.ones_like(I)
        for i in range(1, nStepsUp + 1):
            if i % 2 == 1:
                idx = (t > (stepLength + stepLength * 2 * (i - 1))) & (t < (stepLength * 2 * i + 1))
                n_pts = np.sum(idx)
                if n_pts > 0:
                    I[idx] = np.linspace(baseline, baseline + Ival, n_pts)
            else:
                idx = (t > (stepLength * 1.9 + stepLength * 2 * (i - 1))) & (t < (stepLength * 2 * i + 1))
                n_pts = np.sum(idx)
                if n_pts > 0:
                    I[idx] = np.linspace(baseline, baseline + Ival, n_pts)

    I = I[:len(t)]

    return I, dt
