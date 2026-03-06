"""
simulate_izhikevich.py - Faithful translation of simulate_izhikevich.m

This code generates data from an Izhikevich neuron of a type specified
by the user.
"""

import numpy as np
import os


def simulate_izhikevich(cellType, I, dt, jitter=None, plotFlag=None, saveFlag=None, fid=None):
    """
    [v, u, spikes, cid] = simulate_izhikevich(cellType, I, dt, jitter, plotFlag, saveFlag, fid)

    This code generates data from an Izhikevich neuron of a type specified
    by the user and saves it to a .mat file in the specified directory.

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
    I : ndarray
        Stimulus (current input)
    dt : float
        Time step, in ms
    jitter : float, optional
        Add jitter to spike times, uniformly distributed over [-jitter, jitter],
        measured in ms (default: 0)
    plotFlag : int, optional
        0 or 1, plot simulated data (default: 1)
    saveFlag : int, optional
        0 or 1, save simulated data to file (default: 1)
    fid : str, optional
        Root directory for project (default: current directory)

    Returns:
    --------
    v : ndarray
        Voltage response of the neuron
    u : ndarray
        Membrane recovery variable
    spikes : ndarray
        Vector of 0s and 1s indicating spikes
    cid : str
        String that identifies cell type
    """
    I = np.asarray(I).flatten()

    # set defaults
    if jitter is None:
        jitter = 0

    if plotFlag is None:
        plotFlag = 1

    if saveFlag is None:
        saveFlag = 1

    if fid is None:
        fid = os.getcwd()

    # parameters
    # numbered as in Izhikevich 2004
    # a-d values taken from code published by Eugene Izhikevich

    #        a         b       c      d
    pars = np.array([
        [0.02,      0.2,    -65,     6],      # 1. tonic spiking
        [0.02,      0.25,   -65,     6],      # 2. phasic spiking
        [0.02,      0.2,    -50,     2],      # 3. tonic bursting
        [0.02,      0.25,   -55,     0.05],   # 4. phasic bursting
        [0.02,      0.2,    -55,     4],      # 5. mixed mode
        [0.01,      0.2,    -65,     5],      # 6. spike frequency adaptation
        [0.02,      -0.1,   -55,     6],      # 7. Class 1
        [0.2,       0.26,   -65,     0],      # 8. Class 2
        [0.02,      0.2,    -65,     6],      # 9. spike latency
        [0.05,      0.26,   -60,     0],      # 10. subthreshold oscillations
        [0.1,       0.26,   -60,     -1],     # 11. resonator
        [0.02,      -0.1,   -55,     6],      # 12. integrator
        [0.03,      0.25,   -60,     4],      # 13. rebound spike
        [0.03,      0.25,   -52,     0],      # 14. rebound burst
        [0.03,      0.25,   -60,     4],      # 15. threshold variability
        [1,         1.5,    -60,     0],      # 16. bistability
        [1,         0.2,    -60,     -21],    # 17. depolarizing after-potential
        [0.02,      1,      -55,     4],      # 18. accomodation
        [-0.02,     -1,     -60,     8],      # 19. inhibition-induced spiking
        [-0.026,    -1,     -45,     0],      # 20. inhibition-induced bursting
        [1,         1.5,    -60,     0]       # 21. bistability 2
    ])

    cids = ['RS', 'PS', 'TB', 'PB', 'MM', 'FA', 'E1', 'E2', 'SL', 'SO',
            'R', 'I', 'ES', 'EB', 'TV', 'B', 'DA', 'A', 'IS', 'IB', 'B2']

    a = pars[cellType - 1, 0]
    b = pars[cellType - 1, 1]
    c = pars[cellType - 1, 2]
    d = pars[cellType - 1, 3]
    cid = cids[cellType - 1]

    T = len(I) * dt
    t = np.arange(dt, T + dt, dt)

    # initialize variables
    threshold = 30

    v = np.zeros(len(t))
    u = np.zeros(len(t))
    spikes = np.zeros(len(t))

    # different initial v and u values to start different neuron types near
    # stable fixed point (prevent spiking in absence of inputs near t=0)
    if cellType in [16, 21]:  # if bistable
        v[0] = -54
        u[0] = -77
    elif cellType == 12:  # integrator
        v[0] = -90
        u[0] = 0
    elif cellType in [19, 20]:  # inhibition-induced spiking/bursting
        v[0] = -100
        u[0] = 80
    else:
        v[0] = -70
        u[0] = -14

    # Izhikevich model doesn't show this kind of bistability, so simulate
    # responses using first form of bistability
    Iplot = I.copy()
    if cellType == 21:
        I = np.abs(I + 65) - 65

    # run model
    for tt in range(len(I) - 1):
        dvdt = 0.04 * v[tt] ** 2 + 5 * v[tt] + 140 - u[tt] + I[tt]
        v[tt + 1] = v[tt] + dvdt * dt

        dudt = a * (b * v[tt + 1] - u[tt])
        u[tt + 1] = u[tt] + dudt * dt

        if v[tt + 1] > threshold:
            v[tt] = threshold  # makes spikes of uniform height
            v[tt + 1] = c
            u[tt + 1] = u[tt + 1] + d
            spikes[tt + 1] = 1

    # if jitter != 0, add noise to spike times
    if jitter:
        spikeIdx = np.where(spikes)[0]
        jitters = np.round((np.random.rand(len(spikeIdx)) - 0.5) * 2 * jitter / dt).astype(int)
        spikeIdx = spikeIdx + jitters
        # Ensure indices are within bounds
        spikeIdx = np.clip(spikeIdx, 0, len(spikes) - 1)
        spikes = np.zeros_like(spikes)
        spikes[spikeIdx] = 1

    # plot results
    if plotFlag:
        import matplotlib.pyplot as plt

        I_for_plot = Iplot
        sTimes = t[spikes.astype(bool)]

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        ax = axes[0]
        ax.plot(t, I_for_plot)
        ax.set_ylabel('current')
        ax.set_ylim([I_for_plot.min() - 0.05 * np.abs(I_for_plot.min()),
                     I_for_plot.max() + 0.05 * np.abs(I_for_plot.max())])
        ax.set_xlim([dt, T])
        ax.set_title('stimulus')

        ax = axes[1]
        ax.plot(t, v)
        for s in range(len(sTimes)):
            ax.plot([sTimes[s], sTimes[s]], [threshold * 1.05, threshold * 1.05 + 0.2 * threshold], 'k')
        ax.set_xlim([dt, T])
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('voltage')
        ax.set_title('response')

        plt.tight_layout()
        plt.close()

    # save stimulus/response
    if saveFlag:
        data_dir = os.path.join(fid, 'izhikevich_data')
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        save_path = os.path.join(data_dir, f'{cid}_iz.npz')
        np.savez(save_path,
                 cellType=cellType, cid=cid, dt=dt, T=T,
                 a=a, b=b, c=c, d=d, threshold=threshold,
                 I=Iplot, u=u, v=v, spikes=spikes)
        print(f'saved: {save_path}')

    return v, u, spikes, cid
