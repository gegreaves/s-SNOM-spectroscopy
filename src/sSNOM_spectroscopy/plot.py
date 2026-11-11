import matplotlib.pyplot as plt
import numpy as np

def plot_spectrum(THzSpectrum, ax=None, harmonics = None, frequency_range = None, fit_range = None, **kwargs):
    """
    Plot the THz spectrum.

    Parameters:
    - THzSpectrum: An instance of the THzSpectrum class.
    - ax: A matplotlib Axes object. If None, a new figure and axes will be created.

    Returns:
    - ax: The matplotlib Axes object containing the plot.
    """

    if ax is None:
        fig, ax = plt.subplots(2,1, sharex=True)
    else:
        fig = ax[0].figure
        ax_amp = ax[0]
        ax_phase = ax[1]

    if harmonics is None:
        harmonics = [0,1,2,3,4,5]
        
    amplitude = THzSpectrum.amplitude

    if fit_range is not None:
        phase = THzSpectrum.corrected_phase(fit_range)
    else:  
        phase = THzSpectrum.unwrapped_phase

    for n in harmonics:
        ax[0].plot(THzSpectrum.data['Frequency'], amplitude[f'O{n}'], label=f'n={n}')
        ax[1].plot(THzSpectrum.data['Frequency'], phase[f'O{n}'], label=f'n={n}')


    ax[0].set_ylabel('Amplitude')
    ax[0].set_title('THz Spectrum')
    ax[1].set_xlabel('Frequency (THz)')
    ax[1].set_ylabel('Phase')
    ax[1].set_title('THz Spectrum - Phase')

    ax[0].legend()

    return fig,ax



def plot_waveform(THzWaveform, ax=None, harmonics=None, **kwargs):
    """
    Plot the THz waveform.

    Parameters:
    - THzWaveform: An instance of the THzWaveform class.
    - ax: A matplotlib Axes object. If None, a new figure and axes will be created.

    Returns:
    - ax: The matplotlib Axes object containing the plot.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if harmonics is None:
            harmonics = [0,1,2,3,4,5]

    for n in harmonics:
        ax.plot(THzWaveform.data['Time (ps)'], np.real(THzWaveform.data[f'O{n}']), label=f'n={n}', **kwargs)
    
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Sampled waveform (a.u.)')
    ax.set_title('THz Waveform')

    ax.legend()

    return fig,ax

def plot_window(window, ax=None, **kwargs):
    """
    Plot the window function.

    Parameters:
    - window: The window function to be plotted.
    - ax: A matplotlib Axes object. If None, a new figure and axes will be created.

    Returns:
    - ax: The matplotlib Axes object containing the plot.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(window['Time (ps)'], window['Window'], **kwargs)
    
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Window Value')
    ax.set_title('Window Function')

    return fig,ax