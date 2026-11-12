import numpy as np
import pandas as pd

from src.sSNOM_spectroscopy import load, plot, transforms


class THzWaveform:
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata

    @classmethod
    def from_file(cls, filename):
        # Load the data and metadata from the file
        data, metadata = load.load_neaspec_file(filename)

        # convert amp and phase into complex representation
        data = transforms.amp_phase_to_complex(data)

        # interpolate the data to a new M array
        data = transforms.interpolate_M(data)

        # average the data based on position and M
        data = transforms.average(data, groupby_columns=['Row', 'Column', 'M'])

        # convert M to t
        data = transforms.M_to_t(data)

        return cls(data, metadata)

    def plot(self, ax=None, harmonics=None, **kwargs):
        return plot.plot_waveform(self, ax=ax, harmonics=harmonics, **kwargs)

    def apodise(self, window_type='tukey', alpha=None):
        """
        Apodise the waveform data using a specified window function.

        Parameters:
        window_type (str): Window function to apply. Default is 'tukey'.

        Returns:
        THzWaveform: A new THzWaveform instance with apodised data.
        window: The window function applied to the data.
        """
        apodised_data, window = transforms.apodise(self.data,
                                                   window_type=window_type,
                                                   alpha=alpha)
        return THzWaveform(apodised_data, self.metadata), window

    def FFT(self):
        """
        Compute the Fast Fourier Transform (FFT) of the waveform data.

        Returns:
        THzSpectrum: A new THzSpectrum instance containing the FFT data.
        """
        fft_data = transforms.FFT(self.data)
        return THzSpectrum(fft_data, self.metadata)


class THzWaveforms:
    """ For line scans or 2D scans, where each position has a waveform. """
    def __init__(self, waveforms, metadata):
        self.waveforms = waveforms
        self.metadata = metadata

    @classmethod
    def from_file(cls, filename):
        # Load the data and metadata from the file
        data, metadata = load.load_neaspec_file(filename)

        # convert amp and phase into complex representation
        data = transforms.amp_phase_to_complex(data)

        # interpolate the data to a new M array
        data = transforms.interpolate_M(data)

        # average the data based on position and M
        data = transforms.average(data, groupby_columns=['Row', 'Column', 'M'])

        # convert M to t
        data = transforms.M_to_t(data)

        # group by position and create separate THzWaveform objects for each
        waveforms = {}
        for position, group in data.groupby(['Row', 'Column']):
            waveforms[(position[0], position[1])] = THzWaveform(group,
                                                                metadata)
        return cls(waveforms, metadata)

    def apodise(self, window_type='tukey', alpha=None):
        """
        Apodise the waveform data for all waveforms using a specified window
        function.

        Parameters:
        window_type (str): Type of window function to apply. Default is
        'tukey'.

        Returns:
        THzWaveforms: A new THzWaveforms instance with apodised data.
        windows: A dictionary containing the window functions applied to each
        waveform.
        """
        apodised_waveforms = {}
        windows = {}
        for position, waveform in self.waveforms.items():
            apodised_waveform, window = waveform.apodise(
                                        window_type=window_type,
                                        alpha=alpha)
            apodised_waveforms[position] = apodised_waveform
            windows[position] = window
        return THzWaveforms(apodised_waveforms, self.metadata), windows

    def FFT(self):
        """
        Compute the Fast Fourier Transform (FFT) of the waveform data for all
        waveforms.

        Returns:
        THzSpectra: A new THzSpectra instance containing the FFT data.
        """
        fft_spectra = {}
        for position, waveform in self.waveforms.items():
            fft_spectrum = waveform.FFT()
            fft_spectra[position] = fft_spectrum
        return THzSpectra(fft_spectra, self.metadata)


class THzSpectrum:
    def __init__(self, data, metadata=None):
        self.data = data
        self.metadata = metadata

    @classmethod
    def from_file(cls, filename):
        # Load the data and metadata from the file
        data, metadata = load.load_neaspec_file(filename)

        # convert amp and phase into complex representation
        data = transforms.amp_phase_to_complex(data)

        return cls(data, metadata)

    def plot(self, ax=None, harmonics=None, frequency_range=None, **kwargs):
        return plot.plot_spectrum(self,
                                  ax=ax,
                                  harmonics=harmonics,
                                  frequency_range=frequency_range,
                                  **kwargs)

    def normalise(self, reference: 'THzSpectrum'):
        normalised_data = pd.DataFrame({
            "Frequency": self.data["Frequency"],
            **{f"O{n}": (self.data[f"O{n}"]
                         / reference.data[f"O{n}"]) for n in range(6)}
            })

        return THzSpectrum(normalised_data)

    @property
    def amplitude(self):
        return self.data.filter(like='O').apply(lambda x: np.abs(x))

    @property
    def unwrapped_phase(self):
        return (self.data.filter(like='O')
                .apply(lambda x: np.unwrap(np.angle(x))))

    def corrected_phase(self, fit_range):
        """
        Correct the phase by fitting a linear function to the specified
        frequency range and subtracting the offset.

        Parameters:
        fit_range (tuple): A tuple specifying the frequency range for fitting
        (min_freq, max_freq).

        Returns:
        pd.DataFrame: A DataFrame containing the corrected phase values.
        """
        min_freq, max_freq = fit_range
        corrected_phase_data = pd.DataFrame({})

        for n in range(6):
            # Select the data within the specified frequency range
            mask = ((self.data["Frequency"] >= min_freq)
                    & (self.data["Frequency"] <= max_freq))
            freq_subset = self.data.loc[mask, "Frequency"]
            phase_subset = self.unwrapped_phase.loc[mask, f"O{n}"]

            # Fit a linear function to the phase data
            coeffs = np.polyfit(freq_subset, phase_subset, 1)

            # Subtract the linear fit from the unwrapped phase
            corrected_phase_data[f"O{n}"] = (self.unwrapped_phase[f"O{n}"]
                                             - coeffs[1])

        return corrected_phase_data


class THzSpectra:
    """ For line scans or 2D scans, where each position has a spectrum. """
    def __init__(self, spectra, metadata):
        self.spectra = spectra
        self.metadata = metadata

    @classmethod
    def from_file(cls, filename):
        # Load the data and metadata from the file
        data, metadata = load.load_neaspec_file(filename)

        # convert amp and phase into complex representation
        data = transforms.amp_phase_to_complex(data)

        # group by position and create separate THzSpectrum objects for each.
        spectra = {}
        for position, group in data.groupby(['Row', 'Column']):
            spectra[(position[0], position[1])] = THzSpectrum(group, metadata)
        return cls(spectra, metadata)

    def normalise(self, reference: 'THzSpectrum'):
        normalised_spectra = {}
        for position, spectrum in self.spectra.items():
            normalised_spectra[position] = spectrum.normalise(reference)
        return THzSpectra(normalised_spectra, self.metadata)

    def corrected_phase(self, fit_range):
        corrected_phase_spectra = {}
        for position, spectrum in self.spectra.items():
            corrected_phase_spectra[position] = spectrum.corrected_phase(
                fit_range)
        return corrected_phase_spectra
