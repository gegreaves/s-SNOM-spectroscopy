import numpy as np
import pandas as pd
import scipy as sp


def amp_phase_to_complex(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert amplitude and phase to complex representation.

    Parameters:
    amp (array-like): Amplitude values.
    phase (array-like): Phase values in radians.

    Returns:
    complex_array (numpy.ndarray): Complex representation of the input data.
    """
    for n in range(6):
        data[f"O{n}"] = data[f"O{n}A"] * np.exp(1j * data[f"O{n}P"])
        data.drop(columns=[f"O{n}A", f"O{n}P"], inplace=True)
    return data


def interpolate_M(data: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate the data to a new M array.

    Parameters:
    data (pd.DataFrame): Input DataFrame containing the original data.
    Returns:
    """
    M_min = data['M'].min()
    M_max = data['M'].max()
    delta_M = 1e-5
    new_M = np.arange(M_min, M_max, delta_M)

    interpolated_dfs = []

    for (row, column, run), group in data.groupby(['Row', 'Column', 'Run']):
        group = group.sort_values('M')
        interpolated_values = {
            'Row': row,
            'Column': column,
            'Run': run,
            'M': new_M,
        }

        interpolated_values["Z"] = np.interp(new_M, group['M'], group["Z"])
        for n in range(6):
            interpolated_values[f"O{n}"] = np.interp(new_M, group['M'],
                                                     group[f"O{n}"])
        interpolated_df = pd.DataFrame(interpolated_values)
        interpolated_dfs.append(interpolated_df)

    interpolated_data = pd.concat(interpolated_dfs, ignore_index=True)

    return interpolated_data


def average(data: pd.DataFrame, groupby_columns: list) -> pd.DataFrame:
    """
    Average the data based on specified columns.

    Parameters:
    data (pd.DataFrame): Input DataFrame containing the data to be averaged.
    groupby_columns (list): List of column names to group by.

    Returns:
    pd.DataFrame: Averaged DataFrame.
    """
    averaged_data = (
        data.groupby(groupby_columns)
        .mean()
        .reset_index()
        .drop(columns=['Run'])
        )
    return averaged_data


def M_to_t(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert M to t.

    Parameters:
    data (pd.DataFrame): Input DataFrame containing the data.

    Returns:
    pd.DataFrame: DataFrame with M converted to t.
    """
    data['Time (ps)'] = - data['M'] * 2 / 2.998e8 * 1e12 + 405
    # m -> ps
    # x2 for round trip, -ve since recorded backwards,
    # +405 to shift to 0
    # (without possibility of shifting datasets by different amounts)
    data = data.drop(columns=['M'])
    return data


def apodise(data: pd.DataFrame,
            window_type: str = 'tukey',
            alpha: float | None = None) -> pd.DataFrame:
    """
    Apodise the data using a specified window function.

    Parameters:
    data (pd.DataFrame): Input DataFrame containing the data to be apodised.
    window_type (str): Type of window function to use ('hanning', etc.).

    Returns:
    pd.DataFrame: Apodised DataFrame.
    """
    window_spec = (window_type, alpha) if alpha is not None else window_type
    window = sp.signal.windows.get_window(window_spec, len(data))

    data['Z'] *= window

    for n in range(0, 6):
        data[f"O{n}"] *= window

    window = pd.DataFrame({'Window': window, 'Time (ps)': data['Time (ps)']})

    return data, window


def FFT(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform FFT on the data.

    Parameters:
    data (pd.DataFrame): Input DataFrame containing the data to be transformed.

    Returns:
    pd.DataFrame: DataFrame containing the FFT results.
    """
    N = len(data)
    dt = np.mean(np.diff(data['Time (ps)']))
    freq = np.fft.fftfreq(N, d=dt)

    fft_data = {
        'Frequency': freq,
        'Z': data['Z'],
    }

    for n in range(0, 6):
        fft_data[f"O{n}"] = np.fft.fft(data[f"O{n}"])

    mask = freq > 0
    fft_data = {key: value[mask] for key, value in fft_data.items()}

    fft_df = (pd.DataFrame(fft_data)
              .sort_values('Frequency')
              .reset_index(drop=True)
              )

    return fft_df


def average_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Average a list of DataFrames.

    Parameters:
    dfs (list[pd.DataFrame]): List of DataFrames to be averaged.

    Returns:
    pd.DataFrame: Averaged DataFrame.
    """
    combined_df = pd.concat(dfs)
    averaged_df = combined_df.groupby(level=0).mean()
    return averaged_df
