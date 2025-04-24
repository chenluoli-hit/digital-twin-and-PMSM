import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq

# File paths for interpolated data
file_paths = {
    'base': '1.0kW/testdata_csv/1000W_fault_1.csv',
    'akima': '1.0kW/testdata_csv/1000W_fault_1_akima.csv',
    'cubic': '1.0kW/testdata_csv/1000W_fault_1_cubic.csv',
    'linear': '1.0kW/testdata_csv/1000W_fault_1_liner.csv',
    'pchip': '1.0kW/testdata_csv/1000W_fault_1_pchip.csv',
    'savgol': '1.0kW/testdata_csv/1000W_fault_1_savgol.csv'
}

# Sampling rate after interpolation
fs = 20000  # Hz

# Fundamental frequency (adjust based on your motor)
f_base = 200# Hz

# Load data from CSV files
data_dict = {}
for method, path in file_paths.items():
    df = pd.read_csv(path)
    data_dict[method] = df['cDAQ1Mod2/ai0'].values  # Adjust column name if needed

# Compute FFT
def compute_fft(signal, fs):
    N = len(signal)
    freq = fftfreq(N, 1/fs)
    fft_values = fft(signal)
    positive_freq_mask = freq > 0
    freq = freq[positive_freq_mask]
    fft_values = np.abs(fft_values[positive_freq_mask]) / N  # Normalize amplitude
    return freq, fft_values

fft_results = {method: compute_fft(signal, fs) for method, signal in data_dict.items()}

# Extract harmonic amplitudes
def extract_harmonics(freq, fft_values, f_base, harmonics=[1, 3, 5, 7]):
    harmonic_values = {}
    for h in harmonics:
        idx = np.argmin(np.abs(freq - h * f_base))
        harmonic_values[h] = fft_values[idx]
    return harmonic_values

harmonic_results = {method: extract_harmonics(freq, fft_values, f_base)
                    for method, (freq, fft_values) in fft_results.items()}

# Compute THD
def compute_thd(harmonic_values, max_harmonic=10):
    A1 = harmonic_values[1]
    sum_squares = sum([harmonic_values.get(h, 0)**2 for h in range(2, max_harmonic+1)])
    thd = np.sqrt(sum_squares) / A1
    return thd

thd_results = {method: compute_thd(harmonic_values) for method, harmonic_values in harmonic_results.items()}

# Extract frequency band energy
def extract_band_energy(freq, fft_values, f_center, bandwidth=5):
    f_low = f_center - bandwidth / 2
    f_high = f_center + bandwidth / 2
    band_mask = (freq >= f_low) & (freq <= f_high)
    energy = np.sum(fft_values[band_mask]**2)
    return energy

energy_results = {method: extract_band_energy(freq, fft_values, f_base)
                  for method, (freq, fft_values) in fft_results.items()}

# Compile results into a DataFrame
results = pd.DataFrame({
    'Method': list(file_paths.keys()),
    'Fundamental Amplitude': [harmonic_results[m][1] for m in file_paths.keys()],
    '3rd Harmonic Amplitude': [harmonic_results[m][3] for m in file_paths.keys()],
    '5th Harmonic Amplitude': [harmonic_results[m][5] for m in file_paths.keys()],
    '7th Harmonic Amplitude': [harmonic_results[m][7] for m in file_paths.keys()],
    'THD': [thd_results[m] for m in file_paths.keys()],
    'Fundamental Band Energy': [energy_results[m] for m in file_paths.keys()]
})

# Display results
print(results)

# Save results to CSV (optional)
results.to_csv('interpolation_analysis_results.csv', index=False)