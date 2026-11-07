import numpy as np
import matplotlib.pyplot as plt


def f(x):
    print(x)
    return np.exp(-1 / x) * (x > 0)  # Define function only for x > 0


# Define x domain avoiding singularity
x_min, x_max, N = -2, 2, 1400
x = np.linspace(x_min, x_max, N)
x = x[x != 0]

# Sample function on a uniform grid
f_values = f(x)

# Compute FFT
fft_vals = np.fft.fft(f_values, norm='ortho')
k = np.fft.fftfreq(len(f_values), d=(x[1] - x[0])) * 2 * np.pi

# Sort frequencies for plotting
k_sorted_indices = np.argsort(k)
k = k[k_sorted_indices]
fft_vals = fft_vals[k_sorted_indices]

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k, fft_vals.real, label='Re(FFT)')
plt.xlabel('Frequency (k)')
plt.ylabel('Real Part')
plt.title('Real Part of FFT')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(k, fft_vals.imag, label='Im(FFT)', color='r')
plt.xlabel('Frequency (k)')
plt.ylabel('Imaginary Part')
plt.title('Imaginary Part of FFT')
plt.legend()

plt.tight_layout()
plt.show()
