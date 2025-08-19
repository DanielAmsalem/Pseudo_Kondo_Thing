import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp


def fermi_dirac(eps):
    return 1 / (1 + np.exp(eps))


def modified_fd(eps, gamma, kBT=1):
    arg = 0.5 + (gamma - 1j * eps) / (2 * np.pi * kBT)
    return 0.5 + np.imag(sp.digamma(arg))/np.pi


eps_values = np.linspace(-5, 5, 500)


fd_values = fermi_dirac(eps_values)
mod_fd_values = modified_fd(eps_values, gamma=0.02)


plt.figure(figsize=(8, 5))
plt.plot(eps_values, fd_values, label="Fermi-Dirac", linewidth=2)
plt.plot(eps_values, mod_fd_values, label="Digamma", linestyle="dashed", linewidth=2)


plt.xlabel(r"$\epsilon$")
plt.ylabel("Value")
plt.title("Fermi-Dirac vs. Modified Distribution")
plt.legend()
plt.grid()

# Show the plot
plt.show()
