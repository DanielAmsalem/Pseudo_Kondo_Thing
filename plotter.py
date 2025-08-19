import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma


def fermi(x):
    """Fermi function f(x) = 1 / (1 + exp(x))"""
    return 1.0 / (1.0 + np.exp(x))

def Fermi_Function(e):
    return 1 / (1 + np.exp(e))


def Modified_Fermi_Function(gamma, epsilon):
    return 0.5 + np.imag(digamma(0.5 + (gamma - epsilon * 1j) / (2 * np.pi))) / np.pi


def y(x, v, g1, g2):
    """Weighted average of shifted Fermi functions"""
    return (g1 * Modified_Fermi_Function(g1,x + v) + g2 * Modified_Fermi_Function(g2,x)) / (g1 + g2)


def plot_y_for_vs(v_list, g1, g2, xmin=-10, xmax=10, num_points=1000):
    x = np.linspace(xmin, xmax, num_points)
    plt.figure(figsize=(8, 5))

    for v in v_list:
        y_vals = y(x, v, g1, g2)
        plt.plot(x, y_vals, label=f"v = {v}")

    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title(f"Weighted Modified Fermi function for g1={g1}, g2={g2}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# Example usage: plot for multiple v's
plot_y_for_vs(v_list=[0, 1, 3, 5, 20], g1=0.01, g2=0.03)
