import math
import numpy
import numpy as np
from scipy.special import digamma
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.cm as cm

"""
this file describes a 1 QQ with hamiltonian
H = Hbaths + Ht
Ht = v*d^+*ck + h.c.

since i can't go gamma =2pi*rho v^2
we just wewrite hamiltonian as with v(gamma) and devide out constant sqrt(2pi*rho)
"""

# consts
kB = 86.1733  # [ueV/K]

# [K]=[ueV]/kB -> U[K] = U[ueV]/kB
U = 215
T = 0.026 * U / kB  # K
width = 0.001  # controls width of straight line
Gamma1 = 4 * kB * T * width  # IN
Gamma2 = 1 * kB * T * width  # OUT

# gamma due to P(E) : right site only (0)<->(1)
gamma = 0 * T

# V_bias = 0.35U
eVm = 0.35 * U

# voltage on right dot
V0 = U / 10
Vset = np.array([0,1,3,5,10]) * V0  # 0.2*V0, 0.5*V0, 0.7*V0
Vset = Vset[::-1]


def rounder(x):
    if x < 1e-6:
        return 0
    # Find exponent of first non-zero digit
    exponent = math.floor(math.log10(x))

    # Shift, round, shift back
    rounded = round(x / (10 ** exponent)) * (10 ** exponent)

    return rounded


def Fermi_Function(e):
    global kB, T
    return 1 / (1 + np.exp(e / (kB * T)))


def Modified_Fermi_Function(gamma, epsilon):
    global kB, T
    if gamma < 1e-6:
        return Fermi_Function(epsilon)
    return 0.5 + numpy.imag(digamma(0.5 + (gamma - epsilon * 1j) / (2 * numpy.pi * kB * T))) / np.pi


def right_in_out(eps, gamma_level, G1, G2, mu):
    fin = Modified_Fermi_Function(gamma_level + G1 + G2, eps + mu / 2) * G1 + Modified_Fermi_Function(gamma_level + G1 + G2, eps - mu / 2) * G2
    fout = (1.0 - Modified_Fermi_Function(gamma_level + G1 + G2, eps + mu / 2)) * G1 + (1.0 - Modified_Fermi_Function(gamma_level + G1 + G2, eps - mu / 2)) * G2

    return fin, fout


def stationary_distribution(Gamm):
    """
    Compute stationary distribution P for a rate matrix Gamma.
    Gamma[i,j] = rate from i -> j
    """
    n = Gamm.shape[0]

    # Build generator matrix Q
    Q = Gamm.copy().astype(float)
    for i in range(n):
        Q[i, i] = -np.sum(Gamm[i, :])  # diagonal = -outflow

    # Solve PQ=0 subject to sum(P)=1
    # This is equivalent to solving Q^T P^T = 0
    w, v = np.linalg.eig(Q.T)
    nullspace = v[:, np.isclose(w, 0)]
    Ps = np.real(nullspace[:, 0])
    Ps = Ps / np.sum(Ps)  # normalize to 1
    return Ps


Energy_1 = np.linspace(-1, 0.3, 100) * U + eVm

plt.figure(figsize=(8, 8))

for V in Vset:
    cood = []
    alachson = []
    x = []

    for eR in Energy_1:
        site_1 = eR

        # Configure Gamma_ij 2x2
        # (site_1)
        # (n=0) == |0> ; (n=1) == |1>
        Gammaij_normalized = np.zeros((2, 2))

        # right dot transitions
        fin01, fout10 = right_in_out(site_1, gamma, Gamma1, Gamma2, V)

        Gammaij_normalized[0][1] = fin01
        Gammaij_normalized[1][0] = fout10

        P = stationary_distribution(Gammaij_normalized)
        prob = P[1]

        cood += [(eR, prob)]

        alachson += [prob]
        x += [eR]

    cood = np.array(cood)

    cmap = cm.get_cmap('Reds_r')
    norm = plt.Normalize(min(Vset), max(Vset) + U/3)
    plt.scatter(x, alachson, color=cmap(norm(V)), label="V = " + str(round(V / U, 2)) + "*U")

plt.title("Dot population \n" +
          "GammaR1 = " + str(rounder(Gamma1 / (kB * T))) + "kBT\n" +
          "GammaR2 = " + str(rounder(Gamma2 / (kB * T))) + "kBT\n")
plt.xlabel("$\epsilon$[euV]")
plt.ylabel("Population at dot")
plt.gca().invert_xaxis()
plt.legend()
plt.show()
