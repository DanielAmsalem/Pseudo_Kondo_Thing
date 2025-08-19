import math
import numpy
import numpy as np
from scipy.special import digamma
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.cm as cm

"""
this file describes a DQQ with hamiltonian
H = Hbaths + Ht + (tR*nL*dL^+*ckL + tL*nR*dR^+*ckR + h.c.)
Ht = vL*dL^+*ckL + vR*dR^+*ckR + h.c.

since i can't go gamma =2pi*rho v^2
we just wewrite hamiltonian as with v(gamma) and devide out constant sqrt(2pi*rho)
"""

# consts
kB = 86.1733  # [ueV/K]

# [K]=[ueV]/kB -> U[K] = U[ueV]/kB
U = 215
T = 0.026 * U / kB  # K
width = 0.001  # controls width of straight line
GammaL = 1 * kB * T * width
GammaR1 = 1 * kB * T * width  # in
GammaR2 = 3 * kB * T * width  # OUT

# gammas due to P(E) : right site only (0,0)<->(0,1) & (1,0)<->(1,1)
gamma01 = 0 * T
gamma23 = 0 * T

# gammas due to P(E) : left site only (0,0)<->(1,0) & (0,1)<->(1,1)
gamma02 = 0 * T
gamma13 = 0 * T

# V_bias = 0.35U
eVm = 0.35 * U

# voltage on right dot
V0 = U / 10
Vset = [20*V0,15*V0 ,10*V0,5*V0, 0]  # 0.2*V0, 0.5*V0, 0.7*V0

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
    return 0.5 - numpy.imag(digamma(0.5 + (gamma + epsilon * 1j) / (2 * numpy.pi * kB * T))) / np.pi


def right_in_out(eps, gamma_level, G1, G2, mu):
    fin = Modified_Fermi_Function(gamma_level + G1 + G2, eps + mu / 2) * G1 + Modified_Fermi_Function(gamma_level + G1 + G2, eps - mu / 2) * G2
    fout = (1.0 - Modified_Fermi_Function(gamma_level + G1 + G2, eps + mu / 2)) * G1 + (1.0 - Modified_Fermi_Function(gamma_level + G1 + G2, eps - mu / 2)) * G2

    return fin, fout

Energy_1 = np.linspace(-2, 0, 100) * U + eVm
Energy_2 = np.linspace(-2, 0, 100) * U + eVm

plt.figure(figsize=(8, 8))

for V in Vset:
    cood = []
    alachson = []
    x = []

    for eR in Energy_1:
        for eL in Energy_2:
            site_1 = eR
            site_2 = eL

            # Configure Gamma_ij
            # (site_2, site_1)
            # (0,0) == 0 ; (0,1) == 1 ; (1,0) == 2 ; (1,1) == 3
            Gammaij_normalized = np.zeros((4, 4))

            # right dot transitions
            fin01, fout10 = right_in_out(site_1, gamma01, GammaR1, GammaR2, V)
            fin23, fout32 = right_in_out(site_1 + U, gamma23, GammaR1, GammaR2, V)

            Gammaij_normalized[0][1] = fin01
            Gammaij_normalized[1][0] = fout10
            Gammaij_normalized[2][3] = fin23
            Gammaij_normalized[3][2] = fout32

            # left dot transitions
            Gammaij_normalized[0][2] = Modified_Fermi_Function(gamma02 + GammaL,  site_2) * GammaL
            Gammaij_normalized[2][0] = (1 - Modified_Fermi_Function(gamma02 + GammaL, site_2)) * GammaL

            Gammaij_normalized[1][3] = Modified_Fermi_Function(gamma13 + GammaL, site_2 + U) * GammaL
            Gammaij_normalized[3][1] = (1 - Modified_Fermi_Function(gamma13 + GammaL, site_2 + U)) * GammaL

            # staying rates
            Gammaij_normalized[0][0] = 0 - Gammaij_normalized[1][0] - Gammaij_normalized[2][0]
            Gammaij_normalized[1][1] = 0 - Gammaij_normalized[0][1] - Gammaij_normalized[3][1]
            Gammaij_normalized[2][2] = 0 - Gammaij_normalized[0][2] - Gammaij_normalized[3][2]
            Gammaij_normalized[3][3] = 0 - Gammaij_normalized[1][3] - Gammaij_normalized[2][3]

            Gamma = Gammaij_normalized
            nullGamma = null_space(Gamma)

            if nullGamma.shape[1] != 1:
                print(Gammaij_normalized)
                print(nullGamma)
                raise Exception

            P = nullGamma[:, 0]
            P = abs(P) / sum(abs(P))  # normalize P

            prob = P[3] + P[1]

            cood += [(eR, eL, prob)]

            if eL == eR:
                alachson += [prob]


    cood = np.array(cood)
    x = cood[:, 0]

    cmap = cm.get_cmap('Reds_r')
    norm = plt.Normalize(min(Vset), max(Vset) + U)
    plt.scatter(np.unique(x), alachson, color=cmap(norm(V)), label="V = " + str(round(V / U, 2)) + "*U")

plt.title("Right dot population along $\epsilon_{R}$ = $\epsilon_{L}$\n"
          + "GammaL = " + str(rounder(GammaL / (kB * T))) + "kBT" +
          ", GammaR1 = " + str(rounder(GammaR1 / (kB * T))) +
          "kBT, GammaR2 = " + str(rounder(GammaR2 / (kB * T))) + "kBT\n")
plt.xlabel("$\epsilon_{R}$ = $\epsilon_{L}$ [euV]")
plt.ylabel("Population at right dot")
plt.gca().invert_xaxis()
plt.legend()
plt.show()
