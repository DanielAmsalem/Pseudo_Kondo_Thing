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
"""

# consts
kB = 86.1733  # [ueV/K]

# [K]=[ueV]/kB -> U[K] = U[ueV]/kB
U = 215
T = 0.026 * U / kB  # K
vL = 1 * T
vR = 1.3 * T
tR = 0
tL = 0

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
Vset = [0]  # 0.2*V0, 0.5*V0, 0.7*V0


def Fermi_Function(e):
    global kB, T
    return 1 / (1 + np.exp(e / (kB * T)))


def gammaDD(t, nL, t_R):
    global kB, T
    gDDR = t**2 + 2*t*t_R*nL + (nL*t_R)** 2  # units of Kelvin squared
    return kB*gDDR  # should be units of energy


def Modified_Fermi_Function(gamma, epsilon):
    global kB, T
    if gamma < 1e-6:
        return Fermi_Function(epsilon)
    return 0.5 + numpy.imag(digamma(0.5 + (gamma - epsilon * 1j) / (2 * numpy.pi * kB * T))) / np.pi


Energy_1 = np.linspace(-2, 0, 100) * U + eVm
Energy_2 = np.linspace(-2, 0, 100) * U + eVm

plt.figure(figsize=(8, 8))

for V in Vset:
    cood = []
    alachson = []

    for eR in Energy_1:
        for eL in Energy_2:
            site_1 = eR
            site_2 = eL

            # Configure Gamma_ij
            # (site_2, site_1)
            # (0,0) == 0 ; (0,1) == 1 ; (1,0) == 2 ; (1,1) == 3
            Gammaij_normalized = np.zeros((4, 4))

            # right dot transitions
            Gammaij_normalized[0][1] = Modified_Fermi_Function(gamma01 + gammaDD(vR, 0, tR), V/2 - site_1) * gammaDD(vR,0,tR)
            Gammaij_normalized[1][0] = Modified_Fermi_Function(gamma01 + gammaDD(vR, 0, tR), -V/2 + site_1) * gammaDD(vR,0,tR)

            Gammaij_normalized[2][3] = Modified_Fermi_Function(gamma23 + gammaDD(vR, 1, tR), V/2 - site_1 - U) * gammaDD(vR,1,tR)
            Gammaij_normalized[3][2] = Modified_Fermi_Function(gamma23 + gammaDD(vR, 1, tR), -V/2 + site_1 + U) * gammaDD(vR,1,tR)

            # left dot transitions
            Gammaij_normalized[0][2] = Modified_Fermi_Function(gamma02 + gammaDD(vL, 0, tL), - site_2) * gammaDD(vL,0,tL)
            Gammaij_normalized[2][0] = Modified_Fermi_Function(gamma02 + gammaDD(vL, 0, tL), + site_2) * gammaDD(vL,0,tL)

            Gammaij_normalized[1][3] = Modified_Fermi_Function(gamma13 + gammaDD(vL, 1, tL), - site_2 - U) * gammaDD(vL,1,tL)
            Gammaij_normalized[3][1] = Modified_Fermi_Function(gamma13 + gammaDD(vL, 1, tL), + site_2 + U) * gammaDD(vL,1,tL)

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
            # prob = 3 * P[3] + 2 * P[2] - 2 * P[1] - 3 * P[0]
            prob = P[3] + P[1]

            cood += [(eR, eL, prob)]

            if eL == eR:
                alachson += [prob]

    cood = np.array(cood)
    x = cood[:, 0]
    y = cood[:, 1]
    Prob = cood[:, 2]

    cmap = cm.get_cmap('Reds_r')
    norm = plt.Normalize(min(Vset), max(Vset) + 0.03 * U)
    plt.scatter(np.unique(x), alachson, color=cmap(norm(V)), label="V = " + str(round(V / U, 2)) + "*U")

plt.title("Right dot population along $\epsilon_{R}$ = $\epsilon_{L}$\n"
          + "vL = " + str(vL / T) + ", vR = " + str(vR / T) + "\n"
          + "tL = " + str(tL) + ", tR = " + str(tR) + "\n")
plt.xlabel("$\epsilon_{R}$ = $\epsilon_{L}$ [euV]")
plt.ylabel("Population at right dot")
plt.gca().invert_xaxis()
plt.legend()
plt.show()
