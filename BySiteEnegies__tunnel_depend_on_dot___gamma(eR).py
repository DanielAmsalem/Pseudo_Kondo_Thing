import numpy
import numpy as np
from scipy.special import digamma
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.cm as cm
from gamma_calc import gammacalc

# consts
kB = 86.1733  # [ueV/K]
t = 1
tR = 0  #ratio between GammaDD and GammaDD_R
tL = 1  #ratio between GammaDD and GammaDD_L
tR_2 = tR ** 2
tL_2 = tL ** 2

# [K]=[ueV]/kB -> U[K] = U[ueV]/kB
U = 215
T = 0.026 * U / kB  # K

#right site only (0,0)<->(0,1) & (1,0)<->(1,1)
gamma01 = 1*T
gamma23_set = [1*T]  # [1 * T, 2.2 * T, 5 * T, 10 * T

#left site only (0,0)<->(1,0) & (0,1)<->(1,1)
gamma02 = 0
gamma13 = 0


# V_bias = 0.35U
eVm = 0.35 * U


def Fermi_Function(e):
    global kB, T
    return 1 / (1 + np.exp(e / (kB * T)))


def occupancy_2(e):
    global kB, T
    return Fermi_Function(e) ** 2


def Modified_Fermi_Function(gamma, epsilon):
    global kB, T
    if gamma < 1e-6:
        return Fermi_Function(epsilon)
    return 0.5 + numpy.imag(digamma(0.5 + (gamma - epsilon * 1j) / (2 * numpy.pi * kB * T))) / np.pi


Energy_1 = np.linspace(-2, 0, 100) * U + eVm
Energy_2 = np.linspace(-2, 0, 100) * U + eVm

plt.figure(figsize=(8, 8))

for gamma23 in gamma23_set:
    cood = []
    alachson = []

    for eR in Energy_1:
        for eL in Energy_2:
            site_1 = eR
            site_2 = eL

            n_L_2 = occupancy_2(site_2)
            n_R_2 = occupancy_2(site_1)

            # Configure Gamma_ij
            # (site_2, site_1)
            # (0,0) == 0 ; (0,1) == 1 ; (1,0) == 2 ; (1,1) == 3
            Gammaij_normalized = np.zeros((4, 4))

            # tranfers
            Gammaij_normalized[0][1] = Modified_Fermi_Function(gammacalc(eR), -site_1) * (t**2 + 2 *t* tR * Fermi_Function(-site_2) + tR_2 * occupancy_2(-site_2))
            Gammaij_normalized[1][0] = Modified_Fermi_Function(gammacalc(eR), +site_1) * (t**2 + 2*t*tR * Fermi_Function(-site_2) + tR_2 * occupancy_2(-site_2))

            Gammaij_normalized[2][3] = Modified_Fermi_Function(gammacalc(eL), -site_1 - U) * (t**2 + tR_2 * n_L_2 + 2*t*tR*Fermi_Function(site_1))
            Gammaij_normalized[3][2] = Modified_Fermi_Function(gammacalc(eL), + site_1 + U) * (t**2 + tR_2 * n_L_2 + 2*t*tR*Fermi_Function(site_1))

            Gammaij_normalized[0][2] = Modified_Fermi_Function(gamma02, -site_2) * (t**2 + 2 *t*tL * Fermi_Function(-site_1) + tL_2 * occupancy_2(-site_1))
            Gammaij_normalized[2][0] = Modified_Fermi_Function(gamma01, +site_2) ** (t**2 + 2 *t*tL * Fermi_Function(-site_1) + tL_2 * occupancy_2(-site_1))

            Gammaij_normalized[1][3] = Modified_Fermi_Function(gamma13, -site_2 - U) * (t**2 + tL_2 * n_R_2 + 2 *t*tL * Fermi_Function(site_1))
            Gammaij_normalized[3][1] = Modified_Fermi_Function(gamma13, + site_2 + U) * (t**2 + tL_2 * n_R_2 + 2 *t*tL * Fermi_Function(site_1))

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
            prob = 3 * P[3] + 2 * P[2] - 2 * P[1] - 3 * P[0]

            cood += [(eR, eL, prob)]

            if eL == eR:
                alachson += [prob]

    cood = np.array(cood)
    x = cood[:, 0]
    y = cood[:, 1]
    Prob = cood[:, 2]

    Plot2D = False

    if Plot2D:
        grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]

        grid_h = griddata((x, y), Prob, (grid_x, grid_y), method='cubic')

        cmap = plt.get_cmap("coolwarm")
        plt.figure(figsize=(8, 8))
        plt.pcolormesh(grid_x, grid_y, grid_h, cmap=cmap)
        plt.colorbar(label="$\Delta$")

        plt.xlabel("$\epsilon_{R}$ [euV]")
        plt.ylabel("$\epsilon_{L}$ [euV]")
        plt.title("Charge imbalance, Vbias = " + str(int(eVm)) + "ueV, g01=" + str(gamma01 / T) + "T, g23=" + str(
            gamma23 / T) + "T")

        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.show()
        exit()
    else:
        cmap = cm.get_cmap('Reds_r')
        norm = plt.Normalize(min(gamma23_set), max(gamma23_set) + 8 * T)
        plt.scatter(np.unique(x), alachson, color=cmap(norm(gamma23)), label="gamma = " + str(gamma23 / T) + " T")

plt.title("Charge imbalance, Vbias = " + str(int(eVm)) + "ueV, g01=" + str(gamma01 / T) + "T\n" + "t = "+ str(t) +" t_L = " + str(
    tL) + ", t_R = " + str(tR))
plt.xlabel("$\epsilon_{R}$ = $\epsilon_{R}$ [euV]")
plt.ylabel("$\Delta$")
plt.gca().invert_xaxis()
plt.legend()
plt.show()
