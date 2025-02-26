import numpy
import numpy as np
from scipy.special import digamma
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# consts
kB = 86.1733  # [ueV/K]

# [K]=[ueV]/kB -> U[K] = U[ueV]/kB
U = 215
alpha = 2.2
T = 0.026 * U / kB  # K
gamma01 = 0.01*T
gamma23 = 0.01*T

# V_bias = 0.35U
eVm = 0.35*U


def Fermi_Function(e):
    global kB, T
    return 1 / (1 + np.exp(e / (kB * T)))


def Modified_Fermi_Function(gamma, epsilon):
    global kB, T, k
    if gamma < 1e-10:
        return Fermi_Function(epsilon)
    return 0.5 + numpy.imag(digamma(0.5 + (gamma - epsilon * 1j) / (2 * numpy.pi * kB * T)))/np.pi


Energy_1 = np.linspace(-2, 0, 100) * U + eVm
Energy_2 = np.linspace(-2, 0, 100) * U + eVm

cood = []

for eR in Energy_1:
    for eL in Energy_2:
        site_1 = eR
        site_2 = eL

        # Configure Gamma_ij
        # (site_2, site_1)
        # (0,0) == 0 ; (0,1) == 1 ; (1,0) == 2 ; (1,1) == 3
        Gammaij_normalized = np.zeros((4, 4))

        # tranfers
        Gammaij_normalized[0][1] = Modified_Fermi_Function(gamma01, -site_1)
        Gammaij_normalized[1][0] = Modified_Fermi_Function(gamma01, site_1)

        Gammaij_normalized[2][3] = Modified_Fermi_Function(gamma23, - site_2 - U)
        Gammaij_normalized[3][2] = Modified_Fermi_Function(gamma23, site_2 + U)

        Gammaij_normalized[0][2] = Fermi_Function(-site_2)
        Gammaij_normalized[2][0] = Fermi_Function(site_2)

        Gammaij_normalized[1][3] = Fermi_Function(- site_1 - U)
        Gammaij_normalized[3][1] = Fermi_Function(site_1 + U)

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
        d = 2 * P[2] - 2 * P[1] + 3 * P[3] - 3 * P[0]
        cood += [(eR, eL, d)]


cood = np.array(cood)
x = cood[:, 0]
y = cood[:, 1]
Delta = cood[:, 2]

grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]

grid_h = griddata((x, y), Delta, (grid_x, grid_y), method='cubic')

cmap = plt.get_cmap("coolwarm")
plt.figure(figsize=(7, 7))
plt.pcolormesh(grid_x, grid_y, grid_h, cmap=cmap)
plt.colorbar(label="$\Delta$")

plt.xlabel("e_right [ueV]")
plt.ylabel("e_left [ueV]")
plt.title("Charge difference at Vbias = " + str(int(eVm)) + "ueV, g01 = g23 = 0.01T")

plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.show()
