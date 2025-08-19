import numpy
import numpy as np
from scipy.special import digamma
from scipy.linalg import null_space
import matplotlib.pyplot as plt

"""
this file allows us to find where we are in the eR eL phase space
"""

# consts
kB = 86.1733  # [ueV/K]

# [K]=[ueV]/kB -> U[K] = U[ueV]/kB
U = 215
T = 0.026*U/kB  # K
gamma01 = U*0.084
gamma23 = U*0.084

# V_bias = 0.35U
eVm = 0.35*U


def Fermi_Function(e):
    global kB, T
    return 1 / (1 + np.exp(e / (kB * T)))


def Lorentzian(gamma, epsilon):
    return gamma / (numpy.pi * (epsilon ** 2 + gamma ** 2))


def Modified_Fermi_Function(gamma, epsilon):
    global kB, T
    if gamma < 1e-6:
        return Fermi_Function(epsilon)
    return 0.5 + numpy.imag(digamma(0.5 + (gamma - epsilon * 1j) / (2 * numpy.pi * kB * T)))


site_means = np.linspace(-2, 0, 200)*U
site_diffs = np.linspace(-1, 1, 200)*U

cood = []
k=0

for mean in site_means:
    for diff in site_diffs:
        site_1 = mean + diff
        site_2 = mean - diff

        # Configure Gamma_ij
        # (0,0) == 0 ; (0,1) == 1 ; (1,0) == 2 ; (1,1) == 3
        Gammaij_normalized = np.zeros((4, 4))

        # tranfers
        Gammaij_normalized[0][1] = Modified_Fermi_Function(gamma01, site_1)
        Gammaij_normalized[1][0] = Modified_Fermi_Function(gamma01, -site_1)

        Gammaij_normalized[2][3] = Modified_Fermi_Function(gamma23, site_2)
        Gammaij_normalized[3][2] = Modified_Fermi_Function(gamma23, -site_2)

        Gammaij_normalized[0][2] = Fermi_Function(site_2)
        Gammaij_normalized[2][0] = Fermi_Function(-site_2)

        Gammaij_normalized[1][3] = Fermi_Function(site_1)
        Gammaij_normalized[3][1] = Fermi_Function(-site_1)

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

        cood += [(diff, mean, d)]

cood = np.array(cood)
x = np.unique(cood[:, 0])
y = np.unique(cood[:, 1])

X, Y = np.meshgrid(x, y)

Prob = cood[:, 2].reshape(len(x), len(y))

cmap = plt.get_cmap("coolwarm")
plt.figure(figsize=(6, 6))
plt.scatter(X, Y, c=Prob, cmap=cmap)
plt.colorbar(label="charge difference")

plt.xlabel("Voltage difference [ueV]")
plt.ylabel("Voltage mean [ueV]")
plt.title("Charge difference at Vbias = " + str(int(eVm)) + "ueV")

plt.show()

