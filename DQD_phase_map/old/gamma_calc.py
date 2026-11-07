import numpy as np
import matplotlib.pyplot as plt

kappa = 0.01
kappa_2 = kappa * kappa
D = np.array([0.5, 0.5])
# hbar = e = 1

U = 215
eVm = eVm = 0.35 * U
Energy_1 = np.linspace(-2, 0, 100) * U + eVm

gamma = []


def gammacalc(eR_):
    gamma0 = 0
    for D_m in D:
        # gamma ~ -dJ/dt
        gamma0 += 8 * (np.pi ** 2) * kappa_2 * D_m * (1 - D_m) * np.abs(eR_)
    return gamma0


def gammano(eR_, kappa_2_, D_):
    gamma0 = 0
    for D_m in D_:
        # gamma ~ -dJ/dt
        gamma0 += 8 * (np.pi ** 2) * kappa_2_ * D_m * (1 - D_m) * np.abs(eR_)
    return gamma0


# for eR in Energy_1:
#     gamma += [gammacalc(eR)]
# # plt.plot(Energy_1, gamma)
# # plt.show()
