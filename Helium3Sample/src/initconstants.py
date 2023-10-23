import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

"""
For He-3,
    energy coefficient:        lamb = hbar^2/(2m) = 8.0418 [A^2 K]
    density of the system:     dens = n/(L^3) = 0.016355 [A^-3]
    length of the box:         L = (n/dens)^(1/3) [A] = 9.4950 [A]
    volume of the box:         V = L^3 [A^3]
    Fermi temperature:         EF = hbar^2/(2m) (3 pi^2 dens)^(2/3) = 4.959 [K]
    dimensionless temperature: Theta = T/TF [dimensionless]
    average distance:          rs = (3/4*pi*dens)^(1/3) = 2.4439 [A]
    inverse temperature:       beta = 1/T [K^-1]
    energy unit:               [K]
"""

def initconstats(T, n, dens):
    hbar = 6.62607015e-34 / (2 * jnp.pi)
    kb = 1.380649e-23
    mHe3 = 3.0160293 * 1.66053906660e-27
    lamb = hbar**2 / (2 * mHe3 * kb) * (10**20)
    beta = 1 / T
    L = (n/dens)**(1/3)
    Volume = L**3
    EF = lamb * (3 * (jnp.pi**2) * dens)**(2/3)
    Theta = T / EF
    return lamb, beta, L, Volume, EF, Theta
