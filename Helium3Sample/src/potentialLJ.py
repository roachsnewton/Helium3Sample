import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

####################################################################################
# the 1979 Aziz He-He interaction potential:
def func_potential_Aziz(r):
    """
    1979 Aziz functional form for the He-He interaction potential:
        V(x) = eps * [ A * exp(-alpha*x) - (C6/x^6 + C8/x^8 + C10/x^10) * F(r) ],
    where x = r/rm, energy unit is [K], length unit is [A].
    """
    C6 = 1.3732412
    C8 = 0.4253785
    C10 = 0.178100
    D = 1.241314
    alpha = 13.353384
    eps = 10.8
    A = 0.5448504 * (10**6)
    rm = 2.9673
    x = r/rm
    
    Fx = jnp.where(x < D, jnp.exp(-(D/x - 1)**2), 1.0)
    V = A * jnp.exp(-alpha*x) - (C6/(x**6) + C8/(x**8) + C10/(x**10)) * Fx
    V = V * eps
    return V

grad_potential_Aziz = jax.grad(func_potential_Aziz, argnums=0)

####################################################################################
# the Lennard-Jones 6-12 potential:
def func_potential_LJ(r):
    """
    Lennard-Jones 6-12 potential:
        V(x) = 4*epslion * [ (sigma/r)^12 - (sigma/r)^6 ],
    where epslion = 10.8 [K], sigma = 2.57 [A].
    """
    epsilon = 10.8
    sigma = 2.57
    x = sigma/r
    V = (4*epsilon) * (x**12 - x**6)
    return V

grad_potential_LJ = jax.grad(func_potential_LJ, argnums=0)

####################################################################################
# Artificial potential energy
def func_potential_ML(r, params_potential):
    """
        For Aziz potential:
        rlin = 2.48, rc = L/2
        (1) r < rlin,        V_ML(r) = V(rlin) + V'(rlin)(r-rlin)
        (2) r > rcut,        V_ML(r) = 0
        (3) rlin < r < rcut, V_ML(r) = V(r)
        ****** If rlin < 0.1, do not use the linear approximation. ******
    """
    # rlin, rcut, Vrlin, Vrlin_grad, Vtail = params_potential
    # conditions = [r < rlin, r > rcut]
    # choices    = [Vrlin + Vrlin_grad * (r - rlin), 0]
    # V = jnp.select(conditions, choices, default=func_potential_Aziz(r))
    
    """
        For Lennard-Jones 6-12 potential:
    """
    rlin, rcut, Vrlin, Vrlin_grad, Vtail = params_potential
    conditions = [r > rcut]
    choices = [0]
    V = jnp.select(conditions, choices, default=func_potential_LJ(r))
    return V

func_potential_v = jax.vmap(func_potential_ML, in_axes=(0, None))

####################################################################################
# Potential tail cutoff correction (Aziz)
def fun_Vtail(rc, dens):
    """
    Tail of the 1979 Aziz He-He interaction potential (per particle):
        V(x) = 2*pi*dens*eps*
        [ A*rm/alpha^3 * (2*(rm^2) + 2*rc*rm*alpha + (rc^2)*(alpha^2)) * exp(-rc*alpha/rm)
        - (rm^3) * (C6/3*(rm/rc)^3 + C8/5*(rm/rc)^5 + C10/7*(rm/rc)^7)]],
    where x = rm/r, energy unit is [K], length unit is [A].
    """
    # C6 = 1.3732412
    # C8 = 0.4253785
    # C10 = 0.178100
    # alpha = 13.353384
    # eps = 10.8
    # A = 0.5448504 * (10**6)
    # rm = 2.9673
    # Vtail = (A*rm)/(alpha**3) * (2*(rm**2) + 2*rc*rm*alpha + (rc**2)*(alpha**2)) * jnp.exp(-rc*alpha/rm) \
    #         - (rm**3) * ((C6/3)*(rm/rc)**3 + (C8/5)*(rm/rc)**5 + (C10/7)*(rm/rc)**7)
    # Vtail = 2 * jnp.pi * dens * eps * Vtail
    
    """
    Tail of Lennard-Jones 6-12 potential (per particle):
    """
    epsilon, sigma = 10.8, 2.57
    Vtail = - sigma**6 / (3 * rc**3) + sigma**12 / (9 * rc**9)
    Vtail = 2 * jnp.pi * dens * 4 * epsilon * Vtail
    return Vtail

####################################################################################
def psi(x, L, params_potential):
    """
    The potential energy (per cell) for a periodic system of lattice constant L.
        1/2 \sum_{i}\sum_{j neq i} V(r_i, r_j)
    """
    rlin, rcut, Vrlin, Vrlin_grad, Vtail = params_potential
    n, _ = x.shape
    i, j = jnp.triu_indices(n, k=1)
    rij = (x[:, None, :] - x)[i, j]
    
    # Only the nearest neighbor is taken into account.
    # Change the box length from L to 1, and then return L.
    rij = rij - L*jnp.rint(rij/L)
    
    # Distance of the atoms.
    dij = jnp.linalg.norm(rij, axis=-1)
    potential = func_potential_v(dij, params_potential).sum() + Vtail * n
    return potential

from functools import partial

@partial(jax.vmap, in_axes=(0, None, None), out_axes=0)
def potential_energy0(x, L, params_potential):
    return psi(x, L, params_potential)

potential_energy = jax.jit(potential_energy0)
