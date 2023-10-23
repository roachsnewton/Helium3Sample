import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

####################################################################################
def fun_asymptotic(r):
    """
        Asymptotic behavior of the two-particle wavefunction: 
            f(r) = exp[ -(1/2) * u(r) ], 
            u(r) = (b*sigma/r)^m * [1 + (a*r/sigma)*exp(-r/sigma)]
            with: m = 5, -0.5<a<0, 1<b<1.5.
    """
    b = 1.36
    sigma, eps, m = 2.57, 1e-3, 5.0
    ur = (b * sigma / (r + eps))**m
    log_fr = -0.5 * ur
    return log_fr

def fun_asymptotic_ML(r):
    rlin = 1.8
    qrlin, grad_qrlin = -13.7645367212813, 38.21359445108634
    V = jnp.where(r < rlin, qrlin + grad_qrlin * (r - rlin), fun_asymptotic(r))
    return V

fun_asymptotic_v = jax.vmap(fun_asymptotic_ML, in_axes=(0))

####################################################################################
def fun_objective0(x, L):
    n, _ = x.shape
    i, j = jnp.triu_indices(n, k=1)
    rij = (x[:, None, :] - x)[i, j]
    rij = rij - L*jnp.rint(rij/L)
    dij = jnp.linalg.norm(rij, axis=-1)
    logq = fun_asymptotic_v(dij).sum()
    return logq

from functools import partial
@partial(jax.vmap, in_axes=(0, None), out_axes=0)
def make_logq0(x, L):
    return fun_objective0(x, L)

make_logq = jax.jit(make_logq0)
