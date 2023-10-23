import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import time

from ..sampler_spin import make_autoregressive_sampler_spin

def make_loss(log_prob, Es, beta, lamb):
    
    def loss_fn(params, state_indices):
        # Change the energy unit into [K]
        logp = log_prob(params, state_indices)
        E = Es[state_indices].sum(axis=-1) * lamb
        F = jax.lax.stop_gradient(logp / beta + E)

        E_mean = E.mean()
        F_mean = F.mean()
        S_mean = -logp.mean()
        E_std = E.std()
        F_std = F.std()
        S_std = (-logp).std()

        gradF = (logp * (F - F_mean)).mean()

        auxiliary_data = {"F_mean": F_mean, "F_std": F_std,
                          "E_mean": E_mean, "E_std": E_std,
                          "S_mean": S_mean, "S_std": S_std,
                         }

        return gradF, auxiliary_data

    return loss_fn

def pretrain_spin(van, params_van,
             nup, ndn, dim, Emax, twist,
             beta, L, lamb,
             path, key,
             lr, sr, damping, maxnorm,
             batch, epoch=10000):
    
    n = nup + ndn

    from ..orbitals import sp_orbitals, twist_sort
    sp_indices, _ = sp_orbitals(dim, Emax)
    sp_indices_twist, Es_twist = twist_sort(sp_indices, twist)
    del sp_indices
    sp_indices_twist = jnp.array(sp_indices_twist)[::-1]
    Es_twist = (2*jnp.pi/L)**2 * jnp.array(Es_twist)[::-1]
    
    num_states = Es_twist.size
    sampler, log_prob_novmap = make_autoregressive_sampler_spin(van, 
                                sp_indices_twist, nup, ndn, num_states)
    log_prob = jax.vmap(log_prob_novmap, (None, 0), 0)

    loss_fn = make_loss(log_prob, Es_twist, beta, lamb)

    import optax
    if sr:
        from ..sampler_spin import make_classical_score
        score_fn = make_classical_score(log_prob_novmap)
        from ..sr import fisher_sr
        optimizer = fisher_sr(score_fn, damping, maxnorm)
        print("Optimizer fisher_sr: damping = %.5f, maxnorm = %.5f." % (damping, maxnorm))
    else:
        optimizer = optax.adam(lr)
        print("Optimizer adam: lr = %.3f." % lr)
    opt_state = optimizer.init(params_van)

    @jax.jit
    def update(params_van, opt_state, key):
        key, subkey = jax.random.split(key)
        state_indices = sampler(params_van, subkey, batch)
        
        grads, aux = jax.grad(loss_fn, argnums=0, has_aux=True)(params_van, state_indices)
        updates, opt_state = optimizer.update(grads, opt_state,
                                params=(params_van, state_indices) if sr else None)
        params_van = optax.apply_updates(params_van, updates)
        
        return params_van, opt_state, key, aux

    import os
    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w", buffering=1, newline="\n")

    for i in range(1, epoch+1):
        t0 = time.time()
        params_van, opt_state, key, aux = update(params_van, opt_state, key)

        # Energy unit: [K]
        F, F_std, E, E_std, S, S_std = aux["F_mean"], aux["F_std"], \
                                       aux["E_mean"], aux["E_std"], \
                                       aux["S_mean"], aux["S_std"]
        
        # Quantities per particle
        F, F_std = F /n, F_std /n /jnp.sqrt(batch)
        E, E_std = E /n, E_std /n /jnp.sqrt(batch)
        S, S_std = S /n, S_std /n /jnp.sqrt(batch)
        
        t1 = time.time()
        dt = t1 - t0
        print("iter: %04d" % i,
                " F: %.6f" % F, "(%.6f)" % F_std,
                " E: %.6f" % E, "(%.6f)" % E_std,
                " S: %.6f" % S, "(%.6f)" % S_std,
                " dt: %.3f" % dt, flush=True)

        f.write( ("%6d" + "  %.6f"*6 + "  %.3f" + "\n") % (i, 
                F, F_std, E, E_std, S, S_std, dt) )

    return params_van