import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import src
import time

####################################################################################
# Helium-3 Finite-Temperature Simulation Program - Version 1.0
# 
# Description:
# This program simulates the behavior of Helium-3 fermions at finite temperature.
# It focuses on systems with total spin zero (sz=0) and an equal number of up and
# down particles (nup = ndn = n/2). The system with n atoms is assumed to be in 
# a cubic box with periodic boundary conditions. The Hamiltonian is given by: 
#     H = -\lambda \sum_{i} \Laplacian^2 + \sum_{i<j} V(r_{ij}),
# where \lambda = \hbar^2/2m, and V(r) is the Aziz potential.
# 
# Author: Qi Zhang
# Date: Sep 27, 2023
#
####################################################################################

print("\njax.__version__:", jax.__version__)
print("Number of GPU devices:", jax.device_count())
import subprocess
print(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True).stdout, flush=True)
key = jax.random.PRNGKey(421)

####################################################################################
import argparse
parser = argparse.ArgumentParser(description="finite-temperature simulation for the helium-3 fermions")

# folder to save data.
parser.add_argument("--folder", default="/data/zhangqidata/Helium3/datasbackflow/", help="the folder to save data")

# physical parameters. spin up and spin down should be equal: nup = ndn = n/2 (total spin sz = 0)
parser.add_argument("--n"  , type=int, default=14, help="total number of atoms")
parser.add_argument("--dim", type=int, default=3, help="spatial dimension")
parser.add_argument("--dens", type=float, default=0.016355, help="density of the system (A^-3)")
parser.add_argument("--T", type=float, default=0.1, help="temperature T in Kelvin (K)")

# twist angle. For dim = 3, twist = [1/4, 1/4, 1/4]
parser.add_argument("--twist", type=float, nargs="+", default=[1/4, 1/4, 1/4], 
        help="(scaled) twist angle in the range [-1/2, 1/2]^dim")

# many-body state distribution: autoregressive transformer.
parser.add_argument("--Emax", type=int, default=27, help="energy cutoff for the single-particle orbitals")

####################################################################################
# Training parameters.
# autoregressive model: default(2, 16, 4, 32)
parser.add_argument("--nlayers", type=int, default=2, help="CausalTransformer: number of layers")
parser.add_argument("--modelsize", type=int, default=8, help="CausalTransformer: embedding dimension")
parser.add_argument("--nheads", type=int, default=4, help="CausalTransformer:number of heads")
parser.add_argument("--nhidden", type=int, default=16, 
                    help="CausalTransformer: number of hidden units of the MLP within each layer")
parser.add_argument("--remat_van", action='store_true', help="remat transformer model")
parser.add_argument("--pre_epoch", type=int, default=1000, help="pretraining epoch")

# potential parameters.
parser.add_argument("--rlin", type=float, default=1.8, help="Aziz potential is linear below rlin (A)")

# normalizing flow.
parser.add_argument("--depth", type=int, default=2, help="FermiNet: network depth")
parser.add_argument("--spsize", type=int, default=32, help="FermiNet: single-particle feature size")
parser.add_argument("--tpsize", type=int, default=16, help="FermiNet: two-particle feature size")
parser.add_argument("--Np", type=int, default=5, help="FermiNet: number of powers")
parser.add_argument("--Nf", type=int, default=5, help="FermiNet: number of fequencies")
parser.add_argument("--remat_flow", action='store_true', help="remat flow model")
# using Jastrow factor in the normalizing flow.
# parser.add_argument("--jastrow", action='store_true', help="use Jastrow factor")
# parser.add_argument("--rjas", type=float, default=2.0, help="Jastrow factor cutoff distance (A)")
# parser.add_argument("--rswitch", type=float, default=0.1, help="switching distance for Jastrow factor (A)")
# MCMC.
parser.add_argument("--mc_therm", type=int, default=10, help="MCMC thermalization steps")
parser.add_argument("--mc_steps", type=int, default=50, help="MCMC update steps")
parser.add_argument("--mc_stddev", type=float, default=0.1, 
                    help="standard deviation of the Gaussian proposal in MCMC update")

# technical miscellaneous
parser.add_argument("--hutchinson", action='store_true',  help="use Hutchinson's trick to compute the laplacian")

# optimizer parameters.
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (valid only for adam)")
parser.add_argument("--sr", action='store_true', help="use the second-order stochastic reconfiguration optimizer")
parser.add_argument("--lr_c", type=float, default=1e-2, help="learning rate classical")
parser.add_argument("--lr_q", type=float, default=1e-2, help="learning rate quantum")
parser.add_argument("--decay", type=float, default=1e-2, help="learing rate decay")
parser.add_argument("--damping_c", type=float, default=1e-3, help="damping classical")
parser.add_argument("--damping_q", type=float, default=1e-3, help="damping quantum")
parser.add_argument("--maxnorm_c", type=float, default=1e-3, help="gradnorm maximum classical")
parser.add_argument("--maxnorm_q", type=float, default=1e-3, help="gradnorm maximum quantum")
parser.add_argument("--clip_factor", type=float, default=5.0, help="clip factor for gradient")

# training parameters.
parser.add_argument("--batch", type=int, default=2048, help="batch size (per single gradient accumulation step)")
parser.add_argument("--num_devices", type=int, default=1, help="number of GPU devices")
parser.add_argument("--acc_steps", type=int, default=4, help="gradient accumulation steps")
parser.add_argument("--epoch_finished", type=int, default=0, help="number of epochs already finished")
parser.add_argument("--epoch", type=int, default=3000, help="final epoch")
parser.add_argument("--mala", action='store_true', help="use MALA instead of MCMC")

parser.add_argument("--save_ckpt", type=int, default=10, help="save checkpoint every save_ckpt epochs")

args = parser.parse_args()

####################################################################################
print("\n========== Spinful Helium-3 ==========")

n, dim, T, dens = args.n, args.dim, args.T, args.dens
nup, ndn = n//2, n//2
if nup + ndn != n: raise ValueError("nup + ndn != n")
if dim != 3: raise ValueError("Only dim = 3 is supported!")

# The energy are in the unit of Kelvin (K), and the length are in the unit of Angstrom (A).
import src.initconstants
lamb, beta, L, Volume, EF, Theta = src.initconstants.initconstats(T, n, dens)

twist = jnp.array(args.twist)
print("n = %d, nup = %d, ndn = %d, dim = %d," % (n, nup, ndn ,dim), "twist =", twist)
print("lamb = %f, dens = %f, T = %f, beta = %f" % (lamb, dens, T, beta))
print("EF = %f, Theta = %f, L = %f, Vol = %f" % (EF, Theta, L, Volume))
print(flush=True)

####################################################################################

print("\n========== Initialize single-particle orbitals ==========")

sp_indices, Es = src.sp_orbitals(dim, args.Emax)
Ef = Es[nup-1]
print("beta = %f, Ef = %d, Emax = %d, corresponding delta_logit = %f"
        % (beta, Ef, args.Emax, beta * lamb * (2*jnp.pi/L)**2 * (args.Emax - Ef)))
num_states = Es.size
print("Number of available single-particle orbitals: %d" % num_states)

from scipy.special import comb
print("Total number of many-body states (%dup in %d) * (%ddown in %d): %d" 
      % (nup, num_states, ndn, num_states, comb(num_states, nup)**2) )

sp_indices_twist, Es_twist = src.twist_sort(sp_indices, twist)
del sp_indices, Es
sp_indices_twist, Es_twist = jnp.array(sp_indices_twist)[::-1], jnp.array(Es_twist)[::-1]
print(flush=True)

####################################################################################

print("\n========== Initialize many-body state distribution ==========")

import haiku as hk
def forward_fn(state_idx):
    model = src.Transformer(num_states, args.nlayers, args.modelsize, args.nheads, args.nhidden, args.remat_van)
    return model(state_idx)
van = hk.transform(forward_fn)
state_idx_dummy = sp_indices_twist[-n:].astype(jnp.float64)
params_van = van.init(key, state_idx_dummy)

raveled_params_van, _ = ravel_pytree(params_van)
print("#parameters in the autoregressive model: %d" % raveled_params_van.size)

import src.sampler_spin
sampler, log_prob_novmap = src.sampler_spin.make_autoregressive_sampler_spin(van, 
                                        sp_indices_twist, nup, ndn, num_states)
log_prob = jax.vmap(log_prob_novmap, (None, 0), 0)
print(flush=True)

####################################################################################

print("\n========== Pretraining ==========")
# Pretraining parameters for the free-fermion model.
pre_lr = 1e-3
pre_sr, pre_damping, pre_maxnorm = True, 0.001, 0.001
pre_batch = 8192

freefermion_path = args.folder + "prevan/" \
                + "n_%d_dens_%g_T_%g_Emax_%d" \
                    % (n, dens, T, args.Emax) \
                + ("_twist" + "_%g"*dim + "/") % tuple(twist) \
                + "nlayers_%d_modelsize_%d_nheads_%d_nhidden_%d" % \
                    (args.nlayers, args.modelsize, args.nheads, args.nhidden) \
                + ("_damping_%g_maxnorm_%g" % (pre_damping, pre_maxnorm)
                    if pre_sr else "_lr_%g" % pre_lr) \
                + "_batch_%d" % pre_batch

print("#freefermion_path: ", freefermion_path)

import os
if not os.path.isdir(freefermion_path):
    os.makedirs(freefermion_path)
    print("Create freefermion directory: %s" % freefermion_path)

pretrained_model_filename = src.pretrained_model_filename(freefermion_path)
if os.path.isfile(pretrained_model_filename):
    print("Load pretrained free-fermion model parameters from file: %s" % pretrained_model_filename)
    params_van = src.load_data(pretrained_model_filename)
else:
    print("No pretrained free-fermion model found. Initialize parameters from scratch...")
    import src.freefermion.pretraining_spin
    params_van = src.freefermion.pretraining_spin.pretrain_spin(van, params_van,
                          nup, ndn, dim, args.Emax, twist,
                          beta, L, lamb,
                          freefermion_path, key,
                          pre_lr, pre_sr, pre_damping, pre_maxnorm,
                          pre_batch, args.pre_epoch)
    print("Initialization done. Save the model to file: %s" % pretrained_model_filename)
    src.save_data(params_van, pretrained_model_filename)
print(flush=True)

####################################################################################

print("\n========== Initialize normalizing flow ==========")

import src.flow_spin
mask_flow = ~jnp.eye(n, dtype=bool)
def flow_fn(x):
    model = src.flow_spin.FermiNet_spin(args.depth, args.spsize, args.tpsize, args.Np, args.Nf,
                                        L, mask_flow, 0.01, args.remat_flow)
    return model(x)
flow = hk.transform(flow_fn)
x_dummy = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
params_flow = flow.init(key, x_dummy)

raveled_params_flow, _ = ravel_pytree(params_flow)
print("#parameters in the flow model: %d" % raveled_params_flow.size)

import src.logpsi_spin
logpsi_novmap = src.logpsi_spin.make_logpsi(flow, sp_indices_twist, L)
logphi, logjacdet = src.logpsi_spin.make_logphi_logjacdet(flow, sp_indices_twist, L)
logp = src.logpsi_spin.make_logp(logpsi_novmap)
if args.mala:
    quantum_force = src.logpsi_spin.make_quantum_force(logpsi_novmap)
print(flush=True)

####################################################################################

print("\n========== Initialize potential energy ==========")

import src.potential
rlin, rcut = args.rlin, L/2

Vrlin = src.potential.func_potential_Aziz(rlin)
Vrlin_grad = src.potential.grad_potential_Aziz(rlin)
print("rlin = %f, V(rlin) = %f, V'(rlin) = %f" % (rlin, Vrlin, Vrlin_grad))

Vrcut = src.potential.func_potential_Aziz(rcut)
Vtail = src.potential.fun_Vtail(rcut, dens)
print("rcut = %f, V(rcut) = %f, Vtail(rcut) = %f" %(rcut, Vrcut, Vtail))
params_potential = (rlin, rcut, Vrlin, Vrlin_grad, Vtail)
print(flush=True)

####################################################################################

print("\n========== Initialize optimizer ==========")

import optax
if args.sr:
    classical_score_fn = src.sampler_spin.make_classical_score(log_prob_novmap)
    quantum_score_fn = src.logpsi_spin.make_quantum_score(logpsi_novmap)
    fishers_fn, optimizer = src.hybrid_fisher_sr(classical_score_fn, quantum_score_fn,
            args.lr_c, args.lr_q, args.decay,
            args.damping_c, args.damping_q, args.maxnorm_c, args.maxnorm_q)
    print("Optimizer hybrid_fisher_sr: \n    learining rate = %.5f, %.5f, decay = %.5f." 
          % (args.lr_c, args.lr_q, args.decay))
    print("    damping = %.5f, %.5f, maxnorm = %.5f, %.5f." 
          %(args.damping_c, args.damping_q, args.maxnorm_c, args.maxnorm_q))
else:
    optimizer = optax.adam(args.lr)
    print("Optimizer adam: lr = %.3f." % args.lr)
print(flush=True)

####################################################################################

print("\n========== Checkpointing ==========")
import src.VMC
batch, num_devices, acc_steps = args.batch, args.num_devices, args.acc_steps

path = args.folder + "interacting" \
                   + ("_rlin_%g" % rlin + "/") \
                   + "n_%d_dens_%g_T_%g_Emax_%d" \
                    % (n, dens, T, args.Emax) \
                   + ("_twist" + "_%g"*dim + "/") % tuple(twist) \
                   + "nlayers_%d_modelsize_%d_nheads_%d_nhidden_%d" % \
                      (args.nlayers, args.modelsize, args.nheads, args.nhidden) \
                   + "_depth_%d_spsize_%d_tpsize_%d_Np_%d_Nf_%d" % \
                      (args.depth, args.spsize, args.tpsize, args.Np, args.Nf) \
                   + "_mctherm_%d_mcsteps_%d_mcstddev_%g" % (args.mc_therm, args.mc_steps, args.mc_stddev) \
                   + ("_mala" if args.mala else "") \
                   + ("_hut" if args.hutchinson else "") \
                   + ("_lr_%g_%g_decay_%g_damping_%g_%g_maxnorm_%g_%g" 
        % (args.lr_c, args.lr_q, args.decay, args.damping_c, args.damping_q, args.maxnorm_c, args.maxnorm_q)
                        if args.sr else "_lr_%g" % args.lr) \
                   + "_clip_%g" % (args.clip_factor) \
                   + "_batch_%d_accsteps_%d_ndevices_%d" % (batch, acc_steps, num_devices)
                   
print("#path: ", path, flush=True)

if not os.path.isdir(path):
    os.makedirs(path)
    print("Create directory: %s" % path)
load_ckpt_filename = src.ckpt_filename(args.epoch_finished, path)

print("Number of GPU devices:", num_devices)
if num_devices != jax.device_count():
    raise ValueError("Expected %d GPU devices. Got %d." % (num_devices, jax.device_count()))

if os.path.isfile(load_ckpt_filename):
    print("Load checkpoint file: %s" % load_ckpt_filename)
    ckpt = src.load_data(load_ckpt_filename)
    keys, x, params_van, params_flow, opt_state = \
        ckpt["keys"], ckpt["x"], ckpt["params_van"], ckpt["params_flow"], ckpt["opt_state"]
    x, keys = src.shard(x), src.shard(keys)
    params_van, params_flow = src.replicate((params_van, params_flow), num_devices)
else:
    print("No checkpoint file found. Start from scratch.")
    opt_state = optimizer.init((params_van, params_flow))
    print("Initialize key and coordinate samples...")

    if batch % num_devices != 0:
        raise ValueError("Batch size must be divisible by the number of GPU devices. "
                         "Got batch = %d for %d devices now." % (batch, num_devices))
    batch_per_device = batch // num_devices

    x = jax.random.uniform(key, (num_devices, batch_per_device, n, dim), minval=0., maxval=L)
    keys = jax.random.split(key, num_devices)
    x, keys = src.shard(x), src.shard(keys)
    params_van, params_flow = src.replicate((params_van, params_flow), num_devices)

    for i in range(args.mc_therm):
        if args.mala:
            keys, _, x, accept_rate = src.VMC.sample_stateindices_and_x_mala(keys,
                                    sampler, params_van,
                                    logp, x, params_flow,
                                    quantum_force,
                                    args.mc_steps, args.mc_stddev, L)            
        else:
            print("---- thermal step %d ----" % (i+1))
            keys, _, x, accept_rate = src.VMC.sample_stateindices_and_x_mcmc(keys,
                                    sampler, params_van,
                                    logp, x, params_flow,
                                    args.mc_steps, args.mc_stddev, L)
    print("keys shape:", keys.shape)
    print("x shape:", x.shape)
print(flush=True)

####################################################################################

print("\n========== Training ==========")

logpsi, logpsi_grad_laplacian = src.logpsi_spin.make_logpsi_grad_laplacian(logpsi_novmap, forloop=True,
                                                       hutchinson=args.hutchinson,
                                                       logphi=logphi, logjacdet=logjacdet)


observable_and_lossfn = src.VMC.make_loss(log_prob, logpsi, logpsi_grad_laplacian, 
                                      lamb, L, beta, params_potential, args.clip_factor)

#========== update function ==========
from functools import partial
@partial(jax.pmap, axis_name="p",
        in_axes=(0, 0, None, 0, 0, 0, 0, 0, 0, 0) +
                ((0, 0, 0, None) if args.sr else (None, None, None, None)),
        out_axes=(0, 0, None, 0, 0, 0, 0) +
                ((0, 0, 0) if args.sr else (None, None, None)),
        static_broadcasted_argnums=13 if args.sr else (10, 11, 12, 13),
        donate_argnums=(3, 4))
def update(params_van, params_flow, opt_state, state_indices, x, key,
        data_acc, grads_acc, classical_score_acc, quantum_score_acc,
        classical_fisher_acc, quantum_fisher_acc, quantum_score_mean_acc, final_step):

    data, classical_lossfn, quantum_lossfn = observable_and_lossfn(
            params_van, params_flow, state_indices, x, key)

    grad_params_van, classical_score = jax.jacrev(classical_lossfn)(params_van)
    grad_params_flow, quantum_score = jax.jacrev(quantum_lossfn)(params_flow)
    grads = grad_params_van, grad_params_flow
    grads, classical_score, quantum_score = jax.lax.pmean((grads, classical_score, quantum_score), axis_name="p")
    data_acc, grads_acc, classical_score_acc, quantum_score_acc = jax.tree_map(lambda acc, i: acc + i, 
                                        (data_acc, grads_acc, classical_score_acc, quantum_score_acc),
                                        (data, grads, classical_score, quantum_score))

    if args.sr:
        classical_fisher, quantum_fisher, quantum_score_mean = fishers_fn(params_van, params_flow, state_indices, x)
        classical_fisher_acc += classical_fisher
        quantum_fisher_acc += quantum_fisher
        quantum_score_mean_acc += quantum_score_mean

    if final_step:
        data_acc, grads_acc, classical_score_acc, quantum_score_acc = jax.tree_map(lambda acc: acc / acc_steps,
                                            (data_acc, grads_acc, classical_score_acc, quantum_score_acc))
        grad_params_van, grad_params_flow = grads_acc
        grad_params_van = jax.tree_map(lambda grad, classical_score: grad - data_acc["F_mean"] * classical_score,
                                            grad_params_van, classical_score_acc)
        grad_params_flow = jax.tree_map(lambda grad, quantum_score: grad - data_acc["E_mean"] * quantum_score,
                                            grad_params_flow, quantum_score_acc)
        grads_acc = grad_params_van, grad_params_flow

        if args.sr:
            classical_fisher_acc /= acc_steps
            quantum_fisher_acc /= acc_steps
            quantum_score_mean_acc /= acc_steps
            
        updates, opt_state = optimizer.update(grads_acc, opt_state,
                params=(classical_fisher_acc, quantum_fisher_acc, quantum_score_mean_acc) if args.sr else None)
        params_van, params_flow = optax.apply_updates((params_van, params_flow), updates)

    return params_van, params_flow, opt_state, data_acc, grads_acc, classical_score_acc, quantum_score_acc, \
            classical_fisher_acc, quantum_fisher_acc, quantum_score_mean_acc
    
mix_fisher = False

#========== open file ==========
log_filename = os.path.join(path, "data.txt")
f = open(log_filename, "w" if args.epoch_finished == 0 else "a",
            buffering=1, newline="\n")

#========== circulate ==========
for i in range(args.epoch_finished + 1, args.epoch + 1):
    t0 = time.time()
    
    data_acc = src.replicate({"F_mean": 0., "F2_mean": 0.,
                              "E_mean": 0., "E2_mean": 0.,
                              "K_mean": 0., "K2_mean": 0.,
                              "V_mean": 0., "V2_mean": 0.,
                              "S_mean": 0., "S2_mean": 0.,
                              }, num_devices)
    grads_acc = src.shard( jax.tree_map(jnp.zeros_like, (params_van, params_flow)) )
    classical_score_acc, quantum_score_acc = src.shard( jax.tree_map(jnp.zeros_like, (params_van, params_flow)) )
    if args.sr:
        classical_fisher_acc = src.replicate(jnp.zeros((raveled_params_van.size, raveled_params_van.size)), num_devices)
        quantum_fisher_acc   = src.replicate(jnp.zeros((raveled_params_flow.size, raveled_params_flow.size)), num_devices)
        quantum_score_mean_acc = src.replicate(jnp.zeros(raveled_params_flow.size), num_devices)
    else:
        classical_fisher_acc = quantum_fisher_acc = quantum_score_mean_acc = None
    accept_rate_acc = src.shard(jnp.zeros(num_devices))
    
    ###
    batch_per_device = batch // num_devices
    x_step = jnp.zeros([acc_steps, num_devices, batch_per_device, n, dim])

    for acc in range(acc_steps):
        if args.mala:
            keys, state_indices, x, accept_rate = src.VMC.sample_stateindices_and_x_mala(keys,
                                    sampler, params_van,
                                    logp, x, params_flow,
                                    quantum_force,
                                    args.mc_steps, args.mc_stddev, L)            
        else:
            keys, state_indices, x, accept_rate = src.VMC.sample_stateindices_and_x_mcmc(keys,
                                    sampler, params_van,
                                    logp, x, params_flow,
                                    args.mc_steps, args.mc_stddev, L)

        x_step = x_step.at[acc].set(x)
        accept_rate_acc += accept_rate
        final_step = (acc == (acc_steps-1))
        
        params_van, params_flow, opt_state, data_acc, grads_acc, classical_score_acc, quantum_score_acc, \
        classical_fisher_acc, quantum_fisher_acc, quantum_score_mean_acc \
            = update(params_van, params_flow, opt_state, state_indices, x, keys,
                     data_acc, grads_acc, classical_score_acc, quantum_score_acc,
                     classical_fisher_acc, quantum_fisher_acc, quantum_score_mean_acc, final_step)
    
    data = jax.tree_map(lambda x: x[0], data_acc)
    accept_rate = accept_rate_acc[0] / acc_steps
    F, F2_mean = data["F_mean"], data["F2_mean"]
    E, E2_mean = data["E_mean"], data["E2_mean"]
    K, K2_mean = data["K_mean"], data["K2_mean"]
    V, V2_mean = data["V_mean"], data["V2_mean"]
    S, S2_mean = data["S_mean"], data["S2_mean"]
    F_std = jnp.sqrt((F2_mean - F**2) / (batch*acc_steps))
    E_std = jnp.sqrt((E2_mean - E**2) / (batch*acc_steps))
    K_std = jnp.sqrt((K2_mean - K**2) / (batch*acc_steps))
    V_std = jnp.sqrt((V2_mean - V**2) / (batch*acc_steps))
    S_std = jnp.sqrt((S2_mean - S**2) / (batch*acc_steps))
    
    # Note: quantities with energy dimension obtained above are in units of [K]:
    F, F_std = F/n, F_std/n
    E, E_std = E/n, E_std/n
    K, K_std = K/n, K_std/n
    V, V_std = V/n, V_std/n
    S, S_std = S/n, S_std/n
    
    #========== print ==========
    t1 = time.time()
    dt = t1-t0
    print("iter: %05d" % i,
            " F: %.6f" % F, "(%.6f)" % F_std,
            " E: %.6f" % E, "(%.6f)" % E_std,
            " K: %.6f" % K, "(%.6f)" % K_std,
            " V: %.6f" % V, "(%.6f)" % V_std,
            " S: %.6f" % S, "(%.6f)" % S_std,
            " acc: %.4f" % accept_rate,
            " dt: %.3f" % dt, flush=True)
    
    #========== save ==========
    f.write( ("%6d" + "  %.6f"*10 + "  %.4f"*2 + "\n") % (i,
           F, F_std, E, E_std, K, K_std, V, V_std, S, S_std, accept_rate, dt) )

    if i % args.save_ckpt == 0:
        ckpt = {"keys": keys, "x": x, "x_step": x_step,
                "params_van": jax.tree_map(lambda x: x[0], params_van),
                "params_flow": jax.tree_map(lambda x: x[0], params_flow),
                "opt_state": opt_state}
        save_ckpt_filename = src.ckpt_filename(i, path)
        src.save_data(ckpt, save_ckpt_filename)
        print("Save checkpoint file: %s" % save_ckpt_filename, flush=True)

f.close()


