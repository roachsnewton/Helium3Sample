import jax
import jax.numpy as jnp
import haiku as hk

class FermiNet_spin(hk.Module):
    def __init__(self, depth, spsize, tpsize, Np, Nf,
                 L, mask_flow, init_stddev=0.01, remat: bool=False):
        super().__init__()
        self.depth = depth
        self.L = L
        self.init_stddev = init_stddev
        self.splayers = [hk.Linear(spsize, w_init=hk.initializers.RandomNormal(stddev=self.init_stddev))
                            for _ in range(depth)]
        self.tplayers = [hk.Linear(tpsize, w_init=hk.initializers.RandomNormal(stddev=self.init_stddev))
                            for _ in range(depth-1)]
        self.Np = Np
        self.Nf = Nf
        self.remat = remat
        self.mask_flow = mask_flow

    ####################################################################################
    #========== Backflow ==========
    def backflow(self, x):
        def _fy(r):
            b, m = 2.6, 5.0
            fr = 0.5*(jnp.exp(-(b/r)**m)-1)
            return fr
        n, dim = x.shape
        x -= self.L * jnp.floor(x/self.L)
        rij = x[:, None, :] - x
        rij = rij - self.L*jnp.rint(rij/self.L)
        rij = rij[self.mask_flow].reshape(n, n-1, dim)
        dij = jnp.linalg.norm(rij, axis=-1)
        x = x + jnp.einsum('ij,ijd->id', _fy(dij), rij)
        return x

    ####################################################################################
    #========== Ferminet ==========  
    
    def _spstream0(self, x):
        """ 
            Initial spstream, with shape (n, spsize0).
            spsize0 = d.
            zeros_like(x) = x with all elements set to zero. 
        """
        return jnp.zeros_like(x)

    def _tpstream0(self, x):
        """ 
            Initial tpstream, with shape (n, n, tpsize0).
            tpsize0 = Np + 2d*Nf.
            cos_rij (3), sin_rij (3), dij (1).
        """
        n, _ = x.shape
        rij = x[:, None, :] - x
        dij = jnp.linalg.norm(jnp.sin(jnp.pi*rij/self.L) + jnp.eye(n)[..., None], axis=-1) *(1.0 - jnp.eye(n))
        
        f = []
        for ii in range(1, self.Np+1):
            f += [dij[..., None]**ii]
        for ii in range(1, self.Nf+1):
            f += [jnp.cos(2*ii*jnp.pi*rij/self.L), jnp.sin(2*ii*jnp.pi*rij/self.L)]
        return jnp.concatenate(f, axis=-1)

    def _f(self, spstream, tpstream):
        """
            The feature `f` as input to the sptream network.
            `f` has shape (n, fsize), where fsize = 2*spsize0 + tpsize0.
            f = [f1, mean(f1_up, axis=0), mean(f1_down, axis=0), mean(f2_up, axis=1), mean(f2_down, axis=1)]
        """
        n, _ = spstream.shape
        nup = n // 2
        f = jnp.concatenate((spstream,
                             spstream[:nup,:].mean(axis=0, keepdims=True).repeat(n, axis=0),
                             spstream[nup:,:].mean(axis=0, keepdims=True).repeat(n, axis=0),
                             tpstream[:,:nup,:].mean(axis=1),
                             tpstream[:,nup:,:].mean(axis=1),
                             ), axis=-1)

        return f

    ####################################################################################
    #========== main ==========  
    
    def __call__(self, x):
        
        # FermiNet: x -> z
        spstream, tpstream = self._spstream0(x), self._tpstream0(x)

        def block(spstream, tpstream, i):
            f = self._f(spstream, tpstream)
            if i==0:
                spstream = jax.nn.softplus( self.splayers[i](f) )
                tpstream = jax.nn.softplus( self.tplayers[i](tpstream) )
            else:
                spstream += jax.nn.softplus( self.splayers[i](f) )
                tpstream += jax.nn.softplus( self.tplayers[i](tpstream) )
            return spstream, tpstream

        if self.remat:
            block = hk.remat(block, static_argnums=2)

        for i in range(self.depth-1):
            spstream, tpstream = block(spstream, tpstream, i)

        f = self._f(spstream, tpstream)
        spstream += jax.nn.softplus( self.splayers[-1](f) )
        _, dim = x.shape
        final = hk.Linear(dim, w_init=hk.initializers.RandomNormal(stddev=self.init_stddev))
        x = x + final(spstream)

        # Backflow:
        ##x = self.backflow(x)
        
        return x

