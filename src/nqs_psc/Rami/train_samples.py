
import netket as nk
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# 1. Modèle jouet
L = 4
graph = nk.graph.Chain(length=L, pbc=True)
hi = nk.hilbert.Spin(s=0.5, N=L)
ha = nk.operator.Ising(hilbert=hi, graph=graph, h=1.0)
E0 = nk.exact.lanczos_ed(ha, k=1).item()

# 2. Comparaison des budgets d'échantillons
sample_budgets = [16, 64, 256, 1024]
results = {}

for N_s in sample_budgets:
    sa = nk.sampler.MetropolisLocal(hi, n_chains=16, sweep_size=L)
    model = nk.models.LogStateVector(hi, param_dtype=jnp.complex128)
    vs = nk.vqs.MCState(sa, model, n_samples=N_s, seed=42)
    
    optimizer = nk.optimizer.Sgd(learning_rate=0.05)
    sr = nk.optimizer.SR(diag_shift=0.01, holomorphic=True)
    gs = nk.VMC(ha, optimizer, variational_state=vs, preconditioner=sr)
    
    log = nk.logging.RuntimeLog()
    gs.run(n_iter=60, out=log)
    
    energy = np.array(log.data["Energy"].Mean.real)
    r_hat = np.array(log.data["Energy"].R_hat)  # Gelman-Rubin
    var_E = np.array(log.data["Energy"].Variance.real)
    
    results[f"N_samples={N_s}"] = {
        "rel_err": np.abs((energy - E0) / np.abs(E0)),
        "v_score": L * var_E / (energy ** 2),
        "R_hat": r_hat,
    }

# 3. Référence exacte sans sampling (FullSumState)
vs_exact = nk.vqs.FullSumState(hi, nk.models.LogStateVector(hi, param_dtype=jnp.complex128))
gs_exact = nk.VMC(ha, nk.optimizer.Sgd(learning_rate=0.05), variational_state=vs_exact, preconditioner=sr)
log_exact = nk.logging.RuntimeLog()
gs_exact.run(n_iter=60, out=log_exact)
results["Exact (FullSumState)"] = {
    "rel_err": np.abs((np.array(log_exact.data["Energy"].Mean.real) - E0) / np.abs(E0)),
    "v_score": L * np.array(log_exact.data["Energy"].Variance.real) / (np.array(log_exact.data["Energy"].Mean.real) ** 2),
    "R_hat": np.ones(60),
}