"""
===============================================================================
 Étape 1 : Calculs et Sauvegarde des Données du Benchmark VMC (Ultra-Léger)
===============================================================================
Exécute chaque simulation séquentiellement avec LR = 5e-3 (5 x 10^-3) et sauvegarde
les résultats dans des fichiers JSON individuels dans le dossier 'data/'.

Optimisé pour consommer un minimum de RAM et de ressources CPU afin d'éviter tout crash.
"""

import os
# Limite stricte des ressources pour préserver la mémoire système et le CPU
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["JAX_ENABLE_X64"] = "1"

import json
import gc
import warnings
warnings.filterwarnings("ignore")

import netket as nk
import jax
import jax.numpy as jnp
import numpy as np

# Dossier de sauvegarde des données
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Système Physique & Référence Exacte Lanczos
# -----------------------------------------------------------------------------
L = 4           # Chaîne de 4 spins (2^4 = 16 configurations)
H_VAL = 1.0     # Champ transverse h
LR = 5e-3       # Taux d'apprentissage ajusté à 5 x 10^-3 (0.005)
DIAG_SHIFT = 0.01
N_STEPS = 500   # Nombre d'itérations VMC
N_CHAINS = 4    # 4 chaînes MCMC indispensables pour le calcul de R_hat

graph = nk.graph.Chain(length=L, pbc=True)
hi = nk.hilbert.Spin(s=0.5, N=L)
ha = nk.operator.Ising(hilbert=hi, graph=graph, h=H_VAL)

# Calcul exact de l'état fondamental via Lanczos
E0 = float(nk.exact.lanczos_ed(ha, k=1).item())

print("============================================================")
print(f"Calcul VMC Optimisé | TFIM 1D (L={L}, h={H_VAL})")
print(f"Énergie exacte Lanczos E0 = {E0:.6f}")
print(f"Learning Rate = {LR} (5 x 10^-3) | N_STEPS = {N_STEPS}")
print("============================================================")

# -----------------------------------------------------------------------------
# 2. Exécution Séquentielle des Configurations d'Échantillonnage
# -----------------------------------------------------------------------------
samples_per_chain = [1, 2, 5, 10, 50, 100]

for s_per_chain in samples_per_chain:
    n_s = s_per_chain * N_CHAINS
    filename = os.path.join(DATA_DIR, f"samples_{s_per_chain}_per_chain.json")

    print(f"\n---> Calcul pour {s_per_chain} s/chaîne ({n_s} s/step)...")

    sampler = nk.sampler.MetropolisLocal(hi, n_chains=N_CHAINS, sweep_size=L)
    model = nk.models.LogStateVector(hi, param_dtype=jnp.complex128)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_s, seed=42)

    optimizer = nk.optimizer.Sgd(learning_rate=LR)
    sr = nk.optimizer.SR(diag_shift=DIAG_SHIFT, holomorphic=True)
    vmc = nk.VMC(ha, optimizer, variational_state=vstate, preconditioner=sr)

    log = nk.logging.RuntimeLog()
    vmc.run(n_iter=N_STEPS, out=log)

    iters = np.array(log.data["Energy"].iters, dtype=int).tolist()
    cum_samples = (np.array(iters) * n_s).tolist()
    energy_mean = np.array(log.data["Energy"].Mean.real, dtype=float).tolist()
    energy_var = np.array(log.data["Energy"].Variance.real, dtype=float).tolist()

    rel_err = np.abs((np.array(energy_mean) - E0) / np.abs(E0)).tolist()
    v_score = (L * np.array(energy_var) / (np.array(energy_mean) ** 2)).tolist()
    r_hat = np.array(log.data["Energy"].R_hat, dtype=float).tolist()

    data = {
        "label": f"{s_per_chain} s/chaîne ({n_s} s/step)",
        "s_per_chain": s_per_chain,
        "n_s": n_s,
        "lr": LR,
        "iters": iters,
        "cum_samples": cum_samples,
        "energy_mean": energy_mean,
        "energy_var": energy_var,
        "rel_err": rel_err,
        "v_score": v_score,
        "r_hat": r_hat,
        "E0": E0
    }

    # Sauvegarde immédiate sur disque (écrasement propre avec les nouvelles données)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"   Saved: {filename}")

    # Nettoyage explicite de la mémoire RAM et du cache JAX après chaque test
    del vstate, sampler, vmc, log, optimizer, model
    jax.clear_caches()
    gc.collect()

# -----------------------------------------------------------------------------
# 3. Référence Exacte (FullSumState)
# -----------------------------------------------------------------------------
exact_filename = os.path.join(DATA_DIR, "exact_fullsumstate.json")
print("\n---> Calcul de la référence exacte (FullSumState)...")
model_exact = nk.models.LogStateVector(hi, param_dtype=jnp.complex128)
vstate_exact = nk.vqs.FullSumState(hi, model_exact)
optimizer_exact = nk.optimizer.Sgd(learning_rate=LR)
sr_exact = nk.optimizer.SR(diag_shift=DIAG_SHIFT, holomorphic=True)
vmc_exact = nk.VMC(ha, optimizer_exact, variational_state=vstate_exact, preconditioner=sr_exact)

log_exact = nk.logging.RuntimeLog()
vmc_exact.run(n_iter=N_STEPS, out=log_exact)

iters_exact = np.array(log_exact.data["Energy"].iters, dtype=int).tolist()
energy_exact = np.array(log_exact.data["Energy"].Mean.real, dtype=float).tolist()
energy_var_exact = np.array(log_exact.data["Energy"].Variance.real, dtype=float).tolist()

rel_err_exact = np.abs((np.array(energy_exact) - E0) / np.abs(E0)).tolist()
v_score_exact = (L * np.array(energy_var_exact) / (np.array(energy_exact) ** 2)).tolist()

data_exact = {
    "label": "Exact (FullSumState)",
    "lr": LR,
    "iters": iters_exact,
    "cum_samples": (np.array(iters_exact) * (2**L)).tolist(),
    "energy_mean": energy_exact,
    "rel_err": rel_err_exact,
    "v_score": v_score_exact,
    "r_hat": [1.0] * len(iters_exact),
    "E0": E0
}

with open(exact_filename, "w") as f:
    json.dump(data_exact, f, indent=2)

print(f"   Saved: {exact_filename}")

del vstate_exact, vmc_exact, log_exact
jax.clear_caches()
gc.collect()

print("\n============================================================")
print(f"Tous les calculs (LR = {LR}) sont sauvegardés dans '{DATA_DIR}/'.")
print("Vous pouvez maintenant lancer 'plot_benchmark.py' pour tracer les figures.")
print("============================================================")
