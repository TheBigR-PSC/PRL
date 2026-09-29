"""
Benchmark d'Allocation d'Échantillonnage VMC (NetKet)

Modèle jouet : TFIM 1D (d=1, L=4 spins)
Ansatz       : LogStateVector (Paramètres = Log-amplitudes directes)

Étude comparative du comportement de l'optimisation VMC et MCMC à nombre d'itérations
fixé pour différentes tailles d'échantillons par chaîne (1, 2, 5, 10, 50, 100).

Pourquoi N_CHAINS = 4 ?
-----------------------
Le calcul du diagnostic de Gelman-Rubin (R_hat) nécessite au minimum 2 chaînes MCMC
indépendantes (idéalement 4 ou plus). Avoir N_CHAINS = 4 permet de mesurer la variance
intra-chaîne vs inter-chaînes afin de s'assurer de la bonne ergodicité de l'échantillonnage.

Métriques tracées (Colormap: Magma) :
1. Écart relatif de l'énergie |(E - E_0) / E_0| vs Itérations
2. Diagnostic de convergence MCMC R_hat (Gelman-Rubin) vs Itérations
3. Écart relatif vs Échantillons cumulés (coût total de calcul)
4. V-score standard N * Var(H) / E^2 (Astrakhantsev et al., PRX 2021)
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # Backend non-interactif

import netket as nk
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Configuration du système physique & Référence exacte (Lanczos)
# ==============================================================================
L = 4           # Taille de la chaîne 1D (2^4 = 16 configurations)
H_VAL = 1.0     # Champ transverse h
LR = 0.005       # Taux d'apprentissage SGD
DIAG_SHIFT = 0.01
N_STEPS = 500   # Nombre d'itérations VMC fixe
N_CHAINS = 4    # 4 chaînes MCMC indépendantes (nécessaire pour R_hat)

graph = nk.graph.Chain(length=L, pbc=True)
hi = nk.hilbert.Spin(s=0.5, N=L)
ha = nk.operator.Ising(hilbert=hi, graph=graph, h=H_VAL)

# Énergie exacte de l'état fondamental via Lanczos
E0 = nk.exact.lanczos_ed(ha, k=1).item()

print("============================================================")
print(f"Benchmark d'Échantillonnage VMC | TFIM 1D (L={L}, h={H_VAL})")
print(f"Énergie exacte Lanczos E0 = {E0:.6f}")
print(f"Itérations = {N_STEPS} | Chaînes MCMC = {N_CHAINS} | LR = {LR}")
print("============================================================")


# ==============================================================================
# 2. Entraînement pour chaque configuration de N_samples
# ==============================================================================
# Échantillons par pas et par chaîne (1, 2, 5, 10, 50, 100)
samples_per_chain = [1, 2, 5, 10, 50, 100]
results = {}

for s_per_chain in samples_per_chain:
    n_s = s_per_chain * N_CHAINS
    n_total_exp = n_s * N_STEPS
    label = f"{s_per_chain} s/chaîne ({n_s} s/step)"
    print(f"\n>>> Lancement : {s_per_chain} samples/chaîne ({n_s} s/step) sur {N_STEPS} steps...")

    sampler = nk.sampler.MetropolisLocal(hi, n_chains=N_CHAINS, sweep_size=L)
    model = nk.models.LogStateVector(hi, param_dtype=jnp.complex128)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_s, seed=42)

    optimizer = nk.optimizer.Sgd(learning_rate=LR)
    sr = nk.optimizer.SR(diag_shift=DIAG_SHIFT, holomorphic=True)
    vmc = nk.VMC(ha, optimizer, variational_state=vstate, preconditioner=sr)

    log = nk.logging.RuntimeLog()
    vmc.run(n_iter=N_STEPS, out=log)

    iters = np.array(log.data["Energy"].iters)
    cum_samples = iters * n_s
    energy_mean = np.array(log.data["Energy"].Mean.real)
    energy_var = np.array(log.data["Energy"].Variance.real)

    rel_err = np.abs((energy_mean - E0) / np.abs(E0))
    v_score = L * energy_var / (energy_mean ** 2)
    r_hat = np.array(log.data["Energy"].R_hat)

    results[label] = {
        "iters": iters,
        "cum_samples": cum_samples,
        "energy": energy_mean,
        "rel_err": rel_err,
        "v_score": v_score,
        "r_hat": r_hat,
    }

# Référence exacte sans bruit Monte Carlo (FullSumState)
print(f"\n>>> Lancement de la référence exacte : FullSumState ({N_STEPS} itérations)...")
model_exact = nk.models.LogStateVector(hi, param_dtype=jnp.complex128)
vstate_exact = nk.vqs.FullSumState(hi, model_exact)
optimizer_exact = nk.optimizer.Sgd(learning_rate=LR)
sr_exact = nk.optimizer.SR(diag_shift=DIAG_SHIFT, holomorphic=True)
vmc_exact = nk.VMC(ha, optimizer_exact, variational_state=vstate_exact, preconditioner=sr_exact)

log_exact = nk.logging.RuntimeLog()
vmc_exact.run(n_iter=N_STEPS, out=log_exact)

iters_exact = np.array(log_exact.data["Energy"].iters)
energy_exact = np.array(log_exact.data["Energy"].Mean.real)
energy_var_exact = np.array(log_exact.data["Energy"].Variance.real)

results["Exact (FullSumState)"] = {
    "iters": iters_exact,
    "cum_samples": iters_exact * (2**L),
    "energy": energy_exact,
    "rel_err": np.abs((energy_exact - E0) / np.abs(E0)),
    "v_score": L * energy_var_exact / (energy_exact ** 2),
    "r_hat": np.ones_like(iters_exact),
}


# ==============================================================================
# 3. Figure 2x2 comparative avec la Colormap Magma
# ==============================================================================
fig, axs = plt.subplots(2, 2, figsize=(15, 11))
cmap = plt.get_cmap("magma")

# Palette Magma uniformément répartie de 0.15 (sombre) à 0.85 (lumineux)
def get_magma_palette(n):
    return [cmap(val) for val in np.linspace(0.15, 0.85, n)]

colors = get_magma_palette(len(results))

# Panneau 1 : Écart relatif vs Itérations
for (label, data), col in zip(results.items(), colors):
    ls = "--" if "Exact" in label else "-"
    lw = 2.5 if "Exact" in label else 1.8
    axs[0, 0].plot(data["iters"], data["rel_err"], label=label, lw=lw, ls=ls, color=col)
axs[0, 0].set_yscale("log")
axs[0, 0].set_xlabel("Itération VMC", fontsize=11)
axs[0, 0].set_ylabel(r"Écart relatif $|(E - E_0) / E_0|$", fontsize=11)
axs[0, 0].set_title(f"1. Évolution de l'écart relatif d'énergie ({N_STEPS} steps)", fontsize=12, fontweight="bold")
axs[0, 0].grid(True, which="both", ls="--", alpha=0.5)
axs[0, 0].legend(fontsize=9, loc="upper right")

# Panneau 2 : Diagnostic de Gelman-Rubin R_hat vs Itérations
for (label, data), col in zip(results.items(), colors):
    if "Exact" not in label:
        axs[0, 1].plot(data["iters"], data["r_hat"], label=label, lw=1.8, color=col)
axs[0, 1].axhline(1.05, color="red", linestyle=":", label=r"Seuil critique $\hat{R} = 1.05$", lw=1.5)
axs[0, 1].axhline(1.00, color="gray", linestyle="--", alpha=0.7, label=r"Idéal $\hat{R} = 1.00$", lw=1)
axs[0, 1].set_xlabel("Itération VMC", fontsize=11)
axs[0, 1].set_ylabel(r"Diagnostic $\hat{R}$ (Gelman-Rubin)", fontsize=11)
axs[0, 1].set_ylim(0.95, 1.45)
axs[0, 1].set_title(r"2. Convergence MCMC ($\hat{R} \to 1.00$)", fontsize=12, fontweight="bold")
axs[0, 1].grid(True, which="both", ls="--", alpha=0.5)
axs[0, 1].legend(fontsize=9, loc="upper right")

# Panneau 3 : Écart relatif vs Échantillons cumulés (Coût réel)
for (label, data), col in zip(results.items(), colors):
    ls = "--" if "Exact" in label else "-"
    lw = 2.5 if "Exact" in label else 1.8
    axs[1, 0].plot(data["cum_samples"], data["rel_err"], label=label, lw=lw, ls=ls, color=col)
axs[1, 0].set_yscale("log")
axs[1, 0].set_xlabel("Nombre d'échantillons cumulés évalués", fontsize=11)
axs[1, 0].set_ylabel(r"Écart relatif $|(E - E_0) / E_0|$", fontsize=11)
axs[1, 0].set_title("3. Précision vs Coût de calcul réel", fontsize=12, fontweight="bold")
axs[1, 0].grid(True, which="both", ls="--", alpha=0.5)
axs[1, 0].legend(fontsize=9, loc="upper right")

# Panneau 4 : V-score vs Itérations
for (label, data), col in zip(results.items(), colors):
    ls = "--" if "Exact" in label else "-"
    lw = 2.5 if "Exact" in label else 1.8
    axs[1, 1].plot(data["iters"], data["v_score"], label=label, lw=lw, ls=ls, color=col)
axs[1, 1].set_yscale("log")
axs[1, 1].set_xlabel("Itération VMC", fontsize=11)
axs[1, 1].set_ylabel("V-score standard", fontsize=11)
axs[1, 1].set_title("4. Évolution du V-score (Astrakhantsev PRX 2021)", fontsize=12, fontweight="bold")
axs[1, 1].grid(True, which="both", ls="--", alpha=0.5)
axs[1, 1].legend(fontsize=9, loc="upper right")

# Titre global
fig.suptitle(
    f"Benchmark d'Échantillonnage VMC (Colormap: Magma)\n"
    f"Système: TFIM 1D (h={H_VAL}) | L={L} spins | Ansatz: LogStateVector | N_chains={N_CHAINS} | LR={LR}",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)

plt.tight_layout(rect=[0, 0, 1, 0.94])
output_img = "benchmark_sampling_results.png"
output_pdf = "benchmark_sampling_results.pdf"
plt.savefig(output_img, dpi=300)
plt.savefig(output_pdf)
plt.close("all")

print("\n============================================================")
print(f"Résultats sauvegardés sous '{output_img}' et '{output_pdf}'.")
print("============================================================")
