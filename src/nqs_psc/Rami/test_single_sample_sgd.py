"""
Test de l'hypothèse d'optimisation Deep Learning :
« Est-il optimal d'utiliser un petit batch (1 sample / pas) avec beaucoup de mises à jour,
ou un grand batch avec moins de mises à jour, à budget total de calcul constant ? »

Protocole expérimental :
- Budget total fixé : N_total_samples = N_samples_per_step * N_steps = 10 000 échantillons.
- Configurations testées :
  * N_s = 4     (2500 itérations) -> régime "1 sample par chaîne" (quasi-online)
  * N_s = 16    (625 itérations)
  * N_s = 64    (156 itérations)
  * N_s = 256   (39 itérations)
  * N_s = 1024  (10 itérations)   -> régime "grand batch"

Métriques tracées (en fonction du nombre cumulé d'échantillons traités) :
1. Écart relatif à l'énergie exacte |(E - E_0) / E_0|
2. Diagnostic de convergence Gelman-Rubin R_hat
3. V-score standard
(Colormap: Magma)
"""

import netket as nk
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Système physique (TFIM 1D, L=4)
# ==============================================================================
L = 4
H_VAL = 1.0
LR = 0.05
DIAG_SHIFT = 0.01
N_TOTAL_BUDGET = 8000  # Budget total d'échantillons alloué à chaque expérience

graph = nk.graph.Chain(length=L, pbc=True)
hi = nk.hilbert.Spin(s=0.5, N=L)
ha = nk.operator.Ising(hilbert=hi, graph=graph, h=H_VAL)

E0 = nk.exact.lanczos_ed(ha, k=1).item()
print(f"============================================================")
print(f"Test Budget Constant : N_total = {N_TOTAL_BUDGET} échantillons")
print(f"Système TFIM 1D (L={L}, h={H_VAL}) | Énergie exacte Lanczos E0 = {E0:.6f}")
print(f"============================================================")

# ==============================================================================
# 2. Configurations de Batch Size (N_samples) à budget constant
# ==============================================================================
# Nombre de chaînes fixé à 4 (permettant 1 sample/chaîne au minimum)
N_CHAINS = 4

# (N_samples_par_step, N_steps_correspondant)
batch_configs = [
    (4, N_TOTAL_BUDGET // 4),       # 1 sample par chaîne (ultra-stochastique)
    (16, N_TOTAL_BUDGET // 16),     # 4 samples par chaîne
    (64, N_TOTAL_BUDGET // 64),     # 16 samples par chaîne
    (256, N_TOTAL_BUDGET // 256),   # 64 samples par chaîne
    (1024, N_TOTAL_BUDGET // 1024), # 256 samples par chaîne (grand batch)
]

results = {}

for n_s, n_iters in batch_configs:
    label = f"$N_s = {n_s}$ ({n_iters} steps)"
    print(f"\n>>> Test : {label} ...")
    
    sa = nk.sampler.MetropolisLocal(hi, n_chains=N_CHAINS, sweep_size=L)
    model = nk.models.LogStateVector(hi, param_dtype=jnp.complex128)
    vs = nk.vqs.MCState(sa, model, n_samples=n_s, seed=42)
    
    optimizer = nk.optimizer.Sgd(learning_rate=LR)
    sr = nk.optimizer.SR(diag_shift=DIAG_SHIFT, holomorphic=True)
    gs = nk.VMC(ha, optimizer, variational_state=vs, preconditioner=sr)
    
    log = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=log)
    
    iters = np.array(log.data["Energy"].iters)
    # Axe des abscisses : Nombre cumulé d'échantillons évalués
    cum_samples = iters * n_s
    
    energy_mean = np.array(log.data["Energy"].Mean.real)
    energy_var = np.array(log.data["Energy"].Variance.real)
    rel_err = np.abs((energy_mean - E0) / np.abs(E0))
    v_score = L * energy_var / (energy_mean ** 2)
    r_hat = np.array(log.data["Energy"].R_hat)
    
    results[label] = {
        "cum_samples": cum_samples,
        "iters": iters,
        "energy": energy_mean,
        "rel_err": rel_err,
        "v_score": v_score,
        "r_hat": r_hat,
    }

# ==============================================================================
# 3. Tracé des Métriques (Colormap: Magma)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
cmap = plt.get_cmap("magma")
colors = [cmap(val) for val in np.linspace(0.15, 0.85, len(results))]

# Graphe 1 : Écart relatif vs Nombre Cumulé d'Échantillons
for (label, data), col in zip(results.items(), colors):
    ax1.plot(data["cum_samples"], data["rel_err"], label=label, lw=1.8, color=col)

ax1.set_xlabel("Échantillons cumulés évalués ($N_{samples} \\times \\text{step}$)", fontsize=11)
ax1.set_ylabel(r"Écart relatif $|(E - E_0) / E_0|$", fontsize=11)
ax1.set_yscale("log")
ax1.set_title("Efficacité d'Échantillonnage : Écart relatif vs Budget Total", fontsize=12, fontweight="bold")
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend(fontsize=9, loc="upper right")

# Graphe 2 : Diagnostic Gelman-Rubin R_hat vs Nombre Cumulé d'Échantillons
for (label, data), col in zip(results.items(), colors):
    ax2.plot(data["cum_samples"], data["r_hat"], label=label, lw=1.8, color=col)

# Ligne de référence R_hat = 1.05 (seuil d'acceptabilité)
ax2.axhline(1.05, color="red", linestyle=":", label="Seuil $\\hat{R} = 1.05$", lw=1.5)
ax2.axhline(1.00, color="gray", linestyle="--", alpha=0.7, label="Idéal $\\hat{R} = 1.00$", lw=1)

ax2.set_xlabel("Échantillons cumulés évalués ($N_{samples} \\times \\text{step}$)", fontsize=11)
ax2.set_ylabel("Diagnostic de Gelman-Rubin $\\hat{R}$", fontsize=11)
ax2.set_ylim(0.95, 1.35)
ax2.set_title("Qualité MCMC : Diagnostic $\\hat{R}$ vs Budget Total", fontsize=12, fontweight="bold")
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend(fontsize=9, loc="upper right")

# Titre global avec métadonnées
fig.suptitle(
    f"Test de l'hypothèse 'Small Batch vs Large Batch' à Budget Constant ({N_TOTAL_BUDGET} samples)\n"
    f"Système: TFIM 1D (h={H_VAL}) | Taille: L={L} | Ansatz: LogStateVector | LR={LR} | diag_shift={DIAG_SHIFT}",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)

plt.tight_layout()
output_img = "single_sample_vs_batch_comparison.png"
output_pdf = "single_sample_vs_batch_comparison.pdf"
plt.savefig(output_img, dpi=300, bbox_inches="tight")
plt.savefig(output_pdf, bbox_inches="tight")
print(f"\nGraphiques sauvegardés dans '{output_img}' et '{output_pdf}'.")
plt.show()
