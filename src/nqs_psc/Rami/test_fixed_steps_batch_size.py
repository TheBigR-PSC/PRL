"""
Test VMC à 1000 itérations FIXÉES et STRICTEMENT IDENTIQUES pour tous les runs :
Étude de l'impact de N_samples par pas (de 1 sample/chaîne jusqu'à grand batch).

Protocole expérimental :
- Nombre d'itérations : N_STEPS = 1000 pour TOUTES les configurations sans exception.
- Configurations de N_samples par pas testées sur 1000 steps :
  * N_s = 4    (1 sample par chaîne)  -> Total = 4 000 samples
  * N_s = 16   (4 samples par chaîne) -> Total = 16 000 samples
  * N_s = 64   (16 samples par chaîne)-> Total = 64 000 samples
  * N_s = 256  (64 samples par chaîne)-> Total = 256 000 samples
  * N_s = 1024 (256 samples/chaîne)   -> Total = 1 024 000 samples
  * Référence exacte sans bruit (FullSumState, 1000 steps)

Métriques tracées (Colormap: Magma) :
1. Écart relatif vs Itérations (1 à 1000)
2. Diagnostic de Gelman-Rubin R_hat vs Itérations (1 à 1000)
3. Écart relatif vs Échantillons cumulés (coût réel de calcul)
4. V-score vs Itérations (1 à 1000)
"""

import netket as nk
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Système physique & Référence exacte
# ==============================================================================
L = 4
H_VAL = 1.0
LR = 0.05
DIAG_SHIFT = 0.01
N_STEPS = 1000  # 1000 itérations STRICTEMENT IDENTIQUES pour tous les tests
N_CHAINS = 4    # 4 chaînes MCMC (permet N_s = 4 -> exactement 1 sample / chaîne / step)

graph = nk.graph.Chain(length=L, pbc=True)
hi = nk.hilbert.Spin(s=0.5, N=L)
ha = nk.operator.Ising(hilbert=hi, graph=graph, h=H_VAL)

E0 = nk.exact.lanczos_ed(ha, k=1).item()
print(f"============================================================")
print(f"Test à N_STEPS = {N_STEPS} IDENTIQUE pour tous les runs")
print(f"Système : TFIM 1D (L={L}, h={H_VAL}) | Énergie exacte Lanczos E0 = {E0:.6f}")
print(f"============================================================")


# ==============================================================================
# 2. Entraînement pour chaque configuration de N_samples sur 1000 steps
# ==============================================================================
sample_sizes = [4, 16, 64, 256, 1024]
results = {}

for n_s in sample_sizes:
    n_total_exp = n_s * N_STEPS
    label = f"$N_s = {n_s}$/step (Total: {n_total_exp:,})"
    print(f"\n>>> Lancement : N_samples = {n_s} par step ({N_STEPS} itérations) ...")
    
    sa = nk.sampler.MetropolisLocal(hi, n_chains=N_CHAINS, sweep_size=L)
    model = nk.models.LogStateVector(hi, param_dtype=jnp.complex128)
    vs = nk.vqs.MCState(sa, model, n_samples=n_s, seed=42)
    
    optimizer = nk.optimizer.Sgd(learning_rate=LR)
    sr = nk.optimizer.SR(diag_shift=DIAG_SHIFT, holomorphic=True)
    gs = nk.VMC(ha, optimizer, variational_state=vs, preconditioner=sr)
    
    log = nk.logging.RuntimeLog()
    gs.run(n_iter=N_STEPS, out=log)
    
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

# Référence exacte sans bruit Monte Carlo sur 1000 steps
print(f"\n>>> Lancement référence exacte : FullSumState ({N_STEPS} itérations) ...")
m_exact = nk.models.LogStateVector(hi, param_dtype=jnp.complex128)
vs_exact = nk.vqs.FullSumState(hi, m_exact)
optimizer_exact = nk.optimizer.Sgd(learning_rate=LR)
sr_exact = nk.optimizer.SR(diag_shift=DIAG_SHIFT, holomorphic=True)
gs_exact = nk.VMC(ha, optimizer_exact, variational_state=vs_exact, preconditioner=sr_exact)
log_exact = nk.logging.RuntimeLog()
gs_exact.run(n_iter=N_STEPS, out=log_exact)

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
# 3. Tracé comparatif complet (Figure 2x2 avec Colormap Magma)
# ==============================================================================
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
cmap = plt.get_cmap("magma")

def get_magma_palette(n):
    return [cmap(val) for val in np.linspace(0.15, 0.85, n)]

colors = get_magma_palette(len(results))

# Panneau 1 : Écart relatif vs Itérations (1 à 1000)
for (label, data), col in zip(results.items(), colors):
    ls = "--" if "Exact" in label else "-"
    lw = 2.5 if "Exact" in label else 1.8
    axs[0, 0].plot(data["iters"], data["rel_err"], label=label, lw=lw, ls=ls, color=col)
axs[0, 0].set_yscale("log")
axs[0, 0].set_xlabel("Itération VMC", fontsize=11)
axs[0, 0].set_ylabel(r"Écart relatif $|(E - E_0) / E_0|$", fontsize=11)
axs[0, 0].set_title(f"1. Convergence de l'Énergie ({N_STEPS} itérations)", fontsize=12, fontweight="bold")
axs[0, 0].grid(True, which="both", ls="--", alpha=0.5)
axs[0, 0].legend(fontsize=9, loc="upper right")

# Panneau 2 : Diagnostic de Gelman-Rubin R_hat vs Itérations (1 à 1000)
for (label, data), col in zip(results.items(), colors):
    if "Exact" not in label:
        axs[0, 1].plot(data["iters"], data["r_hat"], label=label, lw=1.8, color=col)
axs[0, 1].axhline(1.05, color="red", linestyle=":", label="Seuil critique $\\hat{R} = 1.05$", lw=1.5)
axs[0, 1].axhline(1.00, color="gray", linestyle="--", alpha=0.7, label="Idéal $\\hat{R} = 1.00$", lw=1)
axs[0, 1].set_xlabel("Itération VMC", fontsize=11)
axs[0, 1].set_ylabel("Diagnostic de Gelman-Rubin $\\hat{R}$", fontsize=11)
axs[0, 1].set_ylim(0.95, 1.45)
axs[0, 1].set_title(f"2. Qualité MCMC : Diagnostic $\\hat{{R}}$ ({N_STEPS} itérations)", fontsize=12, fontweight="bold")
axs[0, 1].grid(True, which="both", ls="--", alpha=0.5)
axs[0, 1].legend(fontsize=9, loc="upper right")

# Panneau 3 : Écart relatif vs Échantillons Cumulés (Coût de calcul réel)
for (label, data), col in zip(results.items(), colors):
    ls = "--" if "Exact" in label else "-"
    lw = 2.5 if "Exact" in label else 1.8
    axs[1, 0].plot(data["cum_samples"], data["rel_err"], label=label, lw=lw, ls=ls, color=col)
axs[1, 0].set_yscale("log")
axs[1, 0].set_xlabel("Nombre Cumulé d'Échantillons Évalués", fontsize=11)
axs[1, 0].set_ylabel(r"Écart relatif $|(E - E_0) / E_0|$", fontsize=11)
axs[1, 0].set_title("3. Précision atteinte vs Coût de calcul total", fontsize=12, fontweight="bold")
axs[1, 0].grid(True, which="both", ls="--", alpha=0.5)
axs[1, 0].legend(fontsize=9, loc="upper right")

# Panneau 4 : V-score vs Itérations (1 à 1000)
for (label, data), col in zip(results.items(), colors):
    ls = "--" if "Exact" in label else "-"
    lw = 2.5 if "Exact" in label else 1.8
    axs[1, 1].plot(data["iters"], data["v_score"], label=label, lw=lw, ls=ls, color=col)
axs[1, 1].set_yscale("log")
axs[1, 1].set_xlabel("Itération VMC", fontsize=11)
axs[1, 1].set_ylabel("V-score standard", fontsize=11)
axs[1, 1].set_title(f"4. Évolution du V-score ({N_STEPS} itérations)", fontsize=12, fontweight="bold")
axs[1, 1].grid(True, which="both", ls="--", alpha=0.5)
axs[1, 1].legend(fontsize=9, loc="upper right")

# Titre global avec métadonnées
fig.suptitle(
    f"Comparaison de N_samples par pas à {N_STEPS} itérations fixées (Colormap: Magma)\n"
    f"Système: TFIM 1D (h={H_VAL}) | Taille: L={L} | Ansatz: LogStateVector | LR={LR} | diag_shift={DIAG_SHIFT}",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)

plt.tight_layout(rect=[0, 0, 1, 0.94])
output_img = "fixed_1000_steps_sampling_comparison.png"
output_pdf = "fixed_1000_steps_sampling_comparison.pdf"
plt.savefig(output_img, dpi=300)
plt.savefig(output_pdf)
print(f"\n============================================================")
print(f"Graphiques sauvegardés dans '{output_img}' et '{output_pdf}'.")
print(f"============================================================")
plt.show()
