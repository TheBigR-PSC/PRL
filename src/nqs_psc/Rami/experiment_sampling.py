"""
Étude systématique de l'impact de l'échantillonnage Monte Carlo (MCMC) en VMC
sur un modèle jouet (TFIM 1D, L=4 spins, ansatz LogStateVector).

4 Expériences réalisées :
1. Budget total d'échantillons (N_samples) vs Référence exacte (FullSumState)
2. Structure des chaînes : Nombre de chaînes vs Longueur par chaîne (N_total constant)
3. Fréquence de balayage (sweep_size)
4. Phase de thermalisation / Burn-in (n_discard_per_chain)

Métriques tracées avec la colormap 'magma' :
- Écart relatif à l'état fondamental exact |(E - E_0) / E_0|
- V-score standard N * Var(H) / E^2 (Astrakhantsev et al., PRX 2021)
- Diagnostic de convergence multi-chaînes R_hat (Gelman & Rubin, 1992)
"""

import netket as nk
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Système physique & Référence exacte
# ==============================================================================
L = 4          # Nombre de spins (2^4 = 16 configurations)
N_ITERS = 70   # Nombre d'itérations VMC par test
H_VAL = 1.0    # Champ transverse h
HAM_NAME = "TFIM 1D"
ANSATZ_NAME = "LogStateVector"
LR = 0.05
DIAG_SHIFT = 0.01

graph = nk.graph.Chain(length=L, pbc=True)
hi = nk.hilbert.Spin(s=0.5, N=L)
ha = nk.operator.Ising(hilbert=hi, graph=graph, h=H_VAL)

E0 = nk.exact.lanczos_ed(ha, k=1).item()
print(f"============================================================")
print(f"Système : {HAM_NAME} (h={H_VAL}) | L={L} | Ansatz : {ANSATZ_NAME}")
print(f"Paramètres : LR={LR}, diag_shift={DIAG_SHIFT}")
print(f"Énergie exacte Lanczos E0 = {E0:.6f}")
print(f"============================================================")


# ==============================================================================
# 2. Fonction d'entraînement standardisée
# ==============================================================================
def run_vmc_experiment(variational_state, n_iters=N_ITERS):
    """Exécute VMC avec Natural Gradient (SR) et extrait les métriques clés."""
    optimizer = nk.optimizer.Sgd(learning_rate=LR)
    sr = nk.optimizer.SR(diag_shift=DIAG_SHIFT, holomorphic=True)
    gs = nk.VMC(ha, optimizer, variational_state=variational_state, preconditioner=sr)
    
    log = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=log)
    
    iters = np.array(log.data["Energy"].iters)
    energy_mean = np.array(log.data["Energy"].Mean.real)
    energy_var = np.array(log.data["Energy"].Variance.real)
    
    # R_hat (Gelman-Rubin) si disponible (uniquement pour MCState)
    if "R_hat" in log.data["Energy"]:
        r_hat = np.array(log.data["Energy"].R_hat)
    else:
        r_hat = np.ones_like(iters)  # Pour FullSumState
        
    rel_err = np.abs((energy_mean - E0) / np.abs(E0))
    v_score = L * energy_var / (energy_mean ** 2)
    
    return {
        "iters": iters,
        "energy": energy_mean,
        "rel_err": rel_err,
        "v_score": v_score,
        "r_hat": r_hat,
    }


# ==============================================================================
# 3. Exécution des 4 Expériences
# ==============================================================================

# --- EXPÉRIENCE 1 : Budget total d'échantillons (N_samples) ---
print("\n>>> Lancement Expérience 1 : Budget total d'échantillons...")
exp1_budgets = [16, 64, 256, 1024]
exp1_results = {}

for n_s in exp1_budgets:
    sa = nk.sampler.MetropolisLocal(hi, n_chains=16, sweep_size=L)
    m = nk.models.LogStateVector(hi, param_dtype=jnp.complex128)
    vs = nk.vqs.MCState(sa, m, n_samples=n_s, seed=42)
    exp1_results[f"$N_{{samples}} = {n_s}$"] = run_vmc_experiment(vs)

# Référence exacte sans bruit Monte Carlo
m_exact = nk.models.LogStateVector(hi, param_dtype=jnp.complex128)
vs_exact = nk.vqs.FullSumState(hi, m_exact)
exp1_results["Exact (FullSumState)"] = run_vmc_experiment(vs_exact)


# --- EXPÉRIENCE 2 : Structure des chaînes (N_chains vs N_samples_per_chain) ---
print("\n>>> Lancement Expérience 2 : Répartition des chaînes (N_total = 512)...")
N_TOTAL_EXP2 = 512
chain_configs = [4, 16, 64, 256]  # n_chains (longueur = 512 / n_chains)
exp2_results = {}

for n_c in chain_configs:
    sa = nk.sampler.MetropolisLocal(hi, n_chains=n_c, sweep_size=L)
    m = nk.models.LogStateVector(hi, param_dtype=jnp.complex128)
    vs = nk.vqs.MCState(sa, m, n_samples=N_TOTAL_EXP2, seed=42)
    l_c = N_TOTAL_EXP2 // n_c
    exp2_results[f"{n_c} chaînes $\\times$ {l_c} pas"] = run_vmc_experiment(vs)


# --- EXPÉRIENCE 3 : Fréquence de balayage (sweep_size) ---
print("\n>>> Lancement Expérience 3 : Fréquence de balayage (sweep_size)...")
sweeps = [1, 2, 4, 8]  # Pas de flips par échantillon (1, L//2, L, 2L)
exp3_results = {}

for sw in sweeps:
    sa = nk.sampler.MetropolisLocal(hi, n_chains=16, sweep_size=sw)
    m = nk.models.LogStateVector(hi, param_dtype=jnp.complex128)
    vs = nk.vqs.MCState(sa, m, n_samples=256, seed=42)
    exp3_results[f"$\\text{{sweep\\_size}} = {sw}$"] = run_vmc_experiment(vs)


# --- EXPÉRIENCE 4 : Phase de thermalisation / Burn-in (n_discard_per_chain) ---
print("\n>>> Lancement Expérience 4 : Thermalisation / Burn-in...")
discards = [0, 5, 20, 80]
exp4_results = {}

for disc in discards:
    sa = nk.sampler.MetropolisLocal(hi, n_chains=16, sweep_size=L)
    m = nk.models.LogStateVector(hi, param_dtype=jnp.complex128)
    vs = nk.vqs.MCState(sa, m, n_samples=256, n_discard_per_chain=disc, seed=42)
    exp4_results[f"$N_{{discard}} = {disc}$"] = run_vmc_experiment(vs)


# ==============================================================================
# 4. Tracé comparatif complet (Figure 2x2 avec Colormap Magma)
# ==============================================================================
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
cmap = plt.get_cmap("magma")

def get_magma_palette(n):
    """Génère n couleurs réparties sur la colormap magma (de 0.15 à 0.85 pour bon contraste)."""
    return [cmap(val) for val in np.linspace(0.15, 0.85, n)]

# Panneau 1 : Expérience 1 (Budget d'échantillons)
colors_exp1 = get_magma_palette(len(exp1_results))
for (label, data), col in zip(exp1_results.items(), colors_exp1):
    ls = "--" if "Exact" in label else "-"
    lw = 2.5 if "Exact" in label else 1.8
    axs[0, 0].plot(data["iters"], data["rel_err"], label=label, lw=lw, ls=ls, color=col)
axs[0, 0].set_yscale("log")
axs[0, 0].set_xlabel("Itération VMC", fontsize=11)
axs[0, 0].set_ylabel(r"Écart relatif $|(E - E_0) / E_0|$", fontsize=11)
axs[0, 0].set_title("1. Impact du Budget d'Échantillons ($N_{samples}$)", fontsize=12, fontweight="bold")
axs[0, 0].grid(True, which="both", ls="--", alpha=0.5)
axs[0, 0].legend(fontsize=9, loc="upper right")

# Panneau 2 : Expérience 2 (Structure des chaînes)
colors_exp2 = get_magma_palette(len(exp2_results))
for (label, data), col in zip(exp2_results.items(), colors_exp2):
    axs[0, 1].plot(data["iters"], data["rel_err"], label=label, lw=1.8, color=col)
axs[0, 1].set_yscale("log")
axs[0, 1].set_xlabel("Itération VMC", fontsize=11)
axs[0, 1].set_ylabel(r"Écart relatif $|(E - E_0) / E_0|$", fontsize=11)
axs[0, 1].set_title("2. Structure des Chaînes ($N_{chains} \\times L_{chain} = 512$)", fontsize=12, fontweight="bold")
axs[0, 1].grid(True, which="both", ls="--", alpha=0.5)
axs[0, 1].legend(fontsize=9, loc="upper right")

# Panneau 3 : Expérience 3 (sweep_size)
colors_exp3 = get_magma_palette(len(exp3_results))
for (label, data), col in zip(exp3_results.items(), colors_exp3):
    axs[1, 0].plot(data["iters"], data["rel_err"], label=label, lw=1.8, color=col)
axs[1, 0].set_yscale("log")
axs[1, 0].set_xlabel("Itération VMC", fontsize=11)
axs[1, 0].set_ylabel(r"Écart relatif $|(E - E_0) / E_0|$", fontsize=11)
axs[1, 0].set_title("3. Impact de la Fréquence de Balayage (`sweep_size`)", fontsize=12, fontweight="bold")
axs[1, 0].grid(True, which="both", ls="--", alpha=0.5)
axs[1, 0].legend(fontsize=9, loc="upper right")

# Panneau 4 : Expérience 4 (n_discard_per_chain)
colors_exp4 = get_magma_palette(len(exp4_results))
for (label, data), col in zip(exp4_results.items(), colors_exp4):
    axs[1, 1].plot(data["iters"], data["rel_err"], label=label, lw=1.8, color=col)
axs[1, 1].set_yscale("log")
axs[1, 1].set_xlabel("Itération VMC", fontsize=11)
axs[1, 1].set_ylabel(r"Écart relatif $|(E - E_0) / E_0|$", fontsize=11)
axs[1, 1].set_title("4. Impact du Burn-in (`n_discard_per_chain`)", fontsize=12, fontweight="bold")
axs[1, 1].grid(True, which="both", ls="--", alpha=0.5)
axs[1, 1].legend(fontsize=9, loc="upper right")

# Titre global et encart récapitulatif
fig.suptitle(
    f"Étude d'Échantillonnage MCMC en VMC (Colormap: Magma)\n"
    f"Système: {HAM_NAME} (h={H_VAL}) | Taille: L={L} | Ansatz: {ANSATZ_NAME} | LR={LR} | diag_shift={DIAG_SHIFT}",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("sampling_experiments_summary.png", dpi=300)
plt.savefig("sampling_experiments_summary.pdf")
print("\n============================================================")
print("Graphiques sauvegardés dans 'sampling_experiments_summary.png' et 'sampling_experiments_summary.pdf'.")
print("============================================================")
plt.show()
