"""
===============================================================================
 Étape 2 : Tracé des Figures du Benchmark VMC à partir des Données Sauvegardées
===============================================================================
Lit les fichiers JSON générés par 'run_benchmark.py' dans le dossier 'data/'
et génère une figure 2x2 comparant les performances avec la colormap Magma.

Hyper rapide (< 1 seconde) et sans aucune consommation mémoire CPU/RAM.
"""

import os
import json
import glob
import matplotlib
matplotlib.use("Agg")  # Mode non-interactif ultra-léger

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Le dossier '{DATA_DIR}' n'existe pas. Lancez d'abord 'python run_benchmark.py'.")

# Charge toutes les configurations sauvegardées
json_files = sorted(glob.glob(os.path.join(DATA_DIR, "samples_*_per_chain.json")))
exact_file = os.path.join(DATA_DIR, "exact_fullsumstate.json")

if not json_files:
    raise FileNotFoundError("Aucun fichier de données n'a été trouvé. Exécutez 'python run_benchmark.py'.")

results = []

# Charger les données MCMC
for filepath in json_files:
    with open(filepath, "r") as f:
        results.append(json.load(f))

# Trier par nombre d'échantillons par chaîne
results.sort(key=lambda d: d.get("s_per_chain", 0))

# Charger la référence exacte si elle existe
exact_data = None
if os.path.exists(exact_file):
    with open(exact_file, "r") as f:
        exact_data = json.load(f)

# -----------------------------------------------------------------------------
# Configuration de la Colormap Magma
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(15, 11))
cmap = plt.get_cmap("magma")

# Génère des teintes Magma bien étalées
n_configs = len(results) + (1 if exact_data else 0)
colors = [cmap(val) for val in np.linspace(0.15, 0.85, n_configs)]

# -----------------------------------------------------------------------------
# Tracé des 4 Panneaux
# -----------------------------------------------------------------------------
# 1. Écart relatif vs Itérations
for data, col in zip(results, colors[:len(results)]):
    axs[0, 0].plot(data["iters"], data["rel_err"], label=data["label"], lw=1.8, color=col)

if exact_data:
    axs[0, 0].plot(exact_data["iters"], exact_data["rel_err"], label=exact_data["label"],
                   lw=2.5, ls="--", color=colors[-1])

axs[0, 0].set_yscale("log")
axs[0, 0].set_xlabel("Itération VMC", fontsize=11)
axs[0, 0].set_ylabel(r"Écart relatif $|(E - E_0) / E_0|$", fontsize=11)
axs[0, 0].set_title("1. Évolution de l'écart relatif d'énergie", fontsize=12, fontweight="bold")
axs[0, 0].grid(True, which="both", ls="--", alpha=0.5)
axs[0, 0].legend(fontsize=9, loc="upper right")

# 2. Diagnostic Gelman-Rubin R_hat vs Itérations
for data, col in zip(results, colors[:len(results)]):
    axs[0, 1].plot(data["iters"], data["r_hat"], label=data["label"], lw=1.8, color=col)

axs[0, 1].axhline(1.05, color="red", linestyle=":", label=r"Seuil critique $\hat{R} = 1.05$", lw=1.5)
axs[0, 1].axhline(1.00, color="gray", linestyle="--", alpha=0.7, label=r"Idéal $\hat{R} = 1.00$", lw=1)
axs[0, 1].set_xlabel("Itération VMC", fontsize=11)
axs[0, 1].set_ylabel(r"Diagnostic $\hat{R}$ (Gelman-Rubin)", fontsize=11)
axs[0, 1].set_ylim(0.95, 1.45)
axs[0, 1].set_title(r"2. Convergence MCMC ($\hat{R} \to 1.00$)", fontsize=12, fontweight="bold")
axs[0, 1].grid(True, which="both", ls="--", alpha=0.5)
axs[0, 1].legend(fontsize=9, loc="upper right")

# 3. Écart relatif vs Échantillons cumulés
for data, col in zip(results, colors[:len(results)]):
    axs[1, 0].plot(data["cum_samples"], data["rel_err"], label=data["label"], lw=1.8, color=col)

if exact_data:
    axs[1, 0].plot(exact_data["cum_samples"], exact_data["rel_err"], label=exact_data["label"],
                   lw=2.5, ls="--", color=colors[-1])

axs[1, 0].set_yscale("log")
axs[1, 0].set_xlabel("Nombre d'échantillons cumulés évalués", fontsize=11)
axs[1, 0].set_ylabel(r"Écart relatif $|(E - E_0) / E_0|$", fontsize=11)
axs[1, 0].set_title("3. Précision vs Coût de calcul réel", fontsize=12, fontweight="bold")
axs[1, 0].grid(True, which="both", ls="--", alpha=0.5)
axs[1, 0].legend(fontsize=9, loc="upper right")

# 4. V-score vs Itérations
for data, col in zip(results, colors[:len(results)]):
    axs[1, 1].plot(data["iters"], data["v_score"], label=data["label"], lw=1.8, color=col)

if exact_data:
    axs[1, 1].plot(exact_data["iters"], exact_data["v_score"], label=exact_data["label"],
                   lw=2.5, ls="--", color=colors[-1])

axs[1, 1].set_yscale("log")
axs[1, 1].set_xlabel("Itération VMC", fontsize=11)
axs[1, 1].set_ylabel("V-score standard", fontsize=11)
axs[1, 1].set_title("4. Évolution du V-score (Astrakhantsev PRX 2021)", fontsize=12, fontweight="bold")
axs[1, 1].grid(True, which="both", ls="--", alpha=0.5)
axs[1, 1].legend(fontsize=9, loc="upper right")

# Titre global
fig.suptitle(
    "Benchmark d'Échantillonnage VMC (Colormap: Magma)\n"
    "Système: TFIM 1D (h=1.0) | L=4 spins | Ansatz: LogStateVector",
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

print("============================================================")
print(f"Graphiques enregistrés avec succès : '{output_img}' et '{output_pdf}'.")
print("============================================================")
