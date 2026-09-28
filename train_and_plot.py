"""
Entraînement VMC pour l'ansatz LogStateVector (Poids par configuration)
sur différents Hamiltoniens de spin 1D (TFIM, Heisenberg, XXZ, J1-J2, TFLI),
avec comparaison des métriques (écart relatif et V-score) sur un même graphe.

Note physique sur la règle de signe de Marshall :
Pour les modèles antiferromagnétiques (Heisenberg, XXZ, J1-J2), une transformation
de jauge unitaire standard (rotation pi autour de z sur le sous-réseau impair :
sigma^x -> -sigma^x, sigma^y -> -sigma^y) rend les éléments non-diagonaux négatifs
(Hamiltonien stoquastique). Selon le théorème de Perron-Frobenius, l'état fondamental
possède alors des amplitudes positives, évitant les pièges de signes lors de l'optimisation VMC.
"""

import netket as nk
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Configuration du Système
# ==============================================================================
L = 4          # Nombre de spins (2^4 = 16 configurations)
N_ITERS = 100  # Nombre d'itérations VMC par Hamiltonien
N_SAMPLES = 512

graph = nk.graph.Chain(length=L, pbc=True)
hi = nk.hilbert.Spin(s=0.5, N=L)
sampler = nk.sampler.MetropolisLocal(hi, n_chains=16)

# ==============================================================================
# 2. Dictionnaire des Hamiltoniens (avec règle de signe de Marshall)
# ==============================================================================
sx = nk.operator.spin.sigmax
sy = nk.operator.spin.sigmay
sz = nk.operator.spin.sigmaz

def get_all_hamiltonians(hi, graph):
    N = hi.size
    hamiltonians = {}
    
    # 1. TFIM : Ising champ transverse (déjà stoquastique)
    hamiltonians["TFIM (h=1.0)"] = nk.operator.Ising(hilbert=hi, graph=graph, h=1.0)
    
    # 2. Heisenberg XXX (sign_rule=True par défaut dans NetKet sur réseau bipartite)
    hamiltonians["Heisenberg (J=1.0)"] = nk.operator.Heisenberg(
        hilbert=hi, graph=graph, J=1.0, sign_rule=True
    )
    
    # 3. XXZ Anisotrope (Delta = 0.5) avec transformation de signe Marshall
    # H = sum_i [ -(sx_i sx_{i+1} + sy_i sy_{i+1}) + Delta * sz_i sz_{i+1} ]
    Delta = 0.5
    hamiltonians["XXZ (Delta=0.5)"] = sum(
        -(sx(hi, i) * sx(hi, (i + 1) % N) + sy(hi, i) * sy(hi, (i + 1) % N))
        + Delta * sz(hi, i) * sz(hi, (i + 1) % N)
        for i in range(N)
    )
    
    # 4. J1-J2 Frustré (J1 = 1.0, J2 = 0.5) avec transformation de signe Marshall
    J1, J2 = 1.0, 0.5
    h_j1 = sum(
        -(sx(hi, i) * sx(hi, (i + 1) % N) + sy(hi, i) * sy(hi, (i + 1) % N))
        + sz(hi, i) * sz(hi, (i + 1) % N)
        for i in range(N)
    )
    h_j2 = sum(
        sx(hi, i) * sx(hi, (i + 2) % N)
        + sy(hi, i) * sy(hi, (i + 2) % N)
        + sz(hi, i) * sz(hi, (i + 2) % N)
        for i in range(N)
    )
    hamiltonians["J1-J2 (J2=0.5)"] = J1 * h_j1 + J2 * h_j2
    
    # 5. TFLI : Ising transverse + longitudinal (hx=1.0, hz=0.5)
    hx, hz = 1.0, 0.5
    h_zz = -sum(sz(hi, i) * sz(hi, (i + 1) % N) for i in range(N))
    h_x = -hx * sum(sx(hi, i) for i in range(N))
    h_z = -hz * sum(sz(hi, i) for i in range(N))
    hamiltonians["TFLI (hx=1.0, hz=0.5)"] = h_zz + h_x + h_z
    
    return hamiltonians

hamiltonians = get_all_hamiltonians(hi, graph)

# ==============================================================================
# 3. Boucle d'entraînement pour l'ansatz LogStateVector
# ==============================================================================
results = {}

for ham_name, ha in hamiltonians.items():
    print(f"\n==========================================")
    print(f"Entraînement LogStateVector sur : {ham_name}")
    print(f"==========================================")
    
    # 1. Énergie exacte Lanczos de référence
    E0 = nk.exact.lanczos_ed(ha, k=1).item()
    print(f"Énergie exacte Lanczos E0 = {E0:.6f}")
    
    # 2. Modèle LogStateVector (1 poids complexe par configuration)
    model = nk.models.LogStateVector(hi, param_dtype=jnp.complex128)
    vs = nk.vqs.MCState(sampler, model, n_samples=N_SAMPLES, seed=42)
    
    # 3. Optimiseur & VMC avec Natural Gradient (SR)
    optimizer = nk.optimizer.Sgd(learning_rate=0.05)
    sr = nk.optimizer.SR(diag_shift=0.01, holomorphic=True)
    gs = nk.VMC(ha, optimizer, variational_state=vs, preconditioner=sr)
    
    log = nk.logging.RuntimeLog()
    gs.run(n_iter=N_ITERS, out=log)
    
    iters = np.array(log.data["Energy"].iters)
    energy_mean = np.array(log.data["Energy"].Mean.real)
    energy_var = np.array(log.data["Energy"].Variance.real)
    
    # 4. Calcul des métriques
    rel_err = np.abs((energy_mean - E0) / np.abs(E0))
    v_score = L * energy_var / (energy_mean ** 2)
    
    results[ham_name] = {
        "iters": iters,
        "energy": energy_mean,
        "rel_err": rel_err,
        "v_score": v_score,
        "E0": E0,
    }

# ==============================================================================
# 4. Tracé comparatif multi-Hamiltoniens sur une même figure
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Graphe 1 : Écart relatif à l'état fondamental exact
for ham_name, data in results.items():
    ax1.plot(data["iters"], data["rel_err"], label=ham_name, lw=2)

ax1.set_xlabel("Itération", fontsize=12)
ax1.set_ylabel(r"Écart relatif $|(E - E_0) / E_0|$", fontsize=12)
ax1.set_yscale("log")
ax1.set_title("Convergence de l'énergie (LogStateVector)", fontsize=13)
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend(fontsize=9, loc="upper right")

# Graphe 2 : V-score standard
for ham_name, data in results.items():
    ax2.plot(data["iters"], data["v_score"], label=ham_name, lw=2)

ax2.set_xlabel("Itération", fontsize=12)
ax2.set_ylabel("V-score standard", fontsize=12)
ax2.set_yscale("log")
ax2.set_title("V-score par itération (LogStateVector)", fontsize=13)
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend(fontsize=9, loc="upper right")

plt.tight_layout()
output_img = "logstatevector_hamiltonians_comparison.png"
output_pdf = "logstatevector_hamiltonians_comparison.pdf"
plt.savefig(output_img, dpi=300)
plt.savefig(output_pdf)
print(f"\nGraphiques sauvegardés dans '{output_img}' et '{output_pdf}'.")
plt.show()
