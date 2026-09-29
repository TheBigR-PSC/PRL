"""
Utilitaires pour l'échantillonnage Metropolis dans NetKet.
Fournit la règle de Global Flip (retournement global de spins) et un constructeur
d'échantillonneur prêt à l'emploi.
"""

import netket as nk
from netket.utils import struct


@struct.dataclass
class GlobalFlipRule(nk.sampler.rules.MetropolisRule):
    """
    Règle de transition Metropolis pour le retournement global simultané
    de tous les spins du système (sigma -> -sigma).
    Particulièrement utile pour les systèmes avec symétrie Z2 (Ising ferromagnétique).
    """  

    def transition(self, sampler, machine, parameters, state, key, σ):
        # Inverse tous les spins simultanément
        return -σ, None


def get_sampler(
    hilbert: nk.hilbert.AbstractHilbert,
    n_chains: int = 16,
    sweep_size: int = None,
    with_global_flip: bool = False,
    global_flip_prob: float = 0.05,
    **kwargs
) -> nk.sampler.MetropolisSampler:
    """
    Construit un échantillonneur Metropolis configurable en une seule ligne.

    Args:
        hilbert: L'espace de Hilbert du système.
        n_chains: Nombre de chaînes de Markov indépendantes en parallèle.
        sweep_size: Nombre de pas élémentaires par échantillon (par défaut: taille du système).
        with_global_flip: Si True, active le mélange local + global flip.
        global_flip_prob: Probabilité de proposer un flip global (ex: 0.05 pour 5%).
        **kwargs: Arguments additionnels pour MetropolisSampler.

    Returns:
        Un échantillonneur NetKet configuré.
    """
    if sweep_size is None:
        sweep_size = hilbert.size

    if not with_global_flip:
        # Échantillonneur standard à flips locaux uniquement
        return nk.sampler.MetropolisLocal(
            hilbert, n_chains=n_chains, sweep_size=sweep_size, **kwargs
        )
    else:
        # Règle composite : (1 - p) flips locaux + p flips globaux
        rule = nk.sampler.rules.MultipleRules(
            [nk.sampler.rules.LocalRule(), GlobalFlipRule()],
            probabilities=[1.0 - global_flip_prob, global_flip_prob],
        )
        return nk.sampler.MetropolisSampler(
            hilbert, rule=rule, n_chains=n_chains, sweep_size=sweep_size, **kwargs
        )
