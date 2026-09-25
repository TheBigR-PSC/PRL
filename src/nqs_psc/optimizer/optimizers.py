import netket as nk
import jax
from functools import partial
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from tqdm import tqdm


@partial(jax.jit, static_argnames="model")
def estimate_energy(model, parameters, hamiltonian_jax, sigma):
    E_loc = compute_local_energies(model, parameters, hamiltonian_jax, sigma)

    E_average = jnp.mean(E_loc)
    E_variance = jnp.var(E_loc)
    E_error = jnp.sqrt(E_variance / E_loc.size)

    return nk.stats.Stats(mean=E_average, error_of_mean=E_error, variance=E_variance)


def compute_local_energies(model, parameters, hamiltonian_jax, sigma):
    eta, H_sigmaeta = hamiltonian_jax.get_conn_padded(sigma)

    logpsi_sigma = model.apply(parameters, sigma)
    logpsi_eta = model.apply(parameters, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)

    res = jnp.sum(H_sigmaeta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)

    return res


@partial(jax.jit, static_argnames="model")
def estimate_energy_and_gradient(model, parameters, hamiltonian_jax, x):
    # reshape the samples to a vector of samples with no extra batch dimensions
    x = x.reshape(-1, x.shape[-1])

    E_loc = compute_local_energies(model, parameters, hamiltonian_jax, x)

    # compute the energy as well
    E_average = jnp.mean(E_loc)
    E_variance = jnp.var(E_loc)
    E_error = jnp.sqrt(E_variance / E_loc.size)
    delta_E_loc = E_loc - E_average
    E = nk.stats.statistics(E_loc)
    E_average = E.mean
    E = nk.stats.Stats(mean=E_average, error_of_mean=E_error, variance=E_variance)
    _, unravel_params_fn = ravel_pytree(parameters)

    # compute the gradient using full jacobian
    jacobian = nk.jax.jacobian(
        model.apply,
        parameters["params"],
        x,
        mode="holomorphic",
        dense=True,
        center=False,
        chunk_size=None,
    )

    E_grad = 1 / E_loc.size * jnp.conj(jacobian.T) @ (delta_E_loc)
    E_grad = unravel_params_fn(E_grad)

    return E, E_grad


def optimize_SGD(model, sampler, ham, chain_length, n_iters=1000, learning_rate=1e-2):
    # Initialize
    parameters = model.init(jax.random.key(1), jnp.ones((ham.hilbert.size,)))
    sampler_state = sampler.init_state(model, parameters, seed=1)

    # Logging
    logger = nk.logging.RuntimeLog()

    for i in tqdm(range(n_iters)):
        # sample
        sampler_state = sampler.reset(model, parameters, state=sampler_state)
        samples, sampler_state = sampler.sample(
            model, parameters, state=sampler_state, chain_length=chain_length
        )

        # compute energy and gradient
        E, E_grad = estimate_energy_and_gradient(model, parameters, ham, samples)

        # update parameters
        parameters = jax.tree.map(
            lambda x, y: x - learning_rate * y, parameters, E_grad
        )

        # log energy
        logger(step=i, item={"Energy": E})
    return logger, model, parameters, sampler


@partial(jax.jit, static_argnames="model")
def compute_S_and_F_and_E(model, parameters, hamiltonian_jax, x):
    # reshape the samples to a vector of samples with no extra batch dimensions
    x = x.reshape(-1, x.shape[-1])
    # on aplatit car les focntions pour calculer énegries et graidents s'atendent à avoir une forme de la forme 5chose, n sites)
    # Cela aplatit les éventuels axes de batch multiples (ex. (4,5,10) → (20,10)),
    # pour obtenir une seule liste de configurations indépendantes.
    # le -1 signifie "tout sauf la dernière dimension"
    # le x.shape[-1]= nombre de sites du système

    E_loc = compute_local_energies(model, parameters, hamiltonian_jax, x)

    # compute the energy as well
    E_average = jnp.mean(E_loc)
    E_variance = jnp.var(E_loc)
    E_error = jnp.sqrt(E_variance / E_loc.size)
    E = nk.stats.statistics(E_loc)
    E_average = E.mean
    E = nk.stats.Stats(mean=E_average, error_of_mean=E_error, variance=E_variance)
    _, unravel_params_fn = ravel_pytree(parameters)
    # cette ligne est super importante quand tu manipules des modèles JAX/Flax, car elle sert à aplatir (vectoriser) tous les paramètres d’un modèle neural en un seul grand vecteur
    # la focntion ravel_pytree a deux argumenst en 1 le vecteur obtenu par transformation du dico parameters
    # et unravle qui est la focntion inverse de la transformation
    # !!!! On fait souvent ca car pour calculer gradient on veut un vecteur or model et les autres fonctions veuletn un pytree en entrée (dictionnaire)
    # différenc entre Stats et statistiques
    # statistiques est une boite à outils qui peiut calculer la moyenne, la variance...
    # statistqiues renvoie alors un OBJET Stat qui contient toutes ses valeurs
    # Stat est donc une classe qui REPRESENTE/EST UN OBJET un ensemble de statistiques

    # compute the gradient using full jacobian
    jacobian = nk.jax.jacobian(
        model.apply,
        parameters["params"],
        x,
        mode="holomorphic",
        dense=True,
        center=False,
        chunk_size=None,
    )
    J = jacobian - jnp.mean(jacobian, axis=0, keepdims=True)
    S = (jnp.conj(J).T @ J) / J.shape[0]
    F = (1 / E_loc.size) * jnp.conj(jacobian.T) @ (E_loc - E_average)
    # Ici, E_grad est encore un vecteur aplati de paramètres. unravel_params_fn est une fonction créée avec jax.flatten_util.ravel_pytree ou équivalent.
    # Elle sert à "restructurer" (unravel) ce vecteur plat en un pytree ayant la même structure que les paramètres du modèle (par ex. dictionnaire avec {"layer1": W, "layer2": b, ...}).
    # Donc après cette ligne, E_grad devient directement dans le bon format pour un optimiseur JAX (comme optax).

    return S, F, E


def natural_gradient(model, parameters, hamiltonian_jax, x, diag, method="chol"):
    # On commence par régulariser
    S, F, E = compute_S_and_F_and_E(model, parameters, hamiltonian_jax, x)

    # Ajout du shift sans construire I ni appeler jnp.diag / jnp.eye
    S_inv = S + jnp.eye(S.shape[0]) * diag

    # On résout
    _, unravel_params_fn = ravel_pytree(parameters)

    if method == "chol":
        L = jnp.linalg.cholesky(S_inv)
        y = jax.scipy.linalg.solve_triangular(L, -F, lower=True)
        delta = jax.scipy.linalg.solve_triangular(L.T, y, lower=False)
        delta = unravel_params_fn(delta)
    if method == "pinv":
        #  Décomposition spectrale
        w, Q = jnp.linalg.eigh(S)
        # Filtrage des petites valeurs propres (valeurs propres nulles)
        w_max = jnp.max(jnp.abs(w))
        mask = jnp.abs(w) > (1e-2 * w_max)
        w_inv = jnp.where(mask, 1.0 / w, 0.0)
        # 4. Pseudo-inverse et solution
        S_pinv = (Q * w_inv) @ Q.T
        delta = S_pinv @ (-F)
        delta = unravel_params_fn(delta)

    if method == "diago":
        w, Q = jnp.linalg.eigh(S_inv)
        Y = Q @ jnp.diag(1.0 / w) @ Q.T
        delta = Y @ (-F)

        delta = unravel_params_fn(delta)

    return delta, E


def optimize_NGD(
    model, sampler, ham, chain_length, diag_shift, n_iters, learning_rate, method="chol"
):
    # Initialize
    parameters = model.init(jax.random.key(1), jnp.ones((ham.hilbert.size,)))
    sampler_state = sampler.init_state(model, parameters, seed=1)

    # Logging
    logger = nk.logging.RuntimeLog()

    for i in tqdm(range(n_iters)):
        # sample
        sampler_state = sampler.reset(model, parameters, state=sampler_state)
        samples, sampler_state = sampler.sample(
            model, parameters, state=sampler_state, chain_length=chain_length
        )

        # compute energy and gradient
        delta, E = natural_gradient(model, parameters, ham, samples, diag_shift, method)

        # update parameters
        parameters = jax.tree.map(lambda x, y: x + learning_rate * y, parameters, delta)

        # log energy
        logger(step=i, item={"Energy": E})
    return logger, model, parameters, sampler
