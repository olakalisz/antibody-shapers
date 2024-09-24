import jax
import jax.numpy as jnp
import jax.random as jr


import importlib.resources as pkg_resources

with pkg_resources.open_binary("antibody_shapers.data", "jpadded_matrix.npy") as f:
    jpadded_matrix = jnp.load(f)


def get_binding_function(
    primary_sites_antibody: jnp.ndarray,
    secondary_sites_antibody: jnp.ndarray,
    secondary_sites_antigen: jnp.ndarray,
    interaction_matrix: jnp.ndarray = None,
):
    """
    Get the binding function for the given binding configuration.
    """
    if interaction_matrix is None:
        interaction_matrix = jpadded_matrix

    def get_binding(antibody: jnp.ndarray, antigen: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the binding energy between an antibody and an antigen using Jax.

        Args:
            antibody (jnp.ndarray[int]): (n, antibody_len) The antibody aminoacid int array.
            antigen (jnp.ndarray[int]): (n, antigen_len) The antigen aminoacid int array.

        Returns:
            jnp.ndarray[float]: The binding energy between the antibody and antigen.
        """
        p_antibody = jnp.pad(antibody, (0, 1), "constant", constant_values=(-1))
        p_antigen = jnp.pad(antigen, (0, 1), "constant", constant_values=(-1))
        binding_1 = interaction_matrix[
            p_antibody[primary_sites_antibody], p_antigen[secondary_sites_antigen]
        ]
        binding_2 = interaction_matrix[
            p_antibody[primary_sites_antibody], p_antibody[secondary_sites_antibody]
        ]
        binding_full_list = binding_1.sum(axis=1) + binding_2.sum(axis=1)
        return jnp.min(binding_full_list)

    return get_binding


def get_binding_function_extras(
    primary_sites_antibody: jnp.ndarray,
    secondary_sites_antibody: jnp.ndarray,
    secondary_sites_antigen: jnp.ndarray,
    interaction_matrix: jnp.ndarray = None,
):
    """
    Get the binding function for the given binding configuration, along with some extra information
    """
    if interaction_matrix is None:
        interaction_matrix = jpadded_matrix

    def get_binding(antibody: jnp.ndarray, antigen: jnp.ndarray) -> jnp.ndarray:
        if antibody.shape[0] > antigen.shape[0]:
            raise ValueError("Antibody is longer than antigen! Probably the wrong way around :)")
        p_antibody = jnp.pad(antibody, (0, 1), "constant", constant_values=(-1))
        p_antigen = jnp.pad(antigen, (0, 1), "constant", constant_values=(-1))
        binding_1 = interaction_matrix[
            p_antibody[primary_sites_antibody], p_antigen[secondary_sites_antigen]
        ]
        binding_2 = interaction_matrix[
            p_antibody[primary_sites_antibody], p_antibody[secondary_sites_antibody]
        ]
        binding_full_list = binding_1.sum(axis=1) + binding_2.sum(axis=1)
        arg_min = jnp.argmin(binding_full_list)
        return {
            "min": binding_full_list[arg_min],
            "argmin": arg_min,
            "binding_full_list": binding_full_list,
        }

    return get_binding

