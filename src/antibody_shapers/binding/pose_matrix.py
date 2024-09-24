import jax
import numpy as np
from typing import Tuple
import jax.numpy as jnp
import jax.random as jr
import importlib.resources as pkg_resources

import antibody_shapers.gen_alg_basic as ga
import antibody_shapers.shaping_funcs as sf
import antibody_shapers.utils as utils


def get_pose_matrix(
    antibody: jnp.ndarray,
    antigen: jnp.ndarray,
    docking_pose_id: int,
    primary_sites_antibody: jnp.ndarray = ga.prim_site_antibody,
    secondary_sites_antigen: jnp.ndarray = ga.sec_sites_antigen,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get the pose matrix for a docking pose with a given ID.

    Args:
        antibody (jnp.ndarray): The antibody sequence.
        antigen (jnp.ndarray): The antigen sequence.
        docking_pose_id (int): The ID of the docking pose.
        primary_sites_antibody (jnp.ndarray, optional): The primary sites of the antibody. Defaults to ga.prim_site_antibody.
        secondary_sites_antigen (jnp.ndarray, optional): The secondary sites of the antigen. Defaults to ga.sec_sites_antigen.

    Returns:
        jnp.ndarray: docking pose matrix

    """
    primary_site_antibody = primary_sites_antibody[docking_pose_id]
    secondary_site_antigen = secondary_sites_antigen[docking_pose_id]

    p_antibody = jnp.pad(antibody, (0, 1), "constant", constant_values=(-1))
    p_antigen = jnp.pad(antigen, (0, 1), "constant", constant_values=(-1))

    antibody_aas = p_antibody[primary_site_antibody]
    antigen_aas = p_antigen[secondary_site_antigen]

    docking_pose_matrix = jnp.zeros((21, 21))

    docking_pose_matrix = docking_pose_matrix.at[antibody_aas, antigen_aas].add(1)

    return docking_pose_matrix[:20, :20]


def get_pose_id_and_matrix(
    antibody: jnp.ndarray,
    antigen: jnp.ndarray,
    top_indicies: jnp.ndarray = sf.top_indicies,
    primary_sites_antibody: jnp.ndarray = ga.prim_site_antibody,
    secondary_sites_antigen: jnp.ndarray = ga.sec_sites_antigen,
) -> Tuple[int, jnp.ndarray]:
    """
    Get the docking pose ID and pose matrix for a docking pose.

    Args:
        antibody (jnp.ndarray): The antibody sequence.
        antigen (jnp.ndarray): The antigen sequence.
        top_indicies (jnp.ndarray, optional): The top indicies. Defaults to sf.top_indicies.
        primary_sites_antibody (jnp.ndarray, optional): The primary sites of the antibody. Defaults to ga.prim_site_antibody.
        secondary_sites_antigen (jnp.ndarray, optional): The secondary sites of the antigen. Defaults to ga.sec_sites_antigen.

    Returns:
        Tuple[int, jnp.ndarray]: The docking pose ID and the docking pose matrix.
    """
    binding_fn = ga.get_reduced_binding(top_indicies)

    top_indicies_pose_id = binding_fn(antibody, antigen)["argmin"]
    pose_id = top_indicies[top_indicies_pose_id]

    pose_matrix = get_pose_matrix(
        antibody, antigen, pose_id, primary_sites_antibody, secondary_sites_antigen
    )

    return pose_id, pose_matrix


get_pose_id_and_matrix_v_antibody = jax.jit(
    jax.vmap(get_pose_id_and_matrix, in_axes=(0, None))
)
get_pose_id_and_matrix_v_antigen = jax.jit(
    jax.vmap(get_pose_id_and_matrix, in_axes=(None, 0))
)
