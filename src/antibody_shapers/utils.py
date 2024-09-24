import jax.numpy as jnp
import numpy as np

aminoacid_ids = {
    "C": 0,
    "M": 1,
    "F": 2,
    "I": 3,
    "L": 4,
    "V": 5,
    "W": 6,
    "Y": 7,
    "A": 8,
    "G": 9,
    "T": 10,
    "S": 11,
    "N": 12,
    "Q": 13,
    "D": 14,
    "E": 15,
    "R": 16,
    "H": 17,
    "K": 18,
    "P": 19,
}

reversed_aminoacid_ids = {v: k for k, v in aminoacid_ids.items()}


def convert_aa_to_array(sequence: str) -> jnp.ndarray:
    """
    Convert a string sequence to an int array.
    """
    return jnp.array([aminoacid_ids[aa] for aa in sequence], dtype=int)


def convert_array_to_aa(amino_acid_ids: jnp.ndarray) -> list:
    """
    Convert an int array to a list of amino acid characters.
    """
    return [reversed_aminoacid_ids[int(aa)] for aa in amino_acid_ids]


def convert_aa_list(list_sequences: list) -> jnp.ndarray:
    """
    Convert a list of string sequences to an int array.
    """
    return jnp.array([convert_aa_to_array(seq) for seq in list_sequences], dtype=int)


def get_amino_acid_counts(antibody_arr):
    """
    Counts the idxs of each aminoacid in the antibody_arr, it should work for antigens as well.

    Args:
        antibody_arr (jnp.ndarray): (n, antibody_len) or (n, antigen_len)
    Returns:
        jnp.ndarray: (antibody_length, 20) or (antigen_length, 20)
    """
    antibody_idxs_shape = (20,) + antibody_arr.shape
    antibody_idx = np.ones(antibody_idxs_shape) * np.arange(20).reshape(-1, 1, 1)
    to_compare_anitbodies = np.tile(antibody_arr, (20, 1, 1))
    counts = np.sum(antibody_idx == to_compare_anitbodies, axis=-1)
    return counts.T


def get_cosine_sim(counts_matrix):
    """
    Get the cosine similarity of the counts_matrix.
    """
    dot_product = counts_matrix @ counts_matrix.T
    norm = np.sqrt(np.sum(counts_matrix * counts_matrix, axis=1))
    norm = norm.reshape(-1, 1) @ norm.reshape(1, -1)
    return dot_product / norm
