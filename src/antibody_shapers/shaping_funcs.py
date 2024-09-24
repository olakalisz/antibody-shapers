import jax
import os
import jax.random as jr
import jax.numpy as jnp
import numpy as np
import pickle as pkl
import chex
from itertools import chain

from antibody_shapers import utils
import antibody_shapers.gen_alg_basic as ga

# Base directory
split_current_dir = os.getcwd().split('/')
package_name = "antibody_shapers"
base_dir = "/".join(split_current_dir[:split_current_dir.index(package_name)+1])


# Loading the indicies which corrospond to simulation fidelities
top_indicies = jnp.load(f"{base_dir}/data/top_bind_inds.npy")
medium_indicies = jnp.load(f"{base_dir}/data/med_bind_inds.npy")

low_res_bind = ga.get_reduced_binding(top_indicies)
med_res_bind = ga.get_reduced_binding(medium_indicies)
full_res_bind = ga.get_reduced_binding(slice(None))



# Choosing the target "antibody" as viruses target [IMPORTANT]
viral_target = utils.convert_aa_to_array("CARLVQLGLYY")


# The core antigen! The antigen of the dengue virus (PDB code 2R29) that we want to shape.
antigen = "SYSMCTGKFKVVKEIAETQHGTIVIRVQYEGDGSPCKIPFEIMDLEKRHVLGRLITVNPIVTEKDSPVNIEAEPPFGDSYIIIGVEPGQLKLNWFKK"
antigen_array = utils.convert_aa_to_array(antigen)


# Setting the clip terms for the binding. We don't want to reward binding too much more than what exists as a baseline! 
ag_target_binding = ga.get_regular_binding(viral_target, antigen_array)

ag_target_clip = (ag_target_binding - 1.0, jnp.inf)


# This antitarget was selected based on the effects of antibody optimisation.
antibody_antitarget = "GRFLVNLQAKKDREAWYYWGPWNKAYWFSDPGMFDPWKQAEQSYFCNANPVCYAEHFMLGPITQKTPMVYHDPEPSKGGCVTVHNNATDYIMPDCYN"
antibody_antitarget_array = utils.convert_aa_to_array(antibody_antitarget)

different_antibody_antitarget = "MADLEAVLADVSYLMAMEKSKATPAARASKKILLPEPSIRSVMQKYLEDRGEVTFEKIFSQKLGYLLFRDFCLNHLEEARPLVEFYEEIKKYEKLET"
different_antibody_antitarget_array = utils.convert_aa_to_array(different_antibody_antitarget)

@chex.dataclass
class ShapeParams:
    antigen_mut_rate: float
    antigen_pop_size: int
    antigen_selection_temperature: float
    
default_shape_params = ShapeParams(
    antigen_mut_rate = 1.0, 
    antigen_pop_size = 15, 
    antigen_selection_temperature = 0.05,
    )

def get_basic_ag_fitness_func(primary_antibody, bind_f, give_poses = True):
    ag_fit = ga.get_antigen_fitness(
        viral_target,
        primary_antibody,
        target_clip= ag_target_clip,
        give_poses=give_poses,
        bind_func_used=bind_f)
    return ag_fit
    

def jaxy_gen_shape_data(rng, primary_antibody,  bind_f, num_generations = 100, shape_params = default_shape_params, num_seq_reps = 1, num_parr_reps=1):
    """
    This generates multiple runs of the antigen evolution curve, and returns relevent information.
    Primarily, it gives the overall performance of the runs, the binding penalties, and the detailed run information.
    """
    
    mutation_func = lambda rng, x : ga.mutate(rng, x, shape_params.antigen_mut_rate/ x.shape[1])
    
    antigen_bind_f = jax.vmap(bind_f, in_axes = (None, 0))
    
    ag_fit = ga.get_antigen_fitness(
        viral_target, 
        primary_antibody, 
        target_clip= ag_target_clip, 
        give_poses=True, 
        bind_func_used=antigen_bind_f)
    
    v_ag_fitness = jax.vmap(ag_fit, in_axes = (None,0))
    
    single_iter_ag = ga.get_single_iteration(v_ag_fitness, mutation_func, shape_params.antigen_selection_temperature)
    scanned_iter_ag = ga.get_iterations_scan(single_iter_ag)
    
    
    antigen_start_pop = jnp.tile(antigen_array, (shape_params.antigen_pop_size, 1))
    state_ag_pop = ga.GenState(population=antigen_start_pop)
    
    def single_comp(rng):
        final_state_ag, extra_info_ag = scanned_iter_ag(rng, state_ag_pop, num_generations)
        return extra_info_ag
    
    
    rngs = jr.split(rng , num_seq_reps * num_parr_reps)
    
    vmapped_comp = jax.vmap(single_comp, in_axes = 0)
    
    reshaped_rngs = rngs.reshape((num_seq_reps, num_parr_reps, 2))
    
    combined_results = jax.lax.map(vmapped_comp, reshaped_rngs)
    
    def basic_reshape_first_two(x):
        x_shape = x.shape
        return x.reshape((-1,) + x_shape[2:])
    
    combined_results = jax.tree_map(basic_reshape_first_two, combined_results)
    
    binding_penalty = bind_f(primary_antibody, antibody_antitarget_array)["min"]
    
    final_info = {"performances": combined_results["best_fitness"].mean(axis = -1) - binding_penalty,
                  "binding_penalty": binding_penalty,
                  "run_info": combined_results}
    
    return final_info

def single_shape_run(rng, antibody, bind_f, start_antigen = antigen_array,
                     ag_t = viral_target,  ab_at = antibody_antitarget_array,
                     horizon = 100, shape_params = default_shape_params):
    """
    Does a single shaping run.
    Also now looks at the very first fitness, making this pure-greedy compatible.
    """
    
    mutation_func = lambda rng, x : ga.mutate(rng, x, shape_params.antigen_mut_rate/ x.shape[1])
    
    if ag_t is None:
        targ_clip = None
    else:
        binding_str = bind_f(ag_t, start_antigen)["min"]
        targ_clip = (binding_str - 1.0, jnp.inf)
    
    ag_fit = ga.get_antigen_fitness(
        ag_t, 
        antibody, 
        target_clip= targ_clip, 
        give_poses=True, 
        bind_func_used=bind_f)
    
    v_ag_fitness = jax.vmap(ag_fit, in_axes = (None,0))
    
    single_iter_ag = ga.get_single_iteration(v_ag_fitness, mutation_func, shape_params.antigen_selection_temperature)
    scanned_iter_ag = ga.get_iterations_scan(single_iter_ag)
    
    
    antigen_start_pop = jnp.tile(start_antigen, (shape_params.antigen_pop_size, 1))
    state_ag_pop = ga.GenState(population=antigen_start_pop)
    
    final_state_ag, extra_info_ag = scanned_iter_ag(rng, state_ag_pop, horizon)
    
    
    if ab_at is None:
        binding_penalty = 0.0
        bind_index = -1
    else:
        bind_details= bind_f(antibody, ab_at)
        binding_penalty = bind_details["min"]
        bind_index = bind_details["argmin"]
        
    
    all_fit_values = extra_info_ag["best_fitness"] - binding_penalty
    
    final_info = {"ag_performances": all_fit_values,
                  "binding_penalty": binding_penalty,
                  "ab_t_m_pose_index": bind_index,
                  "run_info": extra_info_ag}
    
    return final_info


if __name__ == "__main__":
    # Example of a single run
    seed = 10201
    rng = jr.PRNGKey(seed)

    # Start with a random antibody
    r1, r2 = jr.split(rng)
    start_antibody = jr.randint(r1, (11,), 0, 20)
    
    # Simulated viral escape against a single random antibody
    final_info = single_shape_run(rng=r2,
                                  antibody=start_antibody,
                                  bind_f=low_res_bind,
                                  start_antigen = antigen_array, # Original dengue virus antigen sequence
                                  ag_t = viral_target,
                                  ab_at = antibody_antitarget_array,
                                  horizon = 100,
                                  shape_params = default_shape_params)
    
    with open(f"{base_dir}/results/test_run_results.pkl", "wb") as f:
        pkl.dump(final_info, f)