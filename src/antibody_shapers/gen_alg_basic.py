import os
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import antibody_shapers.binding.calculate_binding as binding
import chex


rng = jr.PRNGKey(0)


split_current_dir = os.getcwd().split('/')
package_name = "antibody_shapers"
base_dir = "/".join(split_current_dir[:split_current_dir.index(package_name)+1])

prim_site_antibody = jnp.load(f"{base_dir}/data/2R29_primary_s_antibody.npy")
sec_sites_antibody = jnp.load(f"{base_dir}/data/2R29_secondary_s_antibody.npy")
sec_sites_antigen = jnp.load(f"{base_dir}/data/2R29_secondary_s_antigen.npy")

get_regular_binding = binding.get_binding_function(prim_site_antibody, sec_sites_antibody, sec_sites_antigen)
get_binding_v_antigens = jax.vmap(get_regular_binding, in_axes = (None, 0))
get_binding_v_antibodies = jax.vmap(get_regular_binding, in_axes = (0, None))


regular_binding_extras = binding.get_binding_function_extras(prim_site_antibody, sec_sites_antibody, sec_sites_antigen)
binding_v_antigens_extras = jax.vmap(regular_binding_extras, in_axes = (None, 0))
binding_v_antibodies_extras = jax.vmap(regular_binding_extras, in_axes = (0, None))


def get_reduced_binding(indicies_to_use):
    bind_extras = binding.get_binding_function_extras(
        prim_site_antibody[indicies_to_use], 
        sec_sites_antibody[indicies_to_use], 
        sec_sites_antigen[indicies_to_use]
        )
    return bind_extras

def get_reduced_poses(indicies_to_use):
    prim_ab = prim_site_antibody[indicies_to_use]
    sec_ab = sec_sites_antibody[indicies_to_use]
    sec_ant = sec_sites_antigen[indicies_to_use]
    return {"prim_ab": prim_ab, "sec_ab": sec_ab, "sec_ant": sec_ant}

def get_vmapped_bind_fs(bind_extras):
    v_antigens_e = jax.vmap(bind_extras, in_axes = (None, 0))
    v_antibodies_e = jax.vmap(bind_extras, in_axes = (0, None))
    return v_antigens_e, v_antibodies_e

def mutate(rng, x, p):
    """
    Mutates the antibody/antigen.
    """
    mutation_shape = x.shape
    r1, r2 = jr.split(rng)
    mutation_mask = jr.bernoulli(r1, p, mutation_shape)
    mutation_delta = jr.randint(r2, mutation_shape, minval=1, maxval=20)
    new_pop =  (x + mutation_mask * mutation_delta) % 20
    return new_pop


def get_antibody_fitness(a_gen_target, a_gen_antitarget, target_clip = None, antitarget_clip = None, give_poses = False, bind_func_used = None):
    """
    Gets the antibodies definition of fitness
    """
    if target_clip is None:
        target_clip = (-jnp.inf, jnp.inf)
    if antitarget_clip is None:
        antitarget_clip = (-jnp.inf, jnp.inf)
    if bind_func_used is None:
        bind_func_used = binding_v_antibodies_extras
    def to_return(rng, antibody):
        binding_info_target = bind_func_used(antibody, a_gen_target)
        binding_values_target = binding_info_target["min"]
        
        if a_gen_antitarget is None:
            binding_info_atarget = {"min": 0.0, "argmin": -1}
        else:
            binding_info_atarget = bind_func_used(antibody, a_gen_antitarget)
        binding_values_antitarget = binding_info_atarget["min"]
        
        
        binding_values_target = jnp.clip(binding_values_target, target_clip[0], target_clip[1])
        
        binding_values_antitarget = jnp.clip(binding_values_antitarget, antitarget_clip[0], antitarget_clip[1])
        extra_info = {"binding_values_target": binding_values_target, 
                      "binding_values_antitarget": binding_values_antitarget}
        if give_poses:
            extra_info["poses_target"] = binding_info_target["argmin"]
            extra_info["poses_antitarget"] = binding_info_atarget["argmin"]
        return -1.0*(binding_values_target - binding_values_antitarget), extra_info

    return to_return

def get_antigen_fitness(a_body_target, a_body_antitarget, target_clip = None, antitarget_clip = None, give_poses = False, bind_func_used = None):
    """
    Gets the antigens definition of fitness
    """
    if target_clip is None:
        target_clip = (-jnp.inf, jnp.inf)
    if antitarget_clip is None:
        antitarget_clip = (-jnp.inf, jnp.inf)
    if bind_func_used is None:
        bind_func_used = binding_v_antigens_extras
    def to_return(rng, antigen):
        binding_info_target = bind_func_used(a_body_target, antigen)
        binding_values_target = binding_info_target["min"]
        
        if a_body_antitarget is None:
            binding_info_atarget = {"min": 0.0, "argmin": -1}
        else:
            binding_info_atarget = bind_func_used(a_body_antitarget, antigen)
        binding_values_antitarget = binding_info_atarget["min"]
        
        binding_values_target_clipped = jnp.clip(binding_values_target, target_clip[0], target_clip[1])
        binding_values_antitarget_clipped = jnp.clip(binding_values_antitarget, antitarget_clip[0], antitarget_clip[1])
        extra_info = {"binding_values_target": binding_values_target,
                        "binding_values_antitarget": binding_values_antitarget}
        if give_poses:
            extra_info["poses_target"] = binding_info_target["argmin"]
            extra_info["poses_antitarget"] = binding_info_atarget["argmin"]
        return -1.0*(binding_values_target_clipped - binding_values_antitarget_clipped), extra_info
    return to_return

@chex.dataclass
class GenState:
    population: jnp.ndarray



def get_single_iteration(fitness_func, mutation_func, selection_temperature):
    """
    Gets a single iteration of a genetic algorithm.
    
    Works in a very general way to prevent overlapping code usage.
    
    fitness_func: function that takes a random number generator and a population and returns the fitness of each member
    mutation_func: function that takes a random number generator and a population and returns a mutated population
    selection_temperature: temperature of the softmax selection
    
    In the future it will be useful to add an addition parameter
    specifying extra logging functions, and potentially allowing
    the selection method to vary. For now this should be sufficient.
    """
    
    def single_iteration(state : GenState, rng):
        population = state.population


        r1, r2, r3 = jr.split(rng, 3)
        fitness_values, extra_fit_info = fitness_func(r1, population)
        
        fit_logits = jax.nn.softmax(fitness_values / selection_temperature)
        
        selected_member = jr.choice(r2, population.shape[0], p = fit_logits)


        new_population = population[jnp.ones(population.shape[0], dtype=int) * selected_member]
        new_pop_with_mutation = mutation_func(r3, new_population)
        
        new_state = GenState(population = new_pop_with_mutation)
        
        best_member_extra_info = jax.tree_map(lambda x: x[selected_member], extra_fit_info)
        
        extra_info = {"best_member": population[selected_member], 
                      "best_fitness" : fitness_values[selected_member],
                      "best_member_extra_info": best_member_extra_info,
                      }
        return new_state, extra_info
    
    return single_iteration


def get_iterations_scan(single_iteration):
    def iterations_scan(rng,state, n_iterations):
        return jax.lax.scan(single_iteration, state, jr.split(rng, n_iterations))
    return iterations_scan