import jax
import jax.numpy as jnp
import jax.random as jr
import pickle as pkl

import numpy as np

import time
import datetime
import pickle as pkl

import antibody_shapers.gen_alg_basic as ga
import antibody_shapers.shaping_funcs as sf
import antibody_shapers.utils as utils
import os
import chex
import datetime

rng = jr.PRNGKey(0)

antibody_pop_size = 40


def get_meta_shaping_iter(mutation_function, eval_function):
    """
    Returns a single meta shaping iteration.
    Designed to work for the shaping algorithms!
    """

    def single_iteration_meta(current_antibody, rng):
        r1, r2 = jr.split(rng)
        new_antibodies = mutation_function(r1, current_antibody, antibody_pop_size)
        eval_results, extra_information = eval_function(r2, new_antibodies)

        best_index = jnp.argmin(eval_results)
        next_antibody = new_antibodies[best_index]

        extra_info = {
            "all_information": extra_information,
            "best_index": best_index,
            "antibody": next_antibody,
        }
        return next_antibody, extra_info

    return single_iteration_meta


def mutate_onemut(rng, antibody, reps):
    r1, r2 = jr.split(rng)
    mut_locations = jr.randint(r1, (reps,), 0, antibody.shape[0])
    mut_mask = jax.nn.one_hot(mut_locations, antibody.shape[0], dtype=jnp.int32)
    mut_delta = jr.randint(r2, (reps, antibody.shape[0]), 1, 20)
    new_antibodies = (antibody + mut_mask * mut_delta) % 20
    return new_antibodies


def mutate_with_base(rng, antibody, reps):
    new_antibodies = mutate_onemut(rng, antibody, reps - 1)
    new_antibodies = jnp.vstack([new_antibodies, antibody])
    return new_antibodies


def get_proper_ab_performance_sample(
    horizon_len, bind_func, start_antigen, ag_t, ab_at, number_reps=10
):
    """
    Returns a function that performs proper meta shaping iteration.

    Args:
        horizon_len (int): The length of the horizon.
        bind_func (function): The binding function.
        start_antigen (array): The starting antigen. If None, a random one is generated.
        ag_t (float): The antigen target.
        ab_at (float): The antibody antitarget.
        number_reps (int, optional): The number of repetitions. Defaults to 10.
    Returns:
        function (rng, antibodies -> (agg_fitness, extra_data)): A function that performs proper a batch of performance evals and returns too much data :))
    """

    def single_thing(rng, antibody):
        r_run, r_other = jr.split(rng)
        if start_antigen is None:
            ag_to_use = jr.randint(r_other, sf.antigen_array.shape, 0, 20)
        else:
            ag_to_use = start_antigen
        resultData = sf.single_shape_run(
            r_run, antibody, bind_func, ag_to_use, ag_t, ab_at, horizon_len
        )
        antigen_performance = resultData["ag_performances"].mean()
        return antigen_performance, resultData

    many_reps = jax.vmap(single_thing, in_axes=(0, None))

    def to_return(rng, antibodies):
        rngs = jr.split(rng, number_reps)
        results_scores, results_extras = many_reps(rngs, antibodies)
        return results_scores.mean(), results_extras

    many_abs = jax.vmap(to_return, in_axes=(None, 0))

    return many_abs


def get_binding_indicies(rng, antigen):
    """
    Get the binding indices of antibodies to an antigen.

    Args:
        rng (numpy.random.Generator): Random number generator.
        antigen (numpy.ndarray): Array representing the antigen.

    Returns:
        numpy.ndarray: Array of filtered binding indices.
    """

    def get_bind_index(antibody):
        return sf.med_res_bind(antibody, antigen)["argmin"]

    get_lots_bind_index = jax.jit(jax.vmap(get_bind_index))

    lots_bind_index = get_lots_bind_index(jr.randint(rng, (100000, 11), 0, 20))

    bin_counts = np.bincount(lots_bind_index, minlength=20)
    argsort_bincounts = np.argsort(bin_counts)[::-1]

    sorted_bin_count = bin_counts[argsort_bincounts]
    final_bound_value = np.argmin(sorted_bin_count) - 1

    new_good_inds = sf.medium_indicies[argsort_bincounts[:final_bound_value]]

    intersecting_indicies = np.any(new_good_inds[:, None] == sf.top_indicies, axis=1)

    filtered_inds = new_good_inds[~intersecting_indicies]

    return filtered_inds


def get_binding_func_for_antigen(rng, antigen):
    """
    Get the binding function for an antigen.

    Args:
        rng (numpy.random.Generator): Random number generator.
        antigen (numpy.ndarray): Array representing the antigen.

    Returns:
        function: Binding function.
    """
    binding_indicies = get_binding_indicies(rng, antigen)
    all_indicies = jnp.concatenate([binding_indicies, sf.top_indicies])
    binding_func = ga.get_reduced_binding(all_indicies)
    return {"function": binding_func, "indicies": all_indicies}


def get_useful_data_single(data_dict):
    """
    Gets the important, useful information from a single run.
    """
    ags_found = data_dict["run_info"]["best_member"]

    def bin_c(x):
        return jnp.bincount(x, length=20)

    ag_val_dist = jax.vmap(bin_c)(ags_found)
    binding_prim = data_dict["run_info"]["best_member_extra_info"][
        "binding_values_antitarget"
    ]
    binding_t = data_dict["run_info"]["best_member_extra_info"]["binding_values_target"]
    binding_at = data_dict["binding_penalty"]
    total_fit = data_dict["run_info"]["best_fitness"] - binding_at
    to_return = {
        "ags_found": ag_val_dist,
        "total_fit": total_fit,
        "binding_primary": binding_prim,
        "binding_ag_t": binding_t,
        "binding_ab_at": binding_at,
    }
    return to_return


def useful_data_mreps(data_dict):
    split_data = jax.vmap(get_useful_data_single)(data_dict)
    mean_info = jax.tree_util.tree_map(lambda x: x.mean(axis=0), split_data)
    var_info = jax.tree_util.tree_map(lambda x: x.var(axis=0), split_data)
    return {"mean_info": mean_info, "var_info": var_info}


def useful_data_antibody_aware(data_dict_full):
    correct_index = data_dict_full["best_index"]
    data_dict = jax.tree_util.tree_map(
        lambda x: x[correct_index], data_dict_full["all_information"]
    )
    to_return = useful_data_mreps(data_dict)
    return to_return


def useful_data_full(data_dict_full):
    vmap_meta_steps = jax.vmap(useful_data_antibody_aware)
    vmap_reps = jax.vmap(vmap_meta_steps)
    info_data = vmap_reps(data_dict_full)
    info_data["antibody"] = data_dict_full["antibody"]
    return info_data


def single_test():

    rng = jr.PRNGKey(0)

    start_ab = jr.randint(rng, (11,), 0, 20)

    antigen_antitarget = jr.randint(rng, sf.antibody_antitarget_array.shape, 0, 20)

    bind_data = get_binding_func_for_antigen(rng, antigen_antitarget)
    bind_function = bind_data["function"]
    bind_inds = bind_data["indicies"]

    sample_func = get_proper_ab_performance_sample(
        10,
        bind_function,
        sf.antigen_array,
        sf.viral_target,
        sf.antibody_antitarget_array,
    )

    first_ab_info = sample_func(rng, jnp.array([start_ab]))
    first_ab_info = jax.tree_util.tree_map(lambda x: x[0], first_ab_info)

    single_iter = jax.jit(get_meta_shaping_iter(mutate_with_base, sample_func))

    def get_meta_results(rng):
        return jax.lax.scan(single_iter, start_ab, jr.split(rng, 300))

    j_get_meta_results = jax.jit(jax.vmap(get_meta_results))

    num_seeds = 10
    results = j_get_meta_results(jr.split(rng, num_seeds))

    formatted_results = useful_data_full(results[1])
    first_ab_results = useful_data_mreps(first_ab_info[1])

    to_return = {
        "first_ab_res": first_ab_results,
        "meta_res": formatted_results,
        "start_ab": start_ab,
        "rng": rng,
    }
    return to_return


def single_run(
    rng,
    start_ab,
    antigen_antitarget,
    bind_f,
    num_seeds=10,
    horizon_len=10,
    reps_for_meta_samples=10,
):
    sample_func = get_proper_ab_performance_sample(
        horizon_len,
        bind_f,
        sf.antigen_array,
        sf.viral_target,
        antigen_antitarget,
        number_reps=reps_for_meta_samples,
    )

    first_ab_info = sample_func(rng, jnp.array([start_ab]))
    first_ab_info = jax.tree_util.tree_map(lambda x: x[0], first_ab_info)

    single_iter = jax.jit(get_meta_shaping_iter(mutate_with_base, sample_func))

    def get_meta_results(rng):
        return jax.lax.scan(
            single_iter, start_ab, jr.split(rng, 30 * 100 // horizon_len)
        )

    j_get_meta_results = jax.jit(jax.vmap(get_meta_results))

    results = j_get_meta_results(jr.split(rng, num_seeds))

    formatted_results = useful_data_full(results[1])
    first_ab_results = useful_data_mreps(first_ab_info[1])

    to_return = {
        "first_ab_res": first_ab_results,
        "meta_res": formatted_results,
        "start_ab": start_ab,
        "rng": rng,
    }
    return to_return


def single_ab_verification(
    rng,
    antibody,
    antigen_antitarget,
    num_seeds=10,
    horizon_len=100,
    binding_fn=None,
    random_start_antigen=False,
):
    if binding_fn is None:
        binding_fn = sf.med_res_bind

    rng_run, rng_other = jr.split(rng)

    if random_start_antigen:
        start_ag = None
    else:
        start_ag = sf.antigen_array

    sample_func = get_proper_ab_performance_sample(
        horizon_len,
        binding_fn,
        start_ag,
        sf.viral_target,
        antigen_antitarget,
        number_reps=num_seeds,
    )

    results = sample_func(rng_run, jnp.array([antibody]))
    results = jax.tree_util.tree_map(lambda x: x[0], results)

    formatted_res = useful_data_mreps(results[1])
    return formatted_res


def get_sensible_subsample(max_val, num_samples, min_val=1):
    lowed_bound_list = jnp.arange(num_samples) + min_val
    log_space_list = jnp.logspace(
        jnp.log10(min_val), jnp.log10(max_val), num_samples, dtype=jnp.int32
    )
    log_space_list = log_space_list.at[0].set(min_val)
    log_space_list = log_space_list.at[-1].set(max_val)
    return jnp.maximum(lowed_bound_list, log_space_list)


def verify_many_ab(
    rng,
    antibodies,
    antigen_antitarget,
    num_subsamples=30,
    num_seeds=10,
    horizon_len=100,
    binding_fn=None,
    random_start_antigen=False,
):
    num_abs = antibodies.shape[0]

    inds_to_try = get_sensible_subsample(num_abs, num_subsamples) - 1

    abs_to_try = antibodies[inds_to_try]

    def try_single_ab(ab):
        return single_ab_verification(
            rng,
            ab,
            antigen_antitarget,
            num_seeds,
            horizon_len,
            binding_fn,
            random_start_antigen=random_start_antigen,
        )

    vmapped_over_abs = jax.vmap(try_single_ab)

    many_ab_results = vmapped_over_abs(abs_to_try)

    return many_ab_results


def verification_process(
    rng,
    data_output,
    antigen_antitarget,
    num_subsamples=30,
    num_seeds=100,
    horizon_len=100,
    abs_per_run=100,
    binding_fn=None,
    random_start_antigen=False,
):

    antibodies = data_output["meta_res"]["antibody"]
    antibodies_r = antibodies.reshape((-1,) + antibodies.shape[-2:])

    def single_run_verification(single_run_antibodies):
        return verify_many_ab(
            rng,
            single_run_antibodies,
            antigen_antitarget,
            num_subsamples,
            num_seeds,
            horizon_len,
            binding_fn=binding_fn,
            random_start_antigen=random_start_antigen,
        )

    inds_to_try = get_sensible_subsample(antibodies.shape[-2], num_subsamples) - 1

    list_of_results = []
    for i in range(0, antibodies_r.shape[0], abs_per_run):
        happy_results = jax.jit(jax.vmap(single_run_verification))(
            antibodies_r[i : i + abs_per_run]
        )

        list_of_results.append(jax.tree.map(lambda x: np.array(x), happy_results))

    many_ab_results_r = jax.tree.map(
        lambda *x: jnp.concatenate(x, axis=0), *list_of_results
    )
    many_ab_results = jax.tree.map(
        lambda x: x.reshape(antibodies.shape[:-2] + x.shape[1:]), many_ab_results_r
    )
    many_ab_results["antibody"] = antibodies
    many_ab_results["verification_idx"] = inds_to_try

    start_antibodies = data_output["start_ab"]

    def try_single_ab(ab):
        return single_ab_verification(
            rng,
            ab,
            antigen_antitarget,
            num_seeds,
            horizon_len,
            binding_fn=binding_fn,
            random_start_antigen=random_start_antigen,
        )

    vmapped_over_abs = jax.jit(jax.vmap(try_single_ab))
    start_ab_results = vmapped_over_abs(start_antibodies)

    to_return = {
        "first_ab_res_ver": start_ab_results,
        "meta_res_ver": many_ab_results,
        "start_ab": start_antibodies,
    }

    return to_return


@chex.dataclass(frozen=True)
class HyperParams:
    horizon_len: int
    num_seeds: int
    num_meta_samps: int


def experiment_process(
    rng,
    antigen_antitarget,
    num_seeds=60,
    runtime_hours=17,
    results_dir="experiment_results",
):
    """
    The main experiment process.
    """

    bind_info = get_binding_func_for_antigen(rng, antigen_antitarget)
    bind_f = bind_info["function"]
    bind_indicies = np.array(bind_info["indicies"])

    num_seeds_per_ab = 5

    def get_single_data_oneab(rng, extras: HyperParams):
        r1, r2 = jr.split(rng)
        start_ab = jr.randint(r1, (11,), 0, 20)
        horizon_len = extras.horizon_len
        reps_for_meta_samples = extras.num_meta_samps
        return single_run(
            r2,
            start_ab,
            antigen_antitarget,
            bind_f,
            num_seeds_per_ab,
            horizon_len,
            reps_for_meta_samples,
        )

    def get_single_data(rng, extras: HyperParams):
        num_things = extras.num_seeds // num_seeds_per_ab
        rngs = jr.split(rng, num_things)
        return jax.vmap(get_single_data_oneab, in_axes=(0, None))(rngs, extras)

    jaxy_single_data = jax.jit(get_single_data, static_argnums=(1,))

    start_time = datetime.datetime.now()
    print(f"Starting time: {start_time}")

    max_time = datetime.timedelta(hours=runtime_hours)

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    exp_num = 0
    while os.path.exists(f"{results_dir}/exp_{exp_num}"):
        exp_num += 1

    base_dir = f"{results_dir}/exp_{exp_num}"

    os.mkdir(base_dir)

    horizon_lens = [100, 20, 10, 5, 1]
    possible_num_meta_samps = [5, 10, 40]

    rep_number = 0
    rng = jr.PRNGKey(exp_num) + rng

    useful_run_info = {
        "Binding Indicies": bind_indicies,
        "Antigen Antitarget": antigen_antitarget,
        "Num Seeds": num_seeds,
        "Horizon Lengths": horizon_lens,
        "rng": rng,
    }

    with open(f"{base_dir}/useful_run_info.pkl", "wb") as f:
        pkl.dump(useful_run_info, f)

    r1, r2 = jr.split(rng)
    while datetime.datetime.now() - start_time < max_time:
        for horizon_len in horizon_lens:
            if horizon_len == 1:
                n_m_samps = 1
                imp_info = {
                    "horizon_len": horizon_len,
                    "num_seeds": num_seeds * 10,
                    "num_meta_samps": n_m_samps,
                }
                imp_info = HyperParams(**imp_info)
                data_output = jaxy_single_data(r1, imp_info)
                with open(
                    f"{base_dir}/rep_{rep_number}_horizon_{horizon_len}_msamps_{n_m_samps}.pkl",
                    "wb",
                ) as f:
                    pkl.dump(data_output, f)
                    print(f"Current time: {datetime.datetime.now()}")
            else:
                for n_m_samps in possible_num_meta_samps:
                    imp_info = {
                        "horizon_len": horizon_len,
                        "num_seeds": num_seeds * 10 // n_m_samps,
                        "num_meta_samps": n_m_samps,
                    }
                    imp_info = HyperParams(**imp_info)
                    data_output = jaxy_single_data(r1, imp_info)
                    with open(
                        f"{base_dir}/rep_{rep_number}_horizon_{horizon_len}_msamps_{n_m_samps}.pkl",
                        "wb",
                    ) as f:
                        pkl.dump(data_output, f)
                        print(f"Current time: {datetime.datetime.now()}")
        r1, r2 = jr.split(r2)
        rep_number += 1
        print(f"Finished rep number: {rep_number}")
        print(f"Current time: {datetime.datetime.now()}")


def get_verification_process(
    rng,
    antigen_antitarget,
    num_seeds=10,
    exp_num=4,
    seed=0,
    random_start_antigen=False,
    read_bind_f=False,
    results_dir="experiment_results",
):
    start_time = datetime.datetime.now()
    print(f"Starting time: {start_time}")

    base_dir = f"{results_dir}/exp_{exp_num}"
    if read_bind_f:
        verification_folder = f"verification_low_res_{seed}"
    else:
        verification_folder = f"verification_med_res_{seed}"

    verification_dir = f"{base_dir}/{verification_folder}"

    if not os.path.exists(base_dir):
        raise ValueError(f"No experiment directory found {base_dir}")

    if not os.path.exists(f"{verification_dir}"):
        os.mkdir(f"{verification_dir}")
    else:
        raise ValueError("Verification directory already exists")

    ## Attempt to load the extra info, and the binding params in particular.
    if not os.path.exists(f"{base_dir}/useful_run_info.pkl"):
        bind_f = None
    else:
        with open(f"{base_dir}/useful_run_info.pkl", "rb") as f:
            useful_run_info = pkl.load(f)
        bind_indicies = useful_run_info["Binding Indicies"]
        if read_bind_f:
            bind_f = ga.get_reduced_binding(bind_indicies)
        else:
            bind_f = None

    if bind_f is None and read_bind_f:
        bind_f = get_binding_func_for_antigen(rng, antigen_antitarget)["function"]

    # Iterate over the data in base_dir
    for file in os.listdir(base_dir):
        if file.endswith(".pkl") and not file.startswith("useful_run_info"):
            with open(f"{base_dir}/{file}", "rb") as f:
                data_output = pkl.load(f)
                verification_data = verification_process(
                    rng,
                    data_output,
                    antigen_antitarget,
                    num_seeds=num_seeds,
                    abs_per_run=250,
                    binding_fn=bind_f,
                    random_start_antigen=random_start_antigen,
                )
                with open(f"{verification_dir}/ver_{file}", "wb") as f:
                    pkl.dump(verification_data, f)
                    print(f"Finished verification for {file}")
                    print(f"Current time: {datetime.datetime.now()}")


if __name__ == "__main__":
    antigen_antitarget = sf.antibody_antitarget_array
    different_antigen_antitarget = sf.different_antibody_antitarget_array

    seed = 0

    rng = jr.PRNGKey(seed)

    experiment_process(
        rng, antigen_antitarget, runtime_hours=17, results_dir="lots_of_accuracy_range"
    )
