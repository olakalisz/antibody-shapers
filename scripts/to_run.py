import scripts.transfer.main_shaping_process as process
import jax.random as jr
import vshaping.shaping_funcs as sf


if __name__ == "__main__":
    antigen_antitarget = sf.antibody_antitarget_array

    seed = 10201

    rng = jr.PRNGKey(seed)
    
    process.experiment_process(rng, antigen_antitarget, runtime_hours=17, results_dir="lots_of_accuracy_range")
    # process.get_verification_process(rng, antigen_antitarget, num_seeds=3, 
    #                              exp_num=4, seed=seed, random_start_antigen = True,
    #                              read_bind_f=True)
