# mynotebooks0416/run_exp1_mp.py
import os

# 必须在任何 numpy/scipy/cirq 导入之前进行设置
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

if __name__ == "__main__":
    
    import multiprocessing as mp
    try:
        mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

    from experiments.scripts.experiment1_fid import run_exp1_experiment_mp, save_results_exp1_mp, plot_fidelity_vs_tau_logx
    from experiments.utils import make_run_dir, load_config

    cfg = load_config("exp1_fid.yaml")

    results, meta = run_exp1_experiment_mp(cfg)

    run_dir = make_run_dir(exp_name="experiment1")
    paths = save_results_exp1_mp(results, meta, run_dir,
                             results_filename="results.txt",
                             samples_filename="samples.tsv",
                             circuit_filename="circuit.txt")

    fig_path = paths["results"].parent / "fid_vs_tau_linear.png"
    plot_fidelity_vs_tau_logx(results, title="test", savepath=fig_path)

    print("Saved results to:", paths)
    print("Saved fig to:", fig_path)
    print("Experiment1 Finished.")