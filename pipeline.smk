configfile: "config.yaml"


rule generate_data:
    output: "{data_dir}}/data_80_{idx}"
    params:
        seq_len=config["seq_len"],
        path_to_meme=config["path_to_meme"],
        pairs_file=config["pairs_file"],
        num_of_seq=config["num_of_seq"]
        data_dir = config["data_dir"]
    shell:
        "python -c 'from script import SimulatedDataGenerator; "
        "SimulatedDataGenerator(seq_len={params.seq_len}, "
        "path_to_meme={params.path_to_meme}, pairs_file={params.pairs_file}, "
        "num_of_seq={params.num_of_seq}, output_name={output}).generate_data()'"

rule model_selection:
    input: expand("data/data_80_{idx}", idx=range(1, config["data_instances"]+1))
    output: "results/model_selection/best_model.json"
    params:
        dataset_path="data/data_80_0",
        out_dir="results/model_selection/",
        n_iter=30
    shell:
        "python -c 'from script import Calibration; "
        "Calibration(dataset_path={params.dataset_path}, numLabels=2, "
        "seq_length={config[seq_len]}, device=\"cuda\", search_space_dict={config[param_dist]}, "
        "out_dir={params.out_dir}, n_iter={params.n_iter})'"

rule train_model:
    input:
        model_config="results/model_selection/best_model.json",
        dataset="data/{seq}/data_80_{idx}"
    output:
        "results/baseline/data_80_{idx}/run_{j}/experiment.log"
    params:
        seed=555
    shell:
        "python -c 'from script import run_experiment, Config, setup_seed; "
        "setup_seed({params.seed}); arg_space = Config.from_json(\"modelsparams/train_config.json\"); "
        "params_dict = load_config({input.model_config}); "
        "arg_space.inputprefix={input.dataset}; "
        "arg_space.directory=\"results/baseline/data_80_{idx}/run_{j}\"; "
        "run_experiment(\"cuda\", arg_space, params_dict)'"
