import copy

from smartredis import Client
from smartsim import Experiment

import determined as det
from determined.experimental.client import Determined

default_config = {
    "name": None,
    "entrypoint": None,
    "resources": {
        "slots_per_trial": 1,
    },
    "environment": {
        "image": None,
        "environment_variables": None,
        "force_pull_image": False
    },
    "searcher": {
        "name": "single",
        "metric": "my_metric",
        "max_length": 1
    }
}

def create_basic_determined_config(
    name,
    entrypoint,
    n_gpus=1,
    env_vars=None,
    metric="my_metric",
    docker_image="liamdetermined/development:smartsim"
):
    config = copy.deepcopy(default_config)
    config["name"] = name
    config["entrypoint"] = entrypoint
    config["resources"]["slots_per_trial"] = n_gpus
    config["searcher"]["metric"] = metric
    config["environment"]["image"] = docker_image
    config["environment"]["environment_variables"] = [f"{k}={v}" for k,v in env_vars.items()]
    return config

if __name__=="__main__":
    info = det.get_cluster_info()
    chief_ip = info.container_addrs[0]

    exp = Experiment("surrogate_training", launcher="local")
    db = exp.create_database(port=6780, interface="lo")
    exp.start(db)

    # Launch simulation and training jobs
    det_client = Determined()
    steps = 100
    size =  64
    sim_config = create_basic_determined_config(
        name="fd_simulation",
        entrypoint=f"python fd_sim.py --steps={steps} --size={size}",
        env_vars={
            "OMP_NUM_THREADS": "8",
            "SSKEYOUT": "fd_simulation",
            "SSDB": f"{chief_ip}:6780"
        }
    )
    sim_exp = det_client.create_experiment(
        config=sim_config,
        model_dir="./"
    )

    nn_depth = 4
    epochs = 40
    ml_config = create_basic_determined_config(
        name="tf_training",
        entrypoint=f"python tf_training.py --depth={nn_depth} --epochs={epochs} --size={size}",
        env_vars = {
            "OMP_NUM_THREADS": "16",
            "SSDB": f"{chief_ip}:6780",
            "SSKEYIN": "fd_simulation"
        }
    )

    ml_exp = det_client.create_experiment(
        config=ml_config,
        model_dir="./"
    )
    ml_exp.wait()

    exp.stop(db)
