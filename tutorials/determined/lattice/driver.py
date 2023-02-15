import yaml
import time

import numpy as np
import matplotlib.pyplot as plt

from smartredis import Client
from smartsim import Experiment

import determined as det
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from determined.experimental.client import Determined



if __name__=="__main__":
    info = det.get_cluster_info()
    chief_ip = info.container_addrs[0]
    with det.core.init(tensorboard_mode="MANUAL") as context:
        tb_writer = TorchWriter().writer
        exp = Experiment("finite_volume_simulation", launcher="local")
        db = exp.create_database(port=6780, interface="lo")
        exp.start(db)

        # simulation parameters and plot settings
        fig = plt.figure(figsize=(12,6), dpi=80)
        time_steps, seed = 3000, 42

        # define how simulation should be executed
        det_client = Determined()
        with open("fv_sim.yaml", 'rb') as f:
            fv_sim_config = yaml.safe_load(f)
        fv_sim_config["hyperparameters"]["seed"] = seed
        fv_sim_config["hyperparameters"]["steps"] = time_steps
        fv_exp = det_client.create_experiment(
            config=fv_sim_config,
            model_dir="./"
        )

        client = Client(address=f"{chief_ip}:6780", cluster=False)
        # poll until data is available
        client.poll_key("cylinder", 200, 100)
        cylinder = client.get_tensor("cylinder").astype(bool)

        for i in range(0, time_steps, 5): # plot every 5th timestep
            print(f"timestep: {i}")
            while not client.dataset_exists(f"data_{i}"):
                time.sleep(0.01)
            dataset = client.get_dataset(f"data_{i}")
            ux, uy = dataset.get_tensor("ux"), dataset.get_tensor("uy")

            plt.cla()
            ux[cylinder], uy[cylinder] = 0, 0
            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (
                np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)
            )
            vorticity[cylinder] = np.nan
            cmap = plt.cm.get_cmap("bwr").copy()
            cmap.set_bad(color='black')
            plt.imshow(vorticity, cmap=cmap)
            plt.clim(-.1, .1)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            tb_writer.add_figure("fv_sim", fig, i, close=False)
            plt.clf()
            if i % 500==0:
                tb_writer.flush()
                context.train.upload_tensorboard_files()

        exp.stop(db)
