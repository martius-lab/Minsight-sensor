import torch
import json
import os


def get_params_dict_for_wandb(params):
    return {
        "epochs": params.total_epochs,
        "batch_size": params.batch_size,
        "lr": params.lr,
        "model": params.model,
    }


def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def write_json(path, dict):
    with open(path, "w") as outfile:
        json.dump(dict, outfile)


class LocalParams:
    def __init__(self, config):
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.model = config["model"]
        self.lr = config["lr"]
        self.working_dir = config["working_dir"]
        self.data_path = config["data_path"]
        self.print_freq = config["print_freq"]
        self.rescale = config["rescale"] == 1
        self.seed = config["seed"]
        self.force_map = config["force_map"] == 1
        self.sensor = config["sensor"]


def load_checkpoint(load_path, model, optim, lr_scheduler, use_gpu):
    """Load all previoulsy saved variables. The program starts clean
    after a resume, so we have to look if a checkpoint file exists in the
    current folder. If not, then we assume the program runs for the first
    time."""

    if os.path.isfile(load_path):
        if use_gpu:
            checkpoint = torch.load(load_path)
        else:
            checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint.get("model_weights"))
        optim.load_state_dict(checkpoint.get("optimizer_weights"))
        lr_scheduler.load_state_dict(checkpoint.get("lr_scheduler"))

    else:
        raise Exception("No checkpoint found in {}".format(load_path))
    return model
