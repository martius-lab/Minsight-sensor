import torch
import numpy as np
import json


def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def write_json(path, dict):
    with open(path, "w") as outfile:
        json.dump(dict, outfile)


def calc_errors_torch(prediction, ground_truth):
    error_pt = torch.linalg.norm(prediction[:, :3] - ground_truth[:, :3], axis=1)
    error_ft = torch.linalg.norm(prediction[:, 3:] - ground_truth[:, 3:], axis=1)
    return torch.mean(error_pt).item(), torch.mean(error_ft).item()


def calc_errors_torch_raw(prediction, ground_truth):
    error_pt = prediction[:, :3] - ground_truth[:, :3]
    error_ft = prediction[:, 3:] - ground_truth[:, 3:]
    return torch.mean(error_pt, 0).tolist(), torch.mean(error_ft, 0).tolist()


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def extract_number(f):
    s = s = re.findall("(\d+).pt", f)
    return (int(s[0]) if s else -1, f)


class LocalParams:
    def __init__(self, config):
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.total_epochs = config["total_epochs"]
        self.model = config["model"]
        self.lr = config["lr"]

        self.data_path = config["data_path"]
        self.print_freq = config["print_freq"]

        self.input_scale = config["input_scale"]
        self.rescale = config["rescale"] == 1

        self.seed = config["seed"]
        self.force_map = config["force_map"] == 1
        if "input_size" in config:
            self.input_size = config["input_size"]
        if "map_dir" in config:
            self.map_dir = config["map_dir"]
        self.flat_output = config["flat_output"] == 1
