import os
import time

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.autograd import Variable
from torch.profiler import ProfilerActivity, profile, record_function
from training.dataset import resize
from training.model import get_model
from training.utils import LocalParams, read_json


def load_input(data_path, i, params):
    img_name = os.path.join(data_path, "PostData/Resized" + str(i) + ".png")
    input_img = cv2.imread(img_name)
    X1 = input_img

    X3 = cv2.imread(data_path + "PostData/Resized_NoContact.png")
    X4 = cv2.imread(data_path + "PostData/Resized_Position_reference.png")
    X1 = resize(X1, params.input_scale)
    X3 = resize(X3, params.input_scale)
    X4 = resize(X4, params.input_scale)

    X13 = TF.to_tensor(cv2.subtract(X1, X3))

    if params.force_map:
        input = torch.cat([X13, X4[0][None, :, :]], 0)
    else:
        input = torch.cat([X13], 0)

    # wrap them in Variable
    input = Variable(input)
    if use_gpu:
        input = input.cuda()

    # Fix channel conversion issue:
    input = torch.unsqueeze(input, 0)
    return input


data_path = "../../Data/training_data_sensor/training_data/"
model_path = "trained_model.pt"

config = read_json("config.json")
params = LocalParams(config)
print("Local Params")
print(params)
params.data_path = data_path
params.model_path = model_path


# use gpu or not
torch.cuda.empty_cache()
use_gpu = torch.cuda.is_available()
print("use_gpu:{}".format(use_gpu))

model, optimizer_ft, exp_lr_scheduler = get_model(params)

if use_gpu:
    checkpoint = torch.load(params.model_path)
else:
    checkpoint = torch.load(params.model_path, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint.get("model_weights"))

if use_gpu:
    model = model.cuda()

print("Testing fully trained model")

model.eval()
model_params = sum(p.numel() for p in model.parameters())

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
    enable_timing=True
)
repetitions = 300
timings = np.zeros((repetitions, 1))

input = load_input(data_path, 100, params)

if use_gpu:
    # GPU-WARM-UP
    for i in range(11):
        # inference(params.force_map)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
        ) as prof:
            with record_function("model_inference"):
                model(input)

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            # inference(params.force_map)
            model(input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean = np.sum(timings[4:]) / repetitions
    std = np.std(timings[4:])
else:
    with torch.no_grad():
        for rep in range(repetitions):
            start = time.time()
            # inference(params.force_map)
            model(input)
            timings[rep] = time.time() - start

    mean = np.sum(timings[4:]) / repetitions
    std = np.std(timings[4:])

print(
    "model: {}, mean_inference_time: {}, std_inference_time: {}, model_params: {}".format(
        params.model,
        mean,
        std,
        model_params,
    )
)
