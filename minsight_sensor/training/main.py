#! /usr/bin/python3

import argparse
import sys

import torch
from torch import nn

sys.path.append('.')
from dataset import ForceDataloaders, Postprocessor
from model import get_model
from train import test_model, train_model
from utils import LocalParams, read_json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--training', default=True)
    parser.add_argument('--model_path', default="../trained_model.pt")
    args = parser.parse_args()

    config = read_json("../config.json")
    params = LocalParams(config)

    # use gpu or not
    torch.cuda.empty_cache() 
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # set random seed
    #torch.manual_seed(params.seed)
    #random.seed(params.seed)

    # read data
    dataloaders, dataset_sizes = ForceDataloaders(params)
    postprocessor = Postprocessor(use_gpu, params)

    # get model
    model, optimizer_ft, lr_scheduler = get_model(params)
    print("Training with model {}".format(params.model))

    if use_gpu:
        model = model.cuda()

    # define loss function
    criterion = nn.MSELoss(reduction='mean')

    if args.training==True:
        model, optimizer_ft, lr_scheduler = train_model(params=params,
                                                        model=model,
                                                        criterion=criterion,
                                                        optimizer=optimizer_ft,
                                                        scheduler=lr_scheduler,
                                                        dataloaders=dataloaders,
                                                        dataset_sizes=dataset_sizes,
                                                        postprocessor=postprocessor,
                                                        use_gpu=use_gpu)
        torch.save({'model_weights': model.state_dict(),
                'optimizer_weights': optimizer_ft.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()}, args.model_path)
    else:
        print("Training completed, testing performance")
        if use_gpu:
            checkpoint = torch.load(args.model_path)
        else:
            checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint.get('model_weights'))
        test_model(params, model, dataloaders["test"], postprocessor, use_gpu)





    
