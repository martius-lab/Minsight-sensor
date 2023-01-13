
from __future__ import division, print_function

import numpy as np
from torch.autograd import Variable
from utils import calc_errors_torch


def train_model(params, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, postprocessor, use_gpu):

    for epoch in range(params.total_epochs):

        val_loc_acc = []
        val_force_acc = []

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for i, (input_imgs, subs, inputs, labels) in enumerate(dataloaders[phase]):
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                if type(outputs) is tuple:
                    outputs = outputs[0]
                loss = criterion(outputs, labels)
                

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # statistics
                running_loss += loss.item()
                batch_loss = running_loss / ((i+1)*params.batch_size)
                
                
                if phase == 'train' and i%params.print_freq == 0:
                    if params.force_map == True:                
                        loc_acc, force_acc = calc_errors_torch(postprocessor.convert_map_to_vec(outputs), postprocessor.convert_map_to_vec(labels))
                    else:
                        loc_acc, force_acc = calc_errors_torch(postprocessor.undo_rescale(outputs), postprocessor.undo_rescale(labels))
                
                    print('[Epoch {}/{}]-[batch:{}/{}] lr:{:.4f} {} Loss: {:.6f}  Position Error: {:.4f}  Force Error: {:.4f}'.format(
                            epoch, params.total_epochs - 1, i, round(dataset_sizes[phase]/params.batch_size)-1, scheduler.get_last_lr()[0], phase, batch_loss, loc_acc, force_acc))
                    
                if phase == "val"  and i%params.print_freq == 0:
                    if params.force_map == True:
                        loc_acc, force_acc = calc_errors_torch(postprocessor.convert_map_to_vec(outputs), postprocessor.convert_map_to_vec(labels))
                    else:
                        loc_acc, force_acc = calc_errors_torch(postprocessor.undo_rescale(outputs), postprocessor.undo_rescale(labels))
                    val_loc_acc.append(loc_acc)
                    val_force_acc.append(force_acc)

            if phase == "val":
                print('[Epoch {}/{}] Vaildation Loss: {:.6f}  Position Error: {:.4f}  Force Error: {:.4f} '.format(
                            epoch, params.total_epochs - 1, batch_loss, sum(val_loc_acc)/len(val_loc_acc), sum(val_force_acc)/len(val_force_acc)))


        scheduler.step()
    return model, optimizer, scheduler

    

def test_model(params, model, dataloader, postprocessor, use_gpu):

    model.eval()

    test_loc_acc_abs = []
    test_force_acc_abs = []

    
    for i, (input_imgs, subs, inputs, labels) in enumerate(dataloader):
   
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            input_imgs = input_imgs.cuda()
        else:
            inputs, labels = Variable(inputs), Variable(labels)

   
        outputs = model(inputs)
        if type(outputs) is tuple:
            outputs = outputs[0]

        if params.force_map == True:
            loc_acc_abs, force_acc_abs = calc_errors_torch(postprocessor.convert_map_to_vec(outputs), postprocessor.convert_map_to_vec(labels))

        else:
            loc_acc_abs, force_acc_abs = calc_errors_torch(postprocessor.undo_rescale(outputs), postprocessor.undo_rescale(labels)) 

        test_loc_acc_abs.append(loc_acc_abs)
        test_force_acc_abs.append(force_acc_abs)

    mean_abs_loc_error = sum(np.abs(test_loc_acc_abs))/len(test_loc_acc_abs)
    mean_abs_force_error = sum(np.abs(test_force_acc_abs))/len(test_force_acc_abs)
 
    print("Test result: Localization error: {}, Force error: {}".format(
            mean_abs_loc_error, mean_abs_force_error))


