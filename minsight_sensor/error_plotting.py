import argparse
import os
import time

import cv2
import matplotlib as mtp
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import torch
from tqdm import tqdm
from training.dataset import Postprocessor, inference
from training.model import get_model
from training.utils import (LocalParams, calc_errors_torch,
                            calc_errors_torch_raw, read_json)


def error_calculation(index, i=None, force_map=False):

    loc_errors_total = np.zeros((len(nodes),1)).tolist()
    force_errors_total = np.zeros((len(nodes),1)).tolist()

    loc_errors_3d = []
    force_errors_3d = []
    force_mag = []

    X3 = cv2.imread(data_path + "PostData/Resized_NoContact.png")
    X4 = cv2.imread(os.path.join(data_path, 'PostData/Resized_Position_reference.png'))
    
    for i in tqdm(index):
        X1 = cv2.imread(os.path.join(data_path, "PostData/Resized"+str(i)+".png"))
        force = np.load(data_path + "PostData/Reformulated_force"+str(i)+".npy")
        force_vec = np.hstack((force[12:15], force[6:9]))
        label = torch.from_numpy(force_vec).float()
        label = torch.unsqueeze(label,0)
        if use_gpu:
            label = label.cuda()
        node_num = int(force[1])
        
        if force_map:
            map_label = torch.load(os.path.join(data_path + params.map_dir, 'Map2_'+str(i)+'.txt'))
            map_label = map_label*4096.0
            map_label = torch.unsqueeze(map_label, 0)
            if use_gpu:
                map_label = map_label.cuda()

        output, _ = inference(model, params, X1, X3, X4, i, input_scale=params.input_scale, use_gpu = use_gpu)
        time.sleep(1)
       
        if force_map:
            loc_acc_total, force_acc_total = calc_errors_torch(postprocessor.convert_map_to_vec(output), postprocessor.convert_map_to_vec(map_label))
            loc_acc_raw, force_acc_raw = calc_errors_torch_raw(postprocessor.convert_map_to_vec(output), postprocessor.convert_map_to_vec(map_label))

        else:
            loc_acc_total, force_acc_total = calc_errors_torch(postprocessor.undo_rescale(output), label)
            loc_acc_raw, force_acc_raw = calc_errors_torch_raw(postprocessor.undo_rescale(output), label)

        force_mag.append(np.linalg.norm(force_vec[3:]))
        loc_errors_3d.append(loc_acc_raw)
        force_errors_3d.append(force_acc_raw)

        loc_errors_total[node_num].append(loc_acc_total)
        force_errors_total[node_num].append(force_acc_total)

    return loc_errors_total, force_errors_total, force_mag, loc_errors_3d, force_errors_3d



def plot_total_errors_cont(force_error, limit, nodes, title, path, cmap='YlOrBr'):

    force_error = np.array([sum(err) / len(err) for err in force_error])
    nonzero_indices = force_error.nonzero()
    
    force_error = force_error[nonzero_indices]
    nodes = nodes[nonzero_indices]
    sensing_nodes = np.argwhere(nodes[:,3]>3.0)
    nodes = nodes[sensing_nodes[0]]
    force_error = force_error[sensing_nodes[0]]

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.05, 2]},figsize=(8,3.7))
    x = nodes[:,1]
    y = nodes[:,2]
    z = force_error
    npts = 500
    ngridx = 200
    ngridy = 500
    xi = np.linspace(-20, 20, ngridx)
    yi = np.linspace(-20, 20, ngridy)
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    cm = plt.cm.get_cmap(cmap)

    levels = np.linspace(0.0, limit, 11)

    cntr1 = ax1.contourf(xi, yi, zi, levels=levels, cmap=cm)
    ax1.scatter(skeleton_surface[:,1],skeleton_surface[:,2],s=0.3,c='gray',alpha=1.0)
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    ax1.axis("equal")
    ax1.axis("off")
    x = nodes[:,2]
    y = nodes[:,3]
    z = force_error
    npts = 500
    ngridx = 200
    ngridy = 500
    xi = np.linspace(-20, 20, ngridx)
    yi = np.linspace(-5, 35, ngridy)
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    cm = plt.cm.get_cmap(cmap)

    cntr2 = ax2.contourf(xi, yi, zi, levels=levels, cmap=cm)
    ax2.scatter(skeleton_surface[:,2],skeleton_surface[:,3],s=0.3,c='gray',alpha=1.0)
    plt.xlim(-20,20)
    plt.ylim(-5,35)
    plt.axis("equal")
    plt.axis("off")

    cbar = plt.colorbar(cntr2, fraction=0.025)
    cbar.set_ticks(np.arange(0.0,limit,round(limit/5.0,1)))

    cbar.ax.tick_params(labelsize=20)
    plt.subplots_adjust(0.01,0.01,0.90,0.85,wspace=0)
    plt.title(title,fontsize=25)

    plt.savefig(os.path.join(path,title+".png"), dpi=600)



def plot_3d_errors_force(force_error, force_mag, limit, title, path, cmap="BuPu"):

    force_error = np.squeeze(force_error)
    print("Mean force error: %s" % np.mean(np.linalg.norm(force_error, axis=1)))
    fig, ax = plt.subplots(figsize=(5.0,3.0))

    error_04 = np.array([error for error, mag in zip(force_error, force_mag) if mag <=0.4])
    error_08 = np.array([error for error, mag in zip(force_error, force_mag) if mag <=0.8 and mag > 0.4])
    error_12 = np.array([error for error, mag in zip(force_error, force_mag) if mag >0.8 and mag <=1.2])


    error_x = [error_04[:,0], error_08[:,0], error_12[:,0]]
    error_y = [error_04[:,1], error_08[:,1], error_12[:,1]]
    error_z = [error_04[:,2], error_08[:,2], error_12[:,2]]
    error_abs = [np.linalg.norm(error_04, axis=1), np.linalg.norm(error_08, axis=1), np.linalg.norm(error_12, axis=1)]

    labels = ["0-0.4", "0.4-0.8", "0.8-1.2"]

    cmap = mtp.cm.get_cmap(cmap)
    vp1 = ax.violinplot(error_x, points=100, positions=np.arange(0.2, len(error_x)+0.2),
                    showmeans=False, showmedians=False, showextrema=False)
    
    for b in vp1['bodies']:
        b.set_alpha(0.9)
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_color(cmap(0.3))

    vp2 = ax.violinplot(error_y, points=100, positions=np.arange(0.1, len(error_y)+0.1),
                    showmeans=False, showmedians=False, showextrema=False)
    
    for b in vp2['bodies']:
        b.set_alpha(0.9)
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_color(cmap(0.5))
    
    vp3 = ax.violinplot(error_z, points=400, positions=np.arange(0.0, len(error_z)),
                    showmeans=False, showmedians=False, showextrema=False)
    
    for b in vp3['bodies']:
        b.set_alpha(0.9)
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_color(cmap(0.7))

    vp4 = ax.violinplot(error_abs, points=400, positions=np.arange(-0.1, len(error_abs)-0.1),
                    showmeans=False, showmedians=False, showextrema=False)
    
    for b in vp4['bodies']:
        b.set_alpha(0.9)
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        b.set_color(cmap(0.9))

    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(0, len(labels)))
    ax.set_xticklabels(labels)
    #ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel('Force Magnitudes[N]')
    ax.set_ylabel('Force error [N]')
    plt.legend([vp4['bodies'][0],vp3['bodies'][0],vp2['bodies'][0],vp1['bodies'][0]], ['Total\nforce', 'Shear\nforce 1', 'Shear\nforce 2', 'Normal\nforce'], ncol=4, loc='lower left', prop={'size': 8})

    #plt.title(title,fontsize=25)
    #plt.show()
    plt.savefig(os.path.join(path,title+".png"),  bbox_inches='tight', dpi=600)

def plot_3d_errors_position(force_error, force_mag, limit, title, path, cmap='BuPu'):

    force_error = np.squeeze(force_error)
    print("Mean position error: %s" % np.mean(np.linalg.norm(force_error, axis=1)))
    fig, ax = plt.subplots(figsize=(5.0,2.0))

    error_04 = np.array([error for error, mag in zip(force_error, force_mag) if mag <=0.4])
    error_08 = np.array([error for error, mag in zip(force_error, force_mag) if mag <=0.8 and mag > 0.4])
    error_12 = np.array([error for error, mag in zip(force_error, force_mag) if mag >0.8 and mag <=1.2])
    error_1up = np.array([error for error, mag in zip(force_error, force_mag) if mag >1.0])

    error_abs = [np.linalg.norm(error_04, axis=1), np.linalg.norm(error_08, axis=1), np.linalg.norm(error_12, axis=1)]

    labels = ["0-0.4", "0.4-0.8", "0.8-1.2"]

    vp4 = ax.violinplot(error_abs, points=400, positions=np.arange(0, len(error_abs)),
                    showmeans=False, showmedians=False, showextrema=False)
    
    cmap = mtp.cm.get_cmap(cmap)

    for b in vp4['bodies']:
        b.set_alpha(0.9)
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_color(cmap(0.5))

    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(0, len(labels)))
    ax.set_xticklabels(labels)

    ax.set_ylim(0, limit)
    ax.set_xlabel('Force Magnitudes [N]')
    ax.set_ylabel('Position error [mm]')

    plt.savefig(os.path.join(path,title+".png"), bbox_inches='tight', dpi=600)


def process_data(error_list):
    error_store = []
    for node in error_list:
        if len(node)>1:
            error_store.append(np.mean(node[1:]))
    return error_store


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--calc', default=True)
    args = parser.parse_args()

    result_path = "results/"
    data_path = "../../Data/training_data_sensor/training_data/"
    nodes = torch.from_numpy(np.load(os.path.join(data_path, "indentation_info.npy")))
    skeleton_surface = np.squeeze(np.load(os.path.join(data_path,"beam_nodes.npy")))

    if args.calc == True:
        model_path = "trained_model.pt"
        config = read_json("config.json")
        os.makedirs(result_path, exist_ok=True)

        test_index = np.load(os.path.join(data_path,"01_test_index.npy"))
    
        params = LocalParams(config)
        params.data_path = data_path
        params.model_path = model_path

        # use gpu or not
        torch.cuda.empty_cache() 
        use_gpu = torch.cuda.is_available()
        print("use_gpu:{}".format(use_gpu))

        model, optimizer_ft, exp_lr_scheduler = get_model(params)

        postprocessor = Postprocessor(use_gpu, params)

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint.get('model_weights'))

        if use_gpu:
            model = model.cuda()
        model.eval()

        loc_errors_total, force_errors_total, force_mag, loc_errors_3d, force_errors_3d = error_calculation(test_index, force_map=params.force_map)

        np.save(os.path.join(result_path, "loc_accuracies.npy"),[loc_errors_total, loc_errors_3d] )
        np.save(os.path.join(result_path, "force_accuracies.npy"),[force_errors_total, force_errors_3d])
        np.save(os.path.join(result_path, "force_magnitudes.npy"),force_mag)

    else:
        loc_errors = np.load(os.path.join(result_path, "loc_accuracies.npy"), allow_pickle=True)
        force_errors = np.load(os.path.join(result_path, "force_accuracies.npy"), allow_pickle=True)
        force_mag = np.load(os.path.join(result_path, "force_magnitudes.npy"), allow_pickle=True)

    total_loc_errors = process_data(loc_errors[0])
    total_force_erros = process_data(force_errors[0])

    plot_total_errors_cont(np.array(force_errors[0],dtype=object), 0.5, nodes[nodes[:, 0].argsort()],"Force Error [N]", result_path, cmap='YlOrBr')
    plot_total_errors_cont(np.array(loc_errors[0],dtype=object), 5.0, nodes[nodes[:, 0].argsort()],"Position Error [mm]", result_path, cmap='BuPu')

    plot_3d_errors_position(np.array([np.array(err) for err in np.array(loc_errors[1:])]),force_mag, 5.0, "Position Error Distribution multi [mm]", result_path, cmap='BuPu')
    plot_3d_errors_force(np.array([np.array(err) for err in np.array(force_errors[1:])]), force_mag, 0.5, "Force Error Distribution multi [N]", result_path,cmap='YlOrBr')

