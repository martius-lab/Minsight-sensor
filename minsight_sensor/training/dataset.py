import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from matplotlib.font_manager import FontProperties
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


def resize(img, scale):
        if scale !=100:
            width = int(img.shape[1] * scale / 100)
            height = int(img.shape[0] * scale / 100)
            dim = (width, height)
            down_size = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            return cv2.resize(down_size, (410,308), interpolation = cv2.INTER_AREA)
        else:
            return img


class ForceDataset(Dataset):

    def __init__(self, root_dir, indices, params):

        self.root_dir = root_dir
        self.image_dir = root_dir + "PostData/"
        self.map_dir = root_dir + params.map_dir
        self.indices = indices
        self.rescale = params.rescale
        self.force_map = params.force_map
        self.scale = params.input_scale

        self.X3_not = cv2.imread(os.path.join(self.image_dir, 'Resized_NoContact.png'))
        self.X3 = TF.to_tensor(self.X3_not)

        self.X4 = TF.to_tensor(resize(cv2.imread((os.path.join(self.image_dir, 'Resized_Position_reference.png'))),self.scale))
        self.max_, self.min_ = self.get_max_min(root_dir)
        

    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir,"Resized"+str(self.indices[idx])+".png")
        X1 = cv2.imread((img_name))
        X1 = resize(X1, self.scale)
        X3 = resize(self.X3_not, self.scale)


        X13 = TF.to_tensor(cv2.subtract(X1,X3))
        X1 = TF.to_tensor(X1)
        if self.force_map:
            X = torch.cat([X13, self.X4[0][None, :, :]], 0)
            sub = torch.cat([X13], 0)
        else:
            X = torch.cat([X13], 0)
            sub = torch.clone(X13)

        if self.force_map:
            label = torch.load(os.path.join(self.map_dir, 'Map2_'+str(self.indices[idx])+'.txt'))
            if self.rescale:
                label = label*4096.0
        else:
            force = np.load(self.image_dir + "Reformulated_force"+str(self.indices[idx])+".npy")
            if self.rescale:
                force = (force-self.min_)/(self.max_-self.min_+1e-10)*1024.0
            force_vec = np.hstack((force[12:15], force[6:9]))
            label = torch.from_numpy(force_vec).float()
        
        return X1, sub, X, label

    @staticmethod
    def get_max_min(dir):
        max_min_dict = np.load(os.path.join(dir,"groundtruth_max_min.npy"), allow_pickle=True).item()
        return max_min_dict.get("max"), max_min_dict.get("min")


def ForceDataloaders(params, num_workers=4):

    train_index = np.load(os.path.join(params.data_path,"01_train_index.npy"))
    test_index = np.load(os.path.join(params.data_path,"01_test_index.npy"))
    val_index = np.load(os.path.join(params.data_path,"01_valid_index.npy"))
  
    print("Train Set: : %i, Test Set: %i, Validation Set: %i" %(len(train_index), len(test_index), len(val_index)))

    train_dataset = ForceDataset(params.data_path, train_index, params)
    train_dataloader = DataLoader(train_dataset, params.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_dataset = ForceDataset(params.data_path, val_index, params)
    val_dataloader = DataLoader(val_dataset, params.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    test_dataset = ForceDataset(params.data_path, test_index, params)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=True, num_workers=num_workers)

    dataset_sizes = {'train': len(train_index), 'val': len(val_index), 'test': len(test_index)}

    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}, dataset_sizes


class Postprocessor():

    def __init__(self, use_gpu, params):
        self.rescale = params.rescale
        self.img_scale = params.input_scale
        root_dir = params.data_path
        self.image_dir = root_dir + "PostData/"
        X3 = resize(cv2.imread(os.path.join(self.image_dir, 'Resized_NoContact.png')), self.img_scale)
        self.X3 = TF.to_tensor(X3)
        self.flat_output = params.flat_output

        self.skeleton_surface = np.load(os.path.join(root_dir,"beam_nodes.npy"))
        self.indentation_info = torch.from_numpy(np.load(os.path.join(root_dir, "indentation_info.npy")))

        if params.force_map == True:
            self.scale = torch.tensor(1.0/4096.0)
            self.const = torch.tensor(0.0)
            self.col_ind = torch.from_numpy(np.load(os.path.join(root_dir + params.map_dir,"column_idx_mapping.npy")))
        else:
            max_, min_ = ForceDataset.get_max_min(root_dir)
            self.max_ = np.hstack((max_[12:15], max_[6:9]))
            self.min_ = np.hstack((min_[12:15], min_[6:9]))
            self.scale = torch.from_numpy(1.0/1024.0*(self.max_-self.min_)).float()
            self.const = torch.from_numpy(self.min_).float()

        if use_gpu:
            self.scale = self.scale.cuda()
            self.const = self.const.cuda()
            self.X3 = self.X3.cuda()
            self.indentation_info = self.indentation_info.cuda()
            
    def undo_rescale(self, output):
        if self.rescale:
            return torch.add(output*self.scale, self.const)
        else:
            return output


    def convert_map_to_vec(self, pred, num_highest_loc=20):
        l = len(pred)
        px = torch.empty(size=(l, 1))
        py = torch.empty(size=(l, 1))
        pz = torch.empty(size=(l, 1))
        fx = torch.empty(size=(l, 1))
        fy = torch.empty(size=(l, 1))
        fz = torch.empty(size=(l, 1))

        if self.flat_output == False:
            for i in range(l):
                map_ = self.undo_rescale(pred[i])

                #localizing the contact by averaging the location of the 20 points with highest force amplitudes
                force = self.indentation_info[torch.argsort(torch.linalg.norm(map_[:,self.col_ind],axis=0), descending=True)[:num_highest_loc],1:4]
                px[i] = torch.mean(force[:,0])
                py[i] = torch.mean(force[:,1])
                pz[i] = torch.mean(force[:,2])
                #Getting the total indentation force by summing over the pixels that have a correspondence to a node of the respective force map channels (this is not exactly reversing the preprocessing)
                fx[i] = torch.sum(map_[0,self.col_ind])
                fy[i] = torch.sum(map_[1,self.col_ind])
                fz[i] = torch.sum(map_[2,self.col_ind])
        else:
            for i in range(l):
                map = self.undo_rescale(pred[i])
                map_ = torch.zeros((3,1350))
                map_ = torch.reshape(map, (3,1350))

                #localizing the contact by averaging the location of the 20 points with highest force amplitudes
                force = self.indentation_info[torch.argsort(torch.linalg.norm(map_[:,:],axis=0), descending=True)[:num_highest_loc],1:4]
                px[i] = torch.mean(force[:,0])
                py[i] = torch.mean(force[:,1])
                pz[i] = torch.mean(force[:,2])
                #Getting the total indentation force by summing over the pixels that have a correspondence to a node of the respective force map channels (this is not exactly reversing the preprocessing)
                fx[i] = torch.sum(map_[0,:])
                fy[i] = torch.sum(map_[1,:])
                fz[i] = torch.sum(map_[2,:])
            
        return torch.hstack((px, py, pz, fx, fy, fz))

    def reconstruct_image(self, image):
        return image+self.X3
        

    def force_map_vis(self, map, threshold=0.0001):

        grey_skel = (0.8,0.8,0.99)
        grey_surf = (0.6,0.6,0.6,0.5)
        arrowscale=0.10
        fs = 28

        font = FontProperties()
        font.set_family('serif')
        font.set_name('Times New Roman')
        font.set_style('italic')
        font.set_size(fs)
        plt.close()
        fig = plt.figure(figsize=(12.32,12.32))

        colors__ = [(0.368, 0.507, 0.71), (0.881, 0.611, 0.142),
            (0.56, 0.692, 0.195), (0.923, 0.386, 0.209),
            (0.528, 0.471, 0.701), (0.772, 0.432, 0.102),
            (0.364, 0.619, 0.782), (0.572, 0.586, 0.) ]  # the last one is a darker green

        map = map.cpu().detach().numpy()

        if self.flat_output == True: 
            filter_mask = np.ones((40,40))
            filter_mask[np.where(np.linalg.norm(map, axis=0)<0.0001)]=0
            predict = map * filter_mask
            force = predict.reshape((3,1600))
        else:
            filter_mask = np.ones((3*1350))
            filter_mask[np.where(np.linalg.norm(map, axis=0)<0.0001)]=0
            predict = map * filter_mask
            force = predict.reshape((3,1350))

        force = force[:,self.col_ind]
        filter_points = np.intersect1d(np.where(np.linalg.norm(force,axis=0)>threshold),np.where(self.indentation_info[:,3]>0))
        
        #3D plot
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.skeleton_surface[:,1],self.skeleton_surface[:,2],self.skeleton_surface[:,3],s=1,color=grey_skel,alpha=0.8)
        ax.scatter(self.indentation_info[:,1],self.indentation_info[:,2],self.indentation_info[:,3],s=1,color=grey_surf)
        ax.quiver(self.indentation_info[filter_points,1],self.indentation_info[filter_points,2],self.indentation_info[filter_points,3],
            force[0,filter_points]*400.0,force[1,filter_points]*400.0,force[2,filter_points]*400.0,color=colors__[3])


        plt.xlim([-25,25])
        plt.xlabel("X")
        plt.ylim([-25,25])
        plt.ylabel("Y")
        ax.zaxis.set_tick_params(labelsize=20)
        ax.view_init(35,-105)
        plt.subplots_adjust(0.08,0.08,0.99,0.99)
        fig.savefig("output.png",format="png")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=180)
        buf.seek(0)
        return Image.open(buf)


def inference(model, params, X1, X3, X4, i=None, input_scale=100, use_gpu=False):

    X1 = resize(X1, input_scale)
    X3 = resize(X3, input_scale)

    X13 = TF.to_tensor(cv2.subtract(X1,X3))

    if params.force_map:
        input = torch.cat([X13, X4[0][None, :, :]], 0)
    else:
        input = torch.cat([X13], 0)

    # wrap them in Variable
    input = Variable(input)
    if use_gpu:
        input = input.cuda()

    # Fix channel conversion issue:
    input = torch.unsqueeze(input,0)

    # forward
    output = model(input)

    return output, X1







