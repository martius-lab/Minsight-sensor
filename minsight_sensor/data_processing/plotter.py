import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

colors__ = [(0.368, 0.507, 0.71), (0.881, 0.611, 0.142),
            (0.56, 0.692, 0.195), (0.923, 0.386, 0.209),
            (0.528, 0.471, 0.701), (0.772, 0.432, 0.102),
            (0.364, 0.619, 0.782), (0.572, 0.586, 0.) ]

grey_skel = (0.8,0.8,0.99)

class Vis:

    def __init__(self, skeleton_surface, indentation_info, col_ind, map_size=40):

        self.skeleton_surface =  skeleton_surface
        self.indentation_info = indentation_info
        self.col_ind = col_ind
        self.map_dim = map_size

    def force_map_vis(self, map, threshold=0.0001):

        fig = plt.figure(figsize=(12.32,12.32))
        map = map.cpu().detach().numpy()
    
        filter_mask = np.ones((self.map_dim,self.map_dim))
        filter_mask[np.where(np.linalg.norm(map, axis=0)<0.0001)]=0
        predict = map * filter_mask
        force = predict.reshape((3,self.map_dim*self.map_dim))
        force[0] = force[0].reshape((self.map_dim,self.map_dim)).flatten()
        force[1] = force[1].reshape((self.map_dim,self.map_dim)).flatten()
        force[2] = force[2].reshape((self.map_dim,self.map_dim)).flatten()
        force = force[:,self.col_ind]
        
        filter_points = np.intersect1d(np.where(np.linalg.norm(force,axis=0)>threshold),np.where(self.indentation_info[:,3]>0))
        
        ax4 = fig.add_subplot(111)
        ax4.scatter(self.skeleton_surface[:,1],self.skeleton_surface[:,2],s=1,color=grey_skel,alpha=0.8)
        ax4.scatter(self.indentation_info[:,1],self.indentation_info[:,2],s=1,color="g")
        ax4.quiver(self.indentation_info[filter_points,1],self.indentation_info[filter_points,2],
            force[0,filter_points]*400.0,force[1,filter_points]*400.0,color=colors__[3])
        
        plt.xlim([-25,25])
        plt.xlabel("X")
        plt.ylim([-25,25])
        plt.ylabel("Y")

        plt.subplots_adjust(0.08,0.08,0.99,0.99)
        fig.savefig("output.png",format="png")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=180)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def force_vec_vis(self, force, threshold=0.0001):

        fig = plt.figure(figsize=(12.32,12.32))

        ax4 = fig.add_subplot(111)
        ax4.scatter(self.skeleton_surface[:,1],self.skeleton_surface[:,2],s=1,color=grey_skel,alpha=0.8)
        ax4.scatter(self.indentation_info[:,1],self.indentation_info[:,2],s=1,color="g")
        ax4.quiver(force[0],force[1],force[3]*10.0,force[4]*10.0, color=colors__[3])

        plt.xlim([-25,25])
        plt.xlabel("X")
        plt.ylim([-25,25])
        plt.ylabel("Y")

        plt.subplots_adjust(0.08,0.08,0.99,0.99)
        fig.savefig("output.png",format="png")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=180)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def force_vec_visyz(self, force, threshold=0.0001):

        fig = plt.figure(figsize=(12.32,12.32))

        ax4 = fig.add_subplot(111)
        ax4.scatter(self.skeleton_surface[:,2],self.skeleton_surface[:,3],s=1,color=grey_skel,alpha=0.8)
        ax4.scatter(self.indentation_info[:,2],self.indentation_info[:,3],s=1,color="g")
        ax4.quiver(force[1],force[2],force[4]*10.0,force[5]*10.0, color=colors__[3])
        
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.subplots_adjust(0.08,0.08,0.99,0.99)

        fig.savefig("output.png",format="png")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=180)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)


    def force_vec_visxz(self, force, threshold=0.0001):

        fig = plt.figure(figsize=(12.32,12.32))
        ax4 = fig.add_subplot(111)
        ax4.scatter(self.skeleton_surface[:,1],self.skeleton_surface[:,3],s=1,color=grey_skel,alpha=0.8)
        ax4.scatter(self.indentation_info[:,1],self.indentation_info[:,3],s=1,color="g")
        ax4.quiver(force[0],force[2],force[3]*10.0,force[5]*10.0, color=colors__[3])
        
        plt.xlabel("X")
        plt.ylabel("Y")

        plt.subplots_adjust(0.08,0.08,0.99,0.99)
        fig.savefig("output.png",format="png")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=180)
        plt.close(fig)
        buf.seek(0)

        return Image.open(buf)