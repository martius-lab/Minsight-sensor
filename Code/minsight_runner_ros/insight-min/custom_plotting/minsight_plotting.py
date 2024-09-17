import torch
import os
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import io


class Plotter:

    def __init__(self, data_path, name, mode="xyz", ret=False):

        # Colors for plotting
        self.colors_ = [
            (0.368, 0.507, 0.71),
            (0.881, 0.611, 0.142),
            (0.56, 0.692, 0.195),
            (0.923, 0.386, 0.209),
            (0.528, 0.471, 0.701),
            (0.772, 0.432, 0.102),
            (0.364, 0.619, 0.782),
            (0.572, 0.586, 0.0),
        ]  # the last one is a darker green

        self.mode = mode
        self.fig = plt.figure(figsize=(10.62, 10.62))
        self.fig.suptitle(name)
        self.indentation_info = torch.from_numpy(
            np.load(os.path.join(data_path, "indentation_info.npy"))
        )
        self.col_ind = torch.from_numpy(
            np.load(os.path.join(data_path, "column_idx_mapping_post.npy"))
        )
        self.skeleton_surface = np.squeeze(
            np.load(os.path.join(data_path, "beam_nodes.npy"))
        )

        if mode == "xyz":
            self.ax = self.fig.add_subplot(111, projection="3d")
            self.ax.scatter(
                self.skeleton_surface[:, 1],
                self.skeleton_surface[:, 2],
                self.skeleton_surface[:, 3],
                s=1,
                alpha=0.2,
            )
            self.ax.scatter(
                self.indentation_info[:, 2],
                self.indentation_info[:, 1],
                self.indentation_info[:, 3],
                s=1,
                color="gray",
            )
            self.ax.set_xlim([-25, 25])
            self.ax.set_xlabel("X")
            self.ax.set_ylim([-25, 25])
            self.ax.set_ylabel("Y")
        elif mode == "xy":
            self.ax = self.fig.add_subplot(111)
            self.ax.scatter(
                self.skeleton_surface[:, 1], self.skeleton_surface[:, 2], s=1, alpha=0.2
            )
            self.ax.scatter(
                self.indentation_info[:, 1],
                self.indentation_info[:, 2],
                s=1,
                color="gray",
            )
            self.ax.set_xlim([-25, 25])
            self.ax.set_xlabel("X")
            self.ax.set_ylim([-25, 25])
            self.ax.set_ylabel("Y")
        elif mode == "yz":
            self.ax = self.fig.add_subplot(111)
            self.ax.scatter(
                self.skeleton_surface[:, 2], self.skeleton_surface[:, 3], s=1, alpha=0.2
            )
            self.ax.scatter(
                self.indentation_info[:, 2],
                self.indentation_info[:, 3],
                s=1,
                color="gray",
            )
            self.ax.set_xlim([-25, 25])
            self.ax.set_xlabel("Y")
            self.ax.set_ylim([-5, 50])
            self.ax.set_ylabel("Z")

        if mode == "xyz":
            self.ax.view_init(35, -105)

        self.canvas = self.fig.canvas

        # Save reference figure
        self.canvas.draw()  # To make sure there is a figure
        self.background = self.canvas.copy_from_bbox(
            self.ax.get_window_extent(self.ax.figure.canvas.renderer)
        )

        # Initialize Artist
        self.artist = self.ax.scatter(0, 0, 0)

        # Initialize plot
        self.ret = ret
        if not ret:
            plt.show(block=False)

    def update_force_map(self, map, threshold=0.008):

        self.artist.remove()

        if torch.is_tensor(map):
            map = map.detach().cpu().numpy()

        filter_mask = np.ones((40, 40))
        filter_mask[np.where(np.linalg.norm(map, axis=0) < threshold)] = 0
        predict = map * filter_mask
        force = predict.reshape((3, 1600))
        force = force[:, self.col_ind]

        filter_points = np.intersect1d(
            np.where(np.linalg.norm(force, axis=0) > threshold),
            np.where(self.indentation_info[:, 3] > 0),
        )

        if self.mode == "xyz":
            self.artist = self.ax.quiver(
                self.indentation_info[filter_points, 2],
                self.indentation_info[filter_points, 1],
                self.indentation_info[filter_points, 3],
                force[0, filter_points] * 400.0,
                force[1, filter_points] * 400.0,
                force[2, filter_points] * 400.0,
                color=self.colors_[3],
            )
            # self.artist.do_3d_projection(self.fig._cachedRenderer)
            self.artist.do_3d_projection()
        elif self.mode == "xy":
            self.artist = self.ax.quiver(
                self.indentation_info[filter_points, 1],
                self.indentation_info[filter_points, 2],
                force[0, filter_points] * 400.0,
                force[1, filter_points] * 400.0,
                color=self.colors_[3],
            )
        elif self.mode == "yz":
            self.artist = self.ax.quiver(
                self.indentation_info[filter_points, 2],
                self.indentation_info[filter_points, 3],
                force[1, filter_points] * 400.0,
                force[2, filter_points] * 400.0,
                color=self.colors_[3],
            )

        self.ax.draw_artist(self.artist)
        self.canvas.blit(self.ax.bbox)
        self.canvas.restore_region(self.background)
        self.canvas.flush_events()

        if self.ret:
            image_array = np.array(self.canvas.renderer.buffer_rgba())
            # Convert the NumPy array to a PIL Image
            return Image.fromarray(image_array)
