import matplotlib.pyplot as plt
import torch
import os
import cv2
import numpy as np
import sys

from .utils import read_json, LocalParams
from .plotting import Plotter

#### Ros dependencies ####
import rclpy
from rclpy.node import Node
from minsight_interfaces.msg import Minsight
from cv_bridge import CvBridge


class SensorPlotter(Node):

    def __init__(self):
        super().__init__("minsight_sensor_plotter")

        config_base_path = "../../../config/"
        config = "config_C1.json"

        # use_gpu = torch.cuda.is_available()
        use_gpu = False

        self.cvbridge = CvBridge()
        self.use_gpu = use_gpu

        self.subscription1 = self.create_subscription(
            Minsight, "sensor_data", self.listener_callback, 10
        )

        # use gpu or not
        torch.cuda.empty_cache()
        print("use_gpu:{}".format(use_gpu))

        self.plotters = []
        self.im_figs = []

        print("Setting up plotters")

        config = read_json(config_base_path + config)
        params = LocalParams(config)
        self.plotters = Plotter(
            os.path.join(params.data_path, params.sensor), params.sensor
        )

        print("Done setting up, now waiting...")

    def __del__(self):
        plt.close()

    def listener_callback(self, msg):

        force_map = np.reshape(msg.forcemap, (3, 40, 40))
        self.plotter.update_force_map(force_map)

        image = self.cvbridge.imgmsg_to_cv2(msg.input, desired_encoding="passthrough")

        cv2.imshow("sensor", image)
        cv2.waitKey(1)


def main(args=None):

    print("Initialize Plotter")
    rclpy.init(args=args)

    sensor_plotter = SensorPlotter()
    rclpy.spin(sensor_plotter)
    sensor_plotter.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
