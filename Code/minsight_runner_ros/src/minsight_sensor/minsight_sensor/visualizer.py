from re import X
import matplotlib
import matplotlib.pyplot as plt
import time

import torch
import os
import time
from torch.autograd import Variable
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
import torchvision.transforms.functional as TF

sys.path.append('/is/sg2/iandrussow/trifinger_robot/tactile_sensing/minsight_runner/insight-min/training')
from model import get_model
from utils import read_json, LocalParams, load_checkpoint
from dataset import Postprocessor

sys.path.append('/is/sg2/iandrussow/trifinger_robot/tactile_sensing/minsight_runner/insight-min/custom_plotting')
from minsight_plotting import Plotter

#### Ros dependencies ####
import rclpy
from rclpy.node import Node

from minsight_interfaces.msg import Minsight
from minsight_interfaces.srv import CameraReset
#Import own topic data format here!
from cv_bridge import CvBridge, CvBridgeError


class SensorPlotter(Node):

    def __init__(self):
        super().__init__('minsight_sensor_plotter')

        config_base_path = "/is/sg2/iandrussow/trifinger_robot/tactile_sensing/minsight_runner/config/"
        configs = ["config_C1.json", "config_C3.json", "config_C6.json"]

         #use_gpu = torch.cuda.is_available()
        use_gpu = False

        self.cvbridge = CvBridge()
        self.use_gpu = use_gpu
        


        self.subscription1 = self.create_subscription(
            Minsight, 
            'sensor_dataC1',
            self.listener_callback1,
            10)
        self.subscription2 = self.create_subscription(
            Minsight, 
            'sensor_dataC3',
            self.listener_callback2,
            10)
        self.subscription3 = self.create_subscription(
            Minsight, 
            'sensor_dataC6',
            self.listener_callback3,
            10)

        # use gpu or not
        torch.cuda.empty_cache() 
        print("use_gpu:{}".format(use_gpu))

        
        self.plotters = []
        self.im_figs = []

        print("Setting up plotters")
        for i in range (3):
            config = read_json(config_base_path + configs[i])
            params = LocalParams(config)
            self.plotters.append(Plotter(os.path.join(params.data_path,params.sensor), params.sensor))

        print("Done setting up, now waiting...")

    def __del__(self):
        plt.close()

    def listener_callback1(self, msg):
        
        force_map = np.reshape(msg.forcemap, (3,40,40))
        self.plotters[0].update_force_map(force_map)

        print("Plot sensor 1: {}".format(np.sum(force_map)))
        image = self.cvbridge.imgmsg_to_cv2(msg.input,desired_encoding='passthrough')

        cv2.imshow("sensor 1", image)
        cv2.waitKey(1)

        #cv2.imwrite("sensor1.png", self.cvbridge.imgmsg_to_cv2(msg.input,desired_encoding='passthrough'))
    
    def listener_callback2(self, msg):

        force_map = np.reshape(msg.forcemap, (3,40,40))
        self.plotters[1].update_force_map(force_map)
        print("Plot sensor 2: {}".format(np.sum(force_map)))

        image = self.cvbridge.imgmsg_to_cv2(msg.input,desired_encoding='passthrough')
        cv2.imshow("sensor 2", image)
        cv2.waitKey(1)

        #cv2.imwrite("sensor2.png", self.cvbridge.imgmsg_to_cv2(msg.input,desired_encoding='passthrough'))

    def listener_callback3(self, msg):

        force_map = np.reshape(msg.forcemap, (3,40,40))
        self.plotters[2].update_force_map(force_map)
        print("Plot sensor 3: {}".format(np.sum(force_map)))

        image = self.cvbridge.imgmsg_to_cv2(msg.input,desired_encoding='passthrough')
        cv2.imshow("sensor 3", image)
        cv2.waitKey(1)

        #cv2.imwrite("sensor3.png", self.cvbridge.imgmsg_to_cv2(msg.input,desired_encoding='passthrough'))


def main(args=None):

   
    print("Initialize Plotter")
    rclpy.init(args=args)

    sensor_plotter = SensorPlotter()
    rclpy.spin(sensor_plotter)
    sensor_plotter.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
    
    

