from re import X
import matplotlib
import matplotlib.pyplot as plt
import time

import torch
import os
import time

import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
import torchvision.transforms.functional as TF

sys.path.append(
    "/is/sg2/iandrussow/trifinger_robot/tactile_sensing/minsight_runner/insight-min/training"
)
from utils import read_json, LocalParams
from inference import MinsightSensor


#### Ros dependencies ####
import rclpy
from rclpy.node import Node

from minsight_interfaces.msg import CameraMsg, Minsight
from minsight_interfaces.srv import CameraReset

from cv_bridge import CvBridge, CvBridgeError


class Sensor(Node):

    def __init__(self):
        super().__init__("minsight_sensor_publisher")

        self.declare_parameter("config_path", rclpy.Parameter.Type.STRING)
        self.config_path = self.get_parameter("config_path").value

        self.declare_parameter("camera_port", rclpy.Parameter.Type.INTEGER)
        self.camera_port = self.get_parameter("camera_port").value

        config = read_json(self.config_path)
        params = LocalParams(config)
        self.force_map = params.force_map

        self.sensor_name = params.sensor

        print(os.path.join(params.data_path, params.sensor))

        use_gpu = torch.cuda.is_available()

        self.publisher = self.create_publisher(Minsight, "sensor_data", 10)

        self.subscription = self.create_subscription(
            CameraMsg, "image_data", self.listener_callback, 10
        )

        self.srv = self.create_service(
            CameraReset, "reset_camera_images", self.reset_no_contact_img
        )

        self.cvbridge = CvBridge()

        self.sensor = MinsightSensor(
            os.path.join(params.data_path, params.sensor), use_gpu, params
        )
        self.sensor.X3 = cv2.imread("default_{}.png".format(self.camera_port))

        self.image = None
        self.time = time.time()

    def __del__(self):
        plt.close()

    def reset_no_contact_img(self, request, response):

        self.get_logger().info("Reset camera image")

        self.sensor.reset_no_contact_img(self.image)
        cv2.imwrite("no_contact{}.png".format(self.sensor_name), self.image)

        response.res = True
        return response

    def listener_callback(self, msg):

        self.image = self.cvbridge.imgmsg_to_cv2(
            msg.input, desired_encoding="passthrough"
        )
        self.time = msg.timestamp

        input = self.sensor.preprocess(np.squeeze(self.image))
        output_raw = self.sensor.inference(input)

        output = np.squeeze(
            self.sensor.postprocessor.undo_rescale(output_raw).cpu().detach().numpy()
        )

        msg = Minsight()
        msg.map_mode = self.force_map
        msg.forcemap = output.flatten()
        msg.input = self.cvbridge.cv2_to_imgmsg(self.image, encoding="passthrough")
        msg.capture_time = self.time
        self.publisher.publish(msg)


def main(args=None):

    print("Initialize Sensor")
    rclpy.init(args=args)

    sensor_publisher = Sensor()
    rclpy.spin(sensor_publisher)
    sensor_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
