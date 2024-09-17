import matplotlib.pyplot as plt
import time
import torchvision.transforms.functional as TF
import torch
import os
import time

import cv2
import numpy as np

#### Ros dependencies ####
import rclpy
from rclpy.node import Node
from torch.autograd import Variable

from minsight_interfaces.msg import CameraMsg, Minsight
from minsight_interfaces.srv import CameraReset

from .model import get_model
from .utils import load_checkpoint, read_json, LocalParams
from .dataset import Postprocessor

from cv_bridge import CvBridge


class MinsightSensor:

    def __init__(self, data_path, use_gpu, params):

        self.force_map = params.force_map
        self.params = params

        self.use_gpu = use_gpu

        self.X3 = cv2.imread(os.path.join(data_path, "NoContact_avg.png"))
        self.X4 = TF.to_tensor(cv2.imread((os.path.join(data_path, "gradient1.png"))))
        self.X5 = TF.to_tensor(cv2.imread((os.path.join(data_path, "gradient2.png"))))

        self.postprocessor = Postprocessor(use_gpu, params)

        # use gpu or not
        torch.cuda.empty_cache()
        print("use_gpu:{}".format(self.use_gpu))

        model, optimizer_ft, exp_lr_scheduler = get_model(self.params)

        checkpoint_path = os.path.join(self.params.working_dir, "checkpoint.pt")
        self.model = load_checkpoint(
            checkpoint_path, model, optimizer_ft, exp_lr_scheduler, self.use_gpu
        )

        if self.use_gpu:
            self.model = self.model.cuda()

        print("Using fully trained model")
        self.model.eval()

    def reset_no_contact_img(self, img):
        self.X3 = img

    def preprocess(self, image):

        X13 = TF.to_tensor(cv2.subtract(image, self.X3))

        if self.force_map:
            X = torch.cat([X13, self.X4[0][None, :, :], self.X5[0][None, :, :]], 0)
        else:
            X = torch.cat([X13], 0)

        return X

    def inference(self, input):

        # wrap them in Variable
        input = Variable(input)
        if self.use_gpu:
            input = input.cuda()

        # Fix channel conversion issue:
        input = torch.unsqueeze(input, 0)

        # forward
        outputs = self.model(input)

        return outputs


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
