import matplotlib.pyplot as plt
import time

import time
import cv2

#### Ros dependencies ####
import rclpy
from rclpy.node import Node
from minsight_interfaces.msg import CameraMsg

from cv_bridge import CvBridge


class Camera(Node):

    def __init__(self):
        super().__init__("minsight_image_publisher")
        self.publisher_ = self.create_publisher(CameraMsg, "image_data", 10)
        self.cvbridge = CvBridge()

        self.declare_parameter("camera_port", rclpy.Parameter.Type.INTEGER)
        self.camera_port = self.get_parameter("camera_port").value

        self.declare_parameter("crop_start", 160)
        self.crop_start = self.get_parameter("crop_start").value

        # We will publish a message every 1/60.0 seconds
        timer_period = 1.0 / 60.0  # seconds
        # Create the timer
        self.timer = self.create_timer(timer_period, self.sensor_callback)

        self.cap = self.set_camera()

        # sensor warmup
        for _ in range(20):
            _ = self.imaging()

        cv2.imwrite("default_{}.png".format(self.camera_port), self.imaging())

    def __del__(self):
        plt.close()

    def set_camera(self):

        # cap = cv2.VideoCapture("/dev/video0")
        cap = cv2.VideoCapture(self.camera_port, cv2.CAP_V4L2)
        print("Setting camera to port %i " % self.camera_port)

        # Check whether user selected camera is opened successfully.
        if not (cap.isOpened()):
            raise NameError("Could not open video device")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        # cap.set(cv2.CAP_PROP_GAMMA, 170)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, 156.0)  # 300
        cap.set(cv2.CAP_PROP_AUTO_WB, -1)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def imaging(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                crop = frame[:, self.crop_start : (self.crop_start + 960)]
                crop = cv2.flip(
                    crop, -1
                )  # Flip the image around both axis, as this is also done during preprocessing of the training data to match the skeleton orientation
                crop = cv2.resize(crop, (410, 308))
                return crop

    def sensor_callback(self):

        input_img = self.imaging()

        msg = CameraMsg()
        msg.timestamp = time.time()
        msg.input = self.cvbridge.cv2_to_imgmsg(input_img, encoding="passthrough")

        self.publisher_.publish(msg)
        # self.get_logger().info('Publishing sensor image')


def main(args=None):

    print("Initialize Camera")
    rclpy.init(args=args)

    publisher = Camera()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
