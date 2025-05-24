from pros_car_py.nav2_utils import (
    get_yaw_from_quaternion,
    get_direction_vector,
    get_angle_to_target,
    calculate_angle_point,
    cal_distance,
)
import math
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import tf2_ros
from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.duration import Duration
import yaml
import os
import cv2

class Nav2Processing:
    def __init__(self, ros_communicator, data_processor):
        self.ros_communicator = ros_communicator
        self.data_processor = data_processor
        self.finishFlag = False
        self.global_plan_msg = None
        self.index = 0
        self.index_length = 0
        self.recordFlag = 0
        self.goal_published_flag = False
        # /-----------------------------------------------------------------------/
        '''
        ArUco info Publisher
        '''
        self.aruco_pub = self.ros_communicator.create_publisher(
            String,
            '/aruco_info',
            10)
        
        '''
        Camera pose Subscriber
        '''
        self.latest_camera_pose = None  # 儲存 Camera pose 資料
        self.camera_pose_subscriber = self.ros_communicator.create_subscription(
            PoseWithCovarianceStamped,
            "/aruco_detector/pose",
            self.camera_pose_callback,
            10
        )
        
        '''
        Camera pose by tf2_ros
        '''
        # TF listener setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.ros_communicator)

        '''
        Camera RGB
        '''
        self.latest_rgb_image = None
        self.bridge = CvBridge()
        self.rgb_subscriber = self.ros_communicator.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            10)
        # /-----------------------------------------------------------------------/
        aruco_path = '/workspaces/src/pros_car_py/pros_car_py/aruco_info/living_room_aruco.yaml'
        if not os.path.exists(aruco_path):
            print(f"[living_room_nav] YAML 檔案不存在: {aruco_path}")
            return "STOP"

        with open(aruco_path, 'r') as f:
            self.aruco_info = yaml.safe_load(f)
        print("[living_room_nav] aruco_info 載入成功")
        # /-----------------------------------------------------------------------/
        map_path = '/workspaces/src/pros_car_py/pros_car_py/map'
        yaml_path = os.path.join(map_path, 'map01.yaml')

        with open(yaml_path, 'r') as f:
            map_metadata = yaml.safe_load(f)

        image_path = os.path.join(map_path, map_metadata['image'])
        resolution = map_metadata['resolution']
        origin = map_metadata['origin']
        occupied_thresh = map_metadata.get('occupied_thresh', 0.65)

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"無法載入地圖圖片: {image_path}")

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.flip(img, 0)  # 顛倒 y 軸

        h, w = img.shape
        self.map = [100 if px < occupied_thresh * 255 else 0 for row in img for px in row]
        print(f"[living_room_nav] 地圖大小: {w} x {h}，總像素數: {len(self.map)}")
        # /-----------------------------------------------------------------------/

    def reset_nav_process(self):
        self.finishFlag = False
        self.recordFlag = 0
        self.goal_published_flag = False

    def finish_nav_process(self):
        self.finishFlag = True
        self.recordFlag = 1

    def get_finish_flag(self):
        return self.finishFlag

    def get_action_from_nav2_plan(self, goal_coordinates=None):
        if goal_coordinates is not None and not self.goal_published_flag:
            self.ros_communicator.publish_goal_pose(goal_coordinates)
            self.goal_published_flag = True
        orientation_points, coordinates = (
            self.data_processor.get_processed_received_global_plan()
        )
        action_key = "STOP"
        if not orientation_points or not coordinates:
            action_key = "STOP"
        else:
            try:
                z, w = orientation_points[0]
                plan_yaw = get_yaw_from_quaternion(z, w)
                car_position, car_orientation = (
                    self.data_processor.get_processed_amcl_pose()
                )
                car_orientation_z, car_orientation_w = (
                    car_orientation[2],
                    car_orientation[3],
                )
                goal_position = self.ros_communicator.get_latest_goal()
                target_distance = cal_distance(car_position, goal_position)
                if target_distance < 0.5:
                    action_key = "STOP"
                    self.finishFlag = True
                else:
                    car_yaw = get_yaw_from_quaternion(
                        car_orientation_z, car_orientation_w
                    )
                    diff_angle = (plan_yaw - car_yaw) % 360.0
                    if diff_angle < 30.0 or (diff_angle > 330 and diff_angle < 360):
                        action_key = "FORWARD"
                    elif diff_angle > 30.0 and diff_angle < 180.0:
                        action_key = "COUNTERCLOCKWISE_ROTATION"
                    elif diff_angle > 180.0 and diff_angle < 330.0:
                        action_key = "CLOCKWISE_ROTATION"
                    else:
                        action_key = "STOP"
            except:
                action_key = "STOP"
        return action_key

    def get_action_from_nav2_plan_no_dynamic_p_2_p(self, goal_coordinates=None):
        if goal_coordinates is not None and not self.goal_published_flag:
            self.ros_communicator.publish_goal_pose(goal_coordinates)
            self.goal_published_flag = True

        # 只抓第一次路径
        if self.recordFlag == 0:
            if not self.check_data_availability():
                return "STOP"
            else:
                print("Get first path")
                self.index = 0
                self.global_plan_msg = (
                    self.data_processor.get_processed_received_global_plan_no_dynamic()
                )
                self.recordFlag = 1
                action_key = "STOP"

        car_position, car_orientation = self.data_processor.get_processed_amcl_pose()

        goal_position = self.ros_communicator.get_latest_goal()
        target_distance = cal_distance(car_position, goal_position)

        # 抓最近的物標(可調距離)
        target_x, target_y = self.get_next_target_point(car_position)

        if target_x is None or target_distance < 0.5:
            self.ros_communicator.reset_nav2()
            self.finish_nav_process()
            return "STOP"

        # 計算角度誤差
        diff_angle = self.calculate_diff_angle(
            car_position, car_orientation, target_x, target_y
        )
        if diff_angle < 20 and diff_angle > -20:
            action_key = "FORWARD"
        elif diff_angle < -20 and diff_angle > -180:
            action_key = "CLOCKWISE_ROTATION"
        elif diff_angle > 20 and diff_angle < 180:
            action_key = "COUNTERCLOCKWISE_ROTATION"
        return action_key

    def check_data_availability(self):
        return (
            self.data_processor.get_processed_received_global_plan_no_dynamic()
            and self.data_processor.get_processed_amcl_pose()
            and self.ros_communicator.get_latest_goal()
        )

    def get_next_target_point(self, car_position, min_required_distance=0.5):
        """
        選擇距離車輛 min_required_distance 以上最短路徑然後返回 target_x, target_y
        """
        if self.global_plan_msg is None or self.global_plan_msg.poses is None:
            print("Error: global_plan_msg is None or poses is missing!")
            return None, None
        while self.index < len(self.global_plan_msg.poses) - 1:
            target_x = self.global_plan_msg.poses[self.index].pose.position.x
            target_y = self.global_plan_msg.poses[self.index].pose.position.y
            distance_to_target = cal_distance(car_position, (target_x, target_y))

            if distance_to_target < min_required_distance:
                self.index += 1
            else:
                self.ros_communicator.publish_selected_target_marker(
                    x=target_x, y=target_y
                )
                return target_x, target_y

        return None, None

    def calculate_diff_angle(self, car_position, car_orientation, target_x, target_y):
        target_pos = [target_x, target_y]
        diff_angle = calculate_angle_point(
            car_orientation[2], car_orientation[3], car_position[:2], target_pos
        )
        return diff_angle

    def filter_negative_one(self, depth_list):
        return [depth for depth in depth_list if depth != -1.0]

    def camera_nav(self):
        """
        YOLO 目標資訊 (yolo_target_info) 說明：

        - 索引 0 (index 0)：
            - 表示是否成功偵測到目標
            - 0：未偵測到目標
            - 1：成功偵測到目標

        - 索引 1 (index 1)：
            - 目標的深度距離 (與相機的距離，單位為公尺)，如果沒偵測到目標就回傳 0
            - 與目標過近時(大約 40 公分以內)會回傳 -1

        - 索引 2 (index 2)：
            - 目標相對於畫面正中心的像素偏移量
            - 若目標位於畫面中心右側，數值為正
            - 若目標位於畫面中心左側，數值為負
            - 若沒有目標則回傳 0

        畫面 n 個等分點深度 (camera_multi_depth) 說明 :

        - 儲存相機畫面中央高度上 n 個等距水平點的深度值。
        - 若距離過遠、過近（小於 40 公分）或是實體相機有時候深度會出一些問題，則該點的深度值將設定為 -1。
        """
        yolo_target_info = self.data_processor.get_yolo_target_info()
        camera_multi_depth = self.data_processor.get_camera_x_multi_depth()
        if camera_multi_depth == None or yolo_target_info == None:
            return "STOP"

        camera_forward_depth = self.filter_negative_one(camera_multi_depth[7:13])
        camera_left_depth = self.filter_negative_one(camera_multi_depth[0:7])
        camera_right_depth = self.filter_negative_one(camera_multi_depth[13:20])

        action = "STOP"
        limit_distance = 0.7

        if all(depth > limit_distance for depth in camera_forward_depth):
            if yolo_target_info[0] == 1:
                if yolo_target_info[2] > 200.0:
                    action = "CLOCKWISE_ROTATION_SLOW"
                elif yolo_target_info[2] < -200.0:
                    action = "COUNTERCLOCKWISE_ROTATION_SLOW"
                else:
                    if yolo_target_info[1] < 0.8:
                        action = "STOP"
                    else:
                        action = "FORWARD_SLOW"
            else:
                action = "FORWARD"
        elif any(depth < limit_distance for depth in camera_left_depth):
            action = "CLOCKWISE_ROTATION"
        elif any(depth < limit_distance for depth in camera_right_depth):
            action = "COUNTERCLOCKWISE_ROTATION"
        return action

    def camera_nav_unity(self):
        """
        YOLO 目標資訊 (yolo_target_info) 說明：

        - 索引 0 (index 0)：
            - 表示是否成功偵測到目標
            - 0：未偵測到目標
            - 1：成功偵測到目標

        - 索引 1 (index 1)：
            - 目標的深度距離 (與相機的距離，單位為公尺)，如果沒偵測到目標就回傳 0
            - 與目標過近時(大約 40 公分以內)會回傳 -1

        - 索引 2 (index 2)：
            - 目標相對於畫面正中心的像素偏移量
            - 若目標位於畫面中心右側，數值為正
            - 若目標位於畫面中心左側，數值為負
            - 若沒有目標則回傳 0

        畫面 n 個等分點深度 (camera_multi_depth) 說明 :

        - 儲存相機畫面中央高度上 n 個等距水平點的深度值。
        - 若距離過遠、過近（小於 40 公分）或是實體相機有時候深度會出一些問題，則該點的深度值將設定為 -1。
        """
        yolo_target_info = self.data_processor.get_yolo_target_info()
        camera_multi_depth = self.data_processor.get_camera_x_multi_depth()
        yolo_target_info[1] *= 100.0
        camera_multi_depth = list(
            map(lambda x: x * 100.0, self.data_processor.get_camera_x_multi_depth())
        )

        if camera_multi_depth == None or yolo_target_info == None:
            return "STOP"

        camera_forward_depth = self.filter_negative_one(camera_multi_depth[7:13])
        camera_left_depth = self.filter_negative_one(camera_multi_depth[0:7])
        camera_right_depth = self.filter_negative_one(camera_multi_depth[13:20])
        action = "STOP"
        limit_distance = 10.0
        print(yolo_target_info[1])
        if all(depth > limit_distance for depth in camera_forward_depth):
            if yolo_target_info[0] == 1:
                if yolo_target_info[2] > 200.0:
                    action = "CLOCKWISE_ROTATION_SLOW"
                elif yolo_target_info[2] < -200.0:
                    action = "COUNTERCLOCKWISE_ROTATION_SLOW"
                else:
                    if yolo_target_info[1] < 2.0:
                        action = "STOP"
                    else:
                        action = "FORWARD_SLOW"
            else:
                action = "FORWARD"
        elif any(depth < limit_distance for depth in camera_left_depth):
            action = "CLOCKWISE_ROTATION"
        elif any(depth < limit_distance for depth in camera_right_depth):
            action = "COUNTERCLOCKWISE_ROTATION"
        return action

    def publish_aruco_info(self):
        if not hasattr(self, 'aruco_pub'):
            print("[publish_aruco_info] Publisher 尚未初始化！")
            return

        yaml_string = yaml.dump(self.aruco_info)
        msg = String()
        msg.data = yaml_string
        self.aruco_pub.publish(msg)
        print("[publish_aruco_map] 發布 aruco_info 成功！")
    
    def camera_pose_callback(self, msg):
        self.latest_camera_pose = msg
        # print(f"[callback] 收到 camera pose: {msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f}")
    
    def image_callback(self, msg):
        try:
            # 將 ROS CompressedImage 轉成 OpenCV 圖片格式
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb_image = cv_image
        except Exception as e:
            print(f"[image_callback] 解碼圖片失敗: {e}")
            self.latest_rgb_image = None

    def get_base_pose_from_tf(self):
        try:
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame="map",
                source_frame="base_link",
                time=rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            pos = trans.transform.translation
            ori = trans.transform.rotation
            return {
                "position": {"x": pos.x, "y": pos.y, "z": pos.z},
                "orientation": {"x": ori.x, "y": ori.y, "z": ori.z, "w": ori.w}
            }
        except Exception as e:
            print(f"[TF] 無法取得 base_link 的位置: {e}")
            return None


    def living_room_nav(self):
        # print(f"[living_room_nav] aruco_info : {self.aruco_info}")
        # print(f"[living_room_nav] 地圖總像素數: {len(self.map)}")
    
        # Publish ArUco_info (給 ArucoDetectorNode)
        self.publish_aruco_info()

        # Subscribe camera pose (從 ArucoDetectorNode) 
        camera_pose_msg = self.latest_camera_pose
        if self.latest_camera_pose is None:
            print("[living_room_nav] 尚未接收到 Camera pose")
            return "STOP"

        pos = self.latest_camera_pose.pose.pose.position
        ori = self.latest_camera_pose.pose.pose.orientation
        camera_pose = {
            "position": {"x": pos.x, "y": pos.y, "z": pos.z},
            "orientation": {"x": ori.x, "y": ori.y, "z": ori.z, "w": ori.w}
        }
        print("[living_room_nav] Camera pose:", camera_pose)

        # Subscribe Camera RGB image (從 CameraNode)    
        if self.latest_rgb_image is None:
            print("[living_room_nav] 尚未接收到相機影像")
            return "STOP"
        print("[living_room_nav] 成功取得 RGB 影像，大小：", self.latest_rgb_image.shape)
        
        # Subscribe camera pose (by tf2_ros)
        camera_pose_from_tf = self.get_base_pose_from_tf()
        if camera_pose_from_tf is None:
            print("[living_room_nav] 無法從 TF 取得 camera pose")
            return "STOP"

        print("[living_room_nav] TF Pose:", camera_pose_from_tf)


        # 這邊你可以根據 pose 控制導航或設定導航目標
        """
        self.map
        self.aruco_info
        camera_pose
        camera_pose_from_tf
        self.latest_rgb_image
        """
        # 目前先回傳 STOP 作為 placeholder
        return "STOP"
    


    def stop_nav(self):
        return "STOP"
