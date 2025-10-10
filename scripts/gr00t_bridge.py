#!/usr/bin/env python3
# ROS Imports
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import Image, JointState

# Dev Imports
import cv2
import json
import time
import base64
import requests
import numpy as np
import matplotlib.pyplot as plt
#from gr00t.eval.robot import RobotInferenceClient

GR00T_HOST = "localhost" 
GR00T_PORT = "5555" 

class Gr00tBridge(Node):

    ############################################################
    ######################## CONSTRUCTOR #######################
    ############################################################
    def __init__(self):
        super().__init__('gr00t_bridge')
        self.bridge = CvBridge()
        self.latest_image = None
        self.create_joints_vector()

        self.g1_refer = {
            "left_leg": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", 
                        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"],

            "right_leg": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", 
                        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],

            "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],

            "left_arm": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", 
                        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"],

            "left_hand": ["left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint", 
                        "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint"],

            "right_arm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", 
                        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"],

            "right_hand": ["right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
                        "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint"]
        }

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 
            '/G1/rgb/image_raw', 
            self.cb_image, 
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState, 
            '/G1/joint_states', 
            self.cb_jointstate, 
            10
        )

        self.instruction_sub = self.create_subscription(
            String,
            '/groot_command',
            self.cb_instruction,
            10
        )

        # Publishers
        self.pub_cmd = self.create_publisher(
            Float64MultiArray, 
            '/G1/joint_command', 
            10
        )

    ############################################################
    ######################### CALLBACKS ########################
    ############################################################
    def cb_image(self, msg: Image):
        try:
            cvb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cvb
        except Exception as e:
            self.get_logger().warn(f"cv_bridge fail: {e}")

    def cb_jointstate(self, msg: JointState):

        for name, pos, vel, eff in zip(msg.name, msg.position, msg.velocity, msg.effort):
            
            reference = ""
            for g1_name, joints in self.g1_refer.items():
                if name in joints:
                    reference = g1_name
                    break

            self.latest_joints[reference]["positions"][reference.index(name)] = pos
            self.latest_joints[reference]["velocities"][reference.index(name)] = vel
            self.latest_joints[reference]["efforts"][reference.index(name)] = eff

        #print(self.latest_joints)

    def cb_instruction(self, msg):
        self.instruction = msg.data
        self.get_logger().info(f"Received Instruction: '{self.instruction}'")

        # VERIFY THE SENSORS DATA
        # if self.latest_image is None or self.latest_joints is None:
        #     self.get_logger().warn("Sensors not updated.")
        #     return
        
        current_joint_states = self.latest_joints

        obs = {
            "video.ego_view": np.array([self.latest_image]).reshape((1, 256, 256, 3)) if self.latest_image is not None else np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
            "state.left_leg": np.array([current_joint_states["left_leg"]["positions"]]),
            "state.right_leg": np.array([current_joint_states["right_leg"]["positions"]]),
            "state.left_arm": np.array([current_joint_states["left_arm"]["positions"]]),
            "state.right_arm": np.array([current_joint_states["right_arm"]["positions"]]),
            "state.left_hand": np.array([current_joint_states["left_hand"]["positions"]]),
            "state.right_hand": np.array([current_joint_states["right_hand"]["positions"]]),
            "state.waist": np.array([current_joint_states["waist"]["positions"]]),
            "annotation.human.task_description": [self.instruction],
        }

        # for key, value in obs.items():
        #     if isinstance(value, np.ndarray):
        #         print(f"Obs: {key}: {value.shape}")


        action = self._example_http_client_call(obs, GR00T_HOST, GR00T_PORT, None)

        for key, value in action.items():
            print(f"Action: {key}: {value.shape}")

            # # Cria um gráfico
            # plt.figure(figsize=(10,6))

            # # Loop pelas 7 juntas
            # for i in range(value.shape[1]):
            #     plt.plot(range(1, value.shape[0]+1), value[:, i], marker='o', label=f'Joint {i+1}')

            # plt.xlabel('Sample')
            # plt.ylabel('Value')
            # plt.title(f'Value of the Joints Over Time {key}')
            # plt.legend()
            # plt.grid(True)
            # plt.show()
            
    
    ############################################################
    ####################### GR00T Methods ######################
    ############################################################

    # def _example_zmq_client_call(self, obs: dict, host: str, port: int, api_token: str):
    #     """
    #     Example ZMQ client call to the server.
    #     """
    #     # Original ZMQ client mode
    #     # Create a policy wrapper
    #     policy_client = RobotInferenceClient(host=host, port=port, api_token=api_token)

    #     print("Available modality config available:")
    #     modality_configs = policy_client.get_modality_config()
    #     print(modality_configs.keys())

    #     time_start = time.time()
    #     action = policy_client.get_action(obs)
    #     print(f"Total time taken to get action from server: {time.time() - time_start} seconds")
    #     return action

    def _example_http_client_call(self, obs: dict, host: str, port: int, api_token: str):
        """
        Example HTTP client call to the server.
        """
        import json_numpy

        json_numpy.patch()
        import requests

        # Send request to HTTP server
        print("Testing HTTP server...")

        time_start = time.time()
        response = requests.post(f"http://{host}:{port}/act", json={"observation": obs})
        print(f"Total time taken to get action from HTTP server: {time.time() - time_start} seconds")

        if response.status_code == 200:
            action = response.json()
            return action
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {}

    # ############################################################
    # ########################## METHODS #########################
    # ############################################################

    def create_joints_vector(self):
        self.latest_joints = {
            "left_leg": {
                "positions": np.zeros(6),
                "velocities": np.zeros(6),
                "efforts": np.zeros(6)
            },
            "right_leg": {
                "positions": np.zeros(6),
                "velocities": np.zeros(6),
                "efforts": np.zeros(6)
            },
            "waist": {
                "positions": np.zeros(3),
                "velocities": np.zeros(3),
                "efforts": np.zeros(3)
            },
            "left_arm": {
                "positions": np.zeros(7),
                "velocities": np.zeros(7),
                "efforts": np.zeros(7)
            },
            "left_hand": {
                "positions": np.zeros(6),
                "velocities": np.zeros(6),
                "efforts": np.zeros(6)
            },
            "right_arm": {
                "positions": np.zeros(7),
                "velocities": np.zeros(7),
                "efforts": np.zeros(7)
            },
            "right_hand": {
                "positions": np.zeros(6),
                "velocities": np.zeros(6),
                "efforts": np.zeros(6)
            }
        }

    # def call_gr00t(self, image_np, joint_names, joint_positions, instruction):
    #     # converte imagem para jpg base64
    #     _, enc = cv2.imencode('.jpg', image_np)
    #     b64 = base64.b64encode(enc.tobytes()).decode('utf-8')
    #     payload = {
    #         "instruction": instruction,
    #         "image_base64": b64,
    #         "proprio": {"names": joint_names, "positions": joint_positions},
    #         "max_action_steps": 16
    #     }
    #     try:
    #         r = requests.post(GR00T_SERVER, json=payload, timeout=2.0)
    #         r.raise_for_status()
    #         return r.json()
    #     except Exception as e:
    #         self.get_logger().error(f"GR00T request error: {e}")
    #         return None

    # def control_loop(self):
    #     if self.latest_image is None or self.latest_joints is None:
    #         return

    #     resp = self.call_gr00t(self.latest_image, self.joint_names, self.latest_joints, self.instruction)
    #     if resp is None:
    #         return

    #     # Exemplo: resposta hipotética -> {"action_type":"joint_positions", "positions":[...]}
    #     if resp.get('action_type') == 'joint_positions':
    #         targets = resp.get('positions', [])
    #         # segurança: clipping para limites conhecidos (se preencher joint_limits)
    #         for i, name in enumerate(self.joint_names):
    #             if name in self.joint_limits and i < len(targets):
    #                 lo, hi = self.joint_limits[name]
    #                 targets[i] = float(np.clip(targets[i], lo, hi))
    #         msg = Float64MultiArray()
    #         msg.data = targets
    #         self.pub_cmd.publish(msg)
    #         self.get_logger().info("Published joint targets from GR00T")
    #     else:
    #         # se GR00T retornar pose EEF, envie para controlador IK local
    #         self.get_logger().info(f"Unhandled GR00T response type: {resp.get('action_type')}")

def main(args=None):
    rclpy.init(args=args)
    node = Gr00tBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()