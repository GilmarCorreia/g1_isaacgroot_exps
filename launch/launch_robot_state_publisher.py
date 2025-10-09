# Default Imports
from launch_ros.actions import Node

# Custom Imports
from launch_interface import LaunchInterface

class RSP(LaunchInterface):

    def __init__(self, robot_description, model_ns):
        self.__setRobotDescription(robot_description)
        self.__setModelNS(model_ns)

        super().__init__(name="RobotStatePublisher")

    ########### SETTERS ###########
    def __setRobotDescription(self, robot_description):
        self.__robot_description = robot_description

    def __setModelNS(self, model_ns):
        self.__model_ns = model_ns

    ########### GETTERS ###########
    def getRobotDescription(self):
        return self.__robot_description

    def getModelNS(self):
        return self.__model_ns
    
    ########### METHODS ###########

    def createInterfaceNodes(self):
        node_robot_state_publisher = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            namespace=[self.getModelNS()],
            parameters=[
                {
                    'use_sim_time': True,
                    'robot_description': self.getRobotDescription()
                }
            ] # add other parameters here if required
        )

        self.addNode(node_robot_state_publisher)