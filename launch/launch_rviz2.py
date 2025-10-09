# Default Imports
import os
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# Custom Imports
from launch_interface import LaunchInterface

class RViz2(LaunchInterface):

    def __init__(self, rviz_subpath, model_ns):
        self.__setRvizSubpath(rviz_subpath)
        self.__setModelNS(model_ns)

        super().__init__(name="RViz2")

    ########### SETTERS ###########
    def __setRvizSubpath(self, rviz_subpath):
        self.__rviz_subpath = rviz_subpath

    def __setModelNS(self, model_ns):
        self.__model_ns = model_ns

    ########### GETTERS ###########
    def getRvizSubpath(self):
        return self.__rviz_subpath

    def getModelNS(self):
        return self.__model_ns
    
    ########### METHODS ###########

    def createInterfaceNodes(self):

        # Loading RViZ2 Node
        node_rviz2 = Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['-d', os.path.join(get_package_share_directory(self.getPkgName()),self.getRvizSubpath())] 
        )

        self.addNode(node_rviz2)

        # # Loading Joint_State_Publisher_Gui Node
        # node_joint_state_publisher_gui = Node(
        #     package='joint_state_publisher_gui',
        #     executable='joint_state_publisher_gui',
        #     output='screen',
        #     namespace=[self.getModelNS()]
        # )

        # self.addNode(node_joint_state_publisher_gui)

