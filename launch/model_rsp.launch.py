import os, sys, subprocess
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import SetParameter
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction

import xacro, platform

# Custom imports
sys.path.append(os.path.join(os.path.dirname(__file__)))
from launch_robot_state_publisher import RSP
from launch_rviz2 import RViz2
#from launch_isaac import IsaacSim

pkg_name = 'g1_isaacgroot_exps'
default_class = "robots"

default_model = "g1"
default_model_name = "G1"

# Models supported organized by class
sensors = []
robots = ["G1"]

def generate_launch_description():

    # Specify the name of the package and path to xacro file within the package
    model_arg = DeclareLaunchArgument(name='model', description='Set the model to configure (Ex: g1)')
    model_name_arg = DeclareLaunchArgument(name='model_name', default_value=str(default_model_name), description='Set the model name (default: G1)')
    model_namespace_arg = DeclareLaunchArgument(name='model_namespace', default_value=str(""), description='Set the model namespace (Ex: /testNS)')
    launch_rviz2_arg = DeclareLaunchArgument(name='launch_rviz2', default_value='false', description='Execute rviz2 automatically (default: false)')

    # Run the node
    return LaunchDescription([
        SetParameter(name="use_sim_time", value=True),
        model_arg,
        model_name_arg, #OPTIONAL_ARG
        model_namespace_arg, #OPTIONAL_ARG,
        launch_rviz2_arg, # OPTIONAL_ARG
        OpaqueFunction(function=launch_setup)
    ])

def getRobotDescriptionRaw(model, model_class, model_name):

    file_subpath = f"models/{model_class}/{model}/urdf/{model}.urdf.xacro"

    # Create a dictionary of xacro arguments
    xacro_args = {
        'model_name': model_name,
    }

    # Use xacro to process the file
    xacro_file = os.path.join(get_package_share_directory(pkg_name),file_subpath)
    if (platform.system() == "Windows"):
        command = ['ros2', 'run', 'xacro', 'xacro', xacro_file]
        for arg, value in xacro_args.items():
            command.append(arg+":="+value)

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            robot_description_raw = result.stdout
    else:
        robot_description_raw = xacro.process_file(xacro_file, mappings=xacro_args).toxml()

    return robot_description_raw

def launch_setup(context, *args, **kwargs):
    # Getting arguments values
    model = LaunchConfiguration('model').perform(context)
    model_name = LaunchConfiguration('model_name').perform(context)
    model_name.replace(" ","_")

    model_namespace = LaunchConfiguration('model_namespace').perform(context)
    launch_rviz2 = (LaunchConfiguration('launch_rviz2').perform(context) == "true")
    
    # Setting parameters
    if (model in sensors):
        model_class = "sensors"
    elif(model in robots):
        model_class = "robots"
    else:
        model_class = default_class

    model_ns = f"{model_namespace}/{model_name}"
    robot_description_raw = getRobotDescriptionRaw(model, model_class, model_name)
    nodes = []

    # Configure the node Robot_State_Publisher
    rspConfig = RSP(robot_description_raw, model_ns)
    nodes.extend(rspConfig.getInterfaceNodes())

    # Launch rviz2
    if launch_rviz2:
        rviz_subpath = f"models/{model_class}/{model}/rviz/rviz_{model}_view.rviz"
        rviz2Config = RViz2(rviz_subpath, model_ns)
        nodes.extend(rviz2Config.getInterfaceNodes())

    return nodes