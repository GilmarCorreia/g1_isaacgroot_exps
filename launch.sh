#!/bin/bash

# Function for custom model configuration
custom_model() {
    if [[ "$args" == "n" ]]; then
        read -p "What name do you want your model to have? (ex: r2d2): " model_name
        read -p "Do you want to use the default namespace development configuration? (y | n): " model_namespace_config
    fi

    model_params="model_name:=$model_name"
    
    if [[ "$model_namespace_config" == "n" ]]; then
        read -p "What namespace do you want your model to have? (ex: /test): " model_namespace
        model_params="$model_params model_namespace:=$model_namespace"
    fi

    simulator_config
}

# Function for simulator configuration
simulator_config() {
    if [[ "$args" == "n" ]]; then
        read -p "Do you want to launch rviz2? (y | n): " rviz_config
    fi

    simulator_params=""
    if [[ "$rviz_config" == "y" ]]; then
        simulator_params="launch_rviz2:=true"
    fi

    launch_simulator
}

# Function to launch the simulator
launch_simulator() {
    # Launch the ROS simulation
    ros2 launch g1_isaacgroot_exps model_rsp.launch.py model:=$model $model_params $simulator_params
}

# Source ROS initialization (replace with your correct ROS init script)
source ~/.bashrc

# Prompt the user for the model to launch
if [[ $# -eq 0 ]]; then
    echo "No arguments supplied"
    read -p "Enter the model to launch (g1): " model
    args=n
else
    args=y
    model=$1
    model_config=y
    rviz_config=y
fi

if [[ "$model" != "g1" ]]; then
    exit 0
fi

# Prompt for model usage
if [[ "$args" == "n" ]]; then
    read -p "Do you want to use the default development configuration? (y | n): " model_config
fi

if [[ "$model_config" == "y" ]]; then
    simulator_config
elif [[ "$model_config" == "n" ]]; then
    custom_model
else
    exit 0
fi

# Start the script by calling the initial scene config function
exit 0