# G1_IsaacGroot_Exps

Humanoid Robotics Engineer - Technical Challenge - Generalist Behavior via NVIDIA Isaac Gr00t N1.5

O objetivo desse repositório é realizar a integração e controle via NVIDIA Isaac Gr00t com o humanoide unitree G1. No primeiro momento foi escolhido a seguinte pilha tecnológica:

- Ubuntu 24.02
- Isaac Sim 5.0
- Isaac Gr00t N1.5
- Cuda Toolkit 12.8
- Anaconda para inicialização do ambiente virtual.
- Python 3.10
- ROS 2 Jazzy

## 1. Instalação

Escolha uma das opções a seguir para prosseguir com a instalação:

1.1. Configuração Local
1.2. Ambiente Docker

### 1.1. Configuração Local

Crie uma pasta e baixe esse repositório para dentro dessa pasta seguindo as seguintes configurações:

```bash
cd $HOME/Desktop
mkdir -p isaac_ws/src
cd isaac_ws/src
git clone link_do_repo

cd g1_isaacgroot_exps
mkdir downloads

isaac_text="
# Isaac Tests
export ISAAC_WS=$HOME/Desktop/isaac_ws
export ISAAC_EXPS=$HOME/Desktop/isaac_ws/src/g1_isaacgroot_exps
"

sudo echo "$isaac_text" >> ~/.bashrc
source ~/.bashrc
```

#### 1.1.1. ROS Jazzy

Instale o ROS2 versão jazzy no seu sistema operacional, para isso execute os comandos disponibilizados em https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html. Os comandos foram copiados para cá:

```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
sudo apt update && sudo apt install ros-dev-tools
sudo apt update
sudo apt upgrade
sudo apt install ros-jazzy-desktop ros-jazzy-xacro ros-jazzy-joint-state-publisher-gui

ros_text="
# ROS Jazzy
source /opt/ros/jazzy/setup.bash
source $ISAAC_WS/install/setup.bash
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
"

sudo echo "$ros_text" >> ~/.bashrc
source ~/.bashrc
``` 

#### 1.1.2. Isaac Sim 5.0

Instalar o Isaac Sim 5.0 no seguinte link https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.0.0-linux-x86_64.zip, descompacte a pasta e altere as variáveis de ambiente do sistema:

```bash
cd $ISAAC_EXPS/downloads
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.0.0-linux-x86_64.zip
tar -xvf isaac-sim-standalone-5.0.0-linux-x86_64.zip

isaac_sim_text="
# Isaac Sim 5.0 Setup
export isaac_sim_package_path=$ISAAC_EXPS/downloads/isaac-sim-standalone-5.0.0-linux-x86_64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$isaac_sim_package_path/exts/isaacsim.ros2.bridge/jazzy/lib
"

sudo echo "$isaac_sim_text" >> ~/.bashrc
source ~/.bashrc
``` 

#### 1.1.3. CUDA Toolkit 12.8

Algum texto

https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local


```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
sudo apt-get install -y nvidia-open
```

#### 1.1.4. Anaconda env

Instale o Anaconda para inicializar ambientes pré-configurados.

```bash
cd $ISAAC_EXPS/downloads
wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
./Anaconda3-2025.06-0-Linux-x86_64.sh 
```

Siga as configurações para montar o ambiente. Crie o ambiente o gr00t:

```bash
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools
pip install -e .[base]
pip install torch
pip install --no-build-isolation flash-attn==2.7.1.post4 
pip install catkin_pkg
```

### 1.2. Ambiente Docker

Em desenvolvimento.

## 2. Montagem do Cenário e Configurações do Robô

## 3. Exemplos

## ?. Referências

https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
https://github.com/NVIDIA/Isaac-GR00T/tree/main
https://forums.developer.nvidia.com/t/livox-mid360/283074/6
https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_rtx_lidar.html
https://docs.isaacsim.omniverse.nvidia.com/5.0.0/robot_setup_tutorials/tutorial_gui_simple_robot.html
https://reliablerobotics.ai/wp-content/uploads/2025/03/G1-User-Manual_compressed.pdf
https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_camera_depth_sensors.html
https://github.com/unitreerobotics/unitree_ros/tree/master
https://sensorlab.arizona.edu/sites/default/files/2023-07/Quick%20Start%20Guide.pdf


