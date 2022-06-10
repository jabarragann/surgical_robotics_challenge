# Instructions to run challenge 1 solution


# Installation
Clone the repository with 
```
git clone https://github.com/jabarragann/surgical_robotics_challenge.git
```
Install ros dependencies
```
apt-get install ros-noetic-cv-bridge
apt-get install python3-tf-conversions
```

Install python dependencies
```
cd solution-scripts
pip install -r requirements.txt
```

or manual install the dependencies with

```
pip install albumentations 
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu 
pip install spatialmath-python
pip install scikit-image
```

Install solution scripts
```
cd solution-scripts/
pip install -e .
```

# Running the solution script
Solution script needs to be run from the project root directory. The `-d` flag indicate the device for a segmentation deep learning model. If cuda is available set it up to `cuda` otherwise set it up to `cpu`
```
python3 challenge1_scripts/Challenge1Solution.py -d cuda -t JhuNeedleTeam
```


# Docker additional configurations
Configurations needed to run code in new docker image.

Ros packages
```
apt-get update
apt-get install ros-noetic-cv-bridge
apt-get install python3-tf-conversions
apt-get install net-tools
apt-get install iputils-ping
apt-get install netcat
```

Ros global variables
```
ROS_MASTER_URI=http://10.0.0.9:11311/
ROS_IP=10.0.0.9
```

       
