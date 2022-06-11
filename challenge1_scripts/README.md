# Instructions to run challenge 1 solution

To run the solution code in this repository, we offer two possibilities. (1) Downloading the ready to use docker image ([docker instructions](#docker-container-instructions)) and (2) installing locally all the necessary dependencies ([local instructions](#local-installation-instructions)).

# Docker container instructions

Download the docker image with
```
docker pull jbarrag3/challenge1_solution:latest
```

Run the docker container
```
docker run -ti --name sol01 --rm --network host jbarrag3/challenge1_solution:latest
```

Within the container terminal run the following instructions to execute the solution script:
```
cd root/challenge1_solution/
python3 challenge1_scripts/Challenge1Solution.py -d cpu -t JhuNeedleTeam
```

## Troubleshoot docker container

TODO write about the problems with docker desktop.


# Local installation instructions
Clone the repository with 
```
git clone https://github.com/jabarragann/surgical_robotics_challenge.git
```
Install ros dependencies
```
apt-get install ros-noetic-cv-bridge
apt-get install python3-tf-conversions
```

Install python dependencies with
```
cd solution-scripts
pip install -r requirements.txt
```

or manually install them with

```
pip install pandas
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


## Additional configurations if installing code in Docker container
Configurations needed to install code in a new docker image.

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

       
