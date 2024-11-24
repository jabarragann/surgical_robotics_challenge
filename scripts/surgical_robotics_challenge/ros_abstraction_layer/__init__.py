import os

__all__ = []

print("ral importing")

try:
    __ros_version_string = os.environ["ROS_VERSION"]
except:
    __ros_version_string = "2"
    print(
        "environment variable ROS_VERSION is not set, did you source your setup.bash?"
    )


if __ros_version_string == "1":
    __all__.append("ral")
    from .ral_ros1 import RAL_ROS1 as ral
elif __ros_version_string == "2":
    __all__.append("ral")
    from .ral_ros2 import RAL_ROS2 as ral
else:
    print(
        "environment variable ROS_VERSION must be either 1 or 2, did you source your setup.bash?"
    )
