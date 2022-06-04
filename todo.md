# Todo list

1. Check angle error metric
1. Problem with metrics. Inconsistent metrics between online and offline pose estimation versions.
2. Check if I can compile AMBF with anaconda built-in ros. 
3. Set automatic connection
4. Transform hard code transformation from left camera to camera frame to avoid extra dependencies on AMBF python client. Add the code to calculate the transformation matrix from the camera config files. See adnan solution in the discussion section.


# Notes

* The `robotstackenv` environment has a problem with matplotlib and PyKDL. If matplotlib is imported first that will create problems. The problem most likely is being caused by having a mixtures of different python interpreters in my computer.

* If you are having problems with paths in VScode, check that you are running the the scripts from the main project folder.