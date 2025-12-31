# iChores package to bridge from ROS1 noetic to ROS2 workspace

1. source noetic workspace with this package build
2. export ROS\_MASTER\_URI and ROS\_IP according to your setup
3. launch topic publishers to ROS2 with following command:
```
roslaunch noetic_to_ros2 bridge.launch
```


