Source files:
- src/robot.py: Functions for rotations and transformations
- src/mapping.py: Class for landmark mapping via EKF update (Part b)
    - **please refer to notebooks for usage**
    - EKFLandmarkMapping: Class provides methods for EKF update and predict. You may achieve IMU-based localization via EKF prediction (Part a) by calling predict method with u (linear velocity and angular velocity) only.
- src/slam.py: Class for visual inertial SLAM (Part c)
    - **please refer to notebooks for usage**
    - EKFSLAM: Class provides methods for EKF update and predict.
- src/utils.py: Functions for data loading and visualization

Scripts:
- hw3_main.py (deprecated): Project 3 starter script

Experiment Notebooks:
- EKFLandmarkMapping-<TestNumber>.ipynb: Experiments of landmark mapping via EKF update (Part b) with IMU-based localization via EKF prediction (Part a)
- EKFSLAM-<TestNumber>.ipynb: Experiments of visual inertial SLAM (Part c)
