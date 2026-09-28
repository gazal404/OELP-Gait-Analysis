# OELP-Gait-Analysis

This repository contains the data-processing pipeline developed for our OELP project on IMU-based gait analysis.

The initial development and testing of the pipeline is being carried out using a publicly available gait dataset. The current implementation covers:

- IMU data loading
- Signal preprocessing and filtering
- Windowing of continuous sensor data
- Feature extraction

The purpose of using the public dataset is to develop and test these processing steps before collecting data from our own IMU setup.

The same preprocessing, windowing, and feature-extraction pipeline will later be adapted and reused with data collected from our thigh- and shank-mounted IMUs.

The current stage of this repository focuses on the data-processing pipeline up to feature extraction. Gait classification and the remaining project components will be developed in later stages.
