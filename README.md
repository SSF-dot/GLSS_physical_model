### **A Physical Model and Calibration Method for Galvanometric Laser Scanning Systems**



This repository contains the source code and sample datasets for the paper titled "A Physical Model and Calibration Method for Galvanometric Laser Scanning Systems". The code implements the proposed physical model, the two-step calibration procedure, and the experimental validation. 



#### Environment & Dependencies

To ensure reproducibility, we recommend using **C++ 17** or higher. The core dependencies and their verified versions are listed below: 

| Dependency   | Version | Description                                       |
| ------------ | ------- | ------------------------------------------------- |
| OpenCV       | 4.5.5   | Scientific computing (SVD, matrix multiplication) |
| Ceres Solver | 1.12.0  | Optimization                                      |

*Note: The code may work with other versions, but the above configuration has been rigorously tested.* 



#### Dataset Structure

../phy_model/table-221.txt    # Input signal pairs (a, b)

../phy_model/reconstructed_points.txt    # Input reconstructed laser spots (x, y, z)

../phy_model/galvo_params_order3.yaml    # Input signal pair-to-angle conversion parameters

../phy_model/angle_dist_error_center.yaml    # Output calibration errors for central region

../phy_model/angle_dist_error_dege.yaml    # Output calibration errors for peripheral region



#### Data Processing Pipeline

Directly call the function `void SolvePhyModel()`.

The main workflow consists of nine stages:

Stage 1: Read the input parameters, including signal pairs, reconstructed spot coordinates and signal pair-to-angle conversion parameters.

Stage 2: Fit the effective outgoing rays using the spot coordinates.

Stage 3: Compute specular transformation matrices.

Stage 4: Divide all signal pair-outgoing ray data into calibration dataset and validation dataset.

Stage 5: Solve rotation matrix and the orientation of the incoming beam using function `ComputeRv2g`

Stage 6: Optimize rotation matrix and the orientation of the incoming beam using function `OptiGalvoRotPhyModel`

Stage 7: Determine translation vector and the position of the incoming beam using function `ComputeTranslationV2g`

Stage 8: Optimize all parameters using function `OptiGalvoRotTranPhyModel`

Stage 9: Output calibration error



