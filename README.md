Golf Ball Launch Calculator
===========================

This project analyzes golf ball launch behavior using computer vision.
It detects the impact frame, locates the ball, and tracks its trajectory
across frames using OpenCV motion processing.

Features
--------
- Automatic detection of ball impact frame
- Ball localization using motion masks and color filtering
- Trajectory tracking using Farneback Optical Flow
- Cropped region processing for stable tracking
- Modular C++ design split across headers and source files
- CMake build system for cross-platform portability

Project Structure
-----------------
include/        - Header files
src/            - Source code
assets/         - Sample input video and example output
CMakeLists.txt  - Build configuration

Requirements
-----------
- C++17 compatible compiler
- OpenCV installed on the system
- CMake

Building the Project
--------------------
1. Create and enter the build directory:
   mkdir build
   cd build

2. Configure the project:
   cmake ..

3. Build the executable:
   cmake --build .

Using the Program
-----------------
Place your input video files in the assets/ directory.
Adjust the path inside main.cpp if needed:

   cv::VideoCapture vid("../assets/sample_video.mov");

Then run the executable from inside the build directory:

   ./main

Notes
-----
This repository does not include DLLs or compiled binaries.
Users must install OpenCV and configure their own environment.

Limitations
-----------
This project was developed and tested using a single low resolution sample video
recorded at 240 fps. The ball is only visible for approximately four frames
after impact, which limits the choice of tracking algorithms, and the accuracy
of trajectory estimation. Because of this, the current implementation focuses on:
 
- motion and color masking
- launch calculation with calculated data

The system has not yet been tested on footage with:

- different resolution or framerate
- different lighting conditions or camera placements  
- longer visible flight paths
- higher speed swings

As a result, the trajectory estimation is preliminary and should be interpreted
as a proof-of-concept rather than a finalized flight prediction model.

