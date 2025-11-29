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
 - include/        - Header files
 - src/            - Source code
 - assets/         - Sample input video and example output
 - CMakeLists.txt  - Build configuration

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
after impact, which limits the choice of tracking algorithms and the accuracy
of trajectory estimation. Because of this, the current implementation focuses on:

- motion and color masking
- launch calculation using short-range tracked data

The testing footage also introduced several structural constraints. The video
was recorded in portrait orientation, but OpenCV automatically rotates portrait
videos into landscape on load. As a result, parts of the pipeline were designed
specifically to compensate for this forced rotation and the altered coordinate
space.

Additionally, the video must include the golfer’s full backswing. The
`findImpactFrame` function intentionally skips the beginning portion of the file
to avoid misidentifying early club movements as ball impact. This behavior is
tuned for the test video and may not generalize to all swing recordings.

The system has not yet been tested on footage with:

- different resolutions or framerates
- different lighting conditions or camera placements
- longer visible flight paths
- higher speed swings
- landscape videos that do not require rotation correction
- footage trimmed to begin immediately before impact

As a result, the trajectory estimation is preliminary and should be interpreted
as a proof-of-concept rather than a finalized flight prediction model.

Future Improvements
-------------------
There are several areas where the system could be expanded or refined to improve
accuracy and general applicability beyond the constraints of the test footage.

1. Orientation Detection
   The current implementation includes special handling for portrait mode input
   because OpenCV automatically rotates such videos into landscape. A future
   version should detect video orientation and adjust processing steps
   accordingly, removing the need to hardcode orientation assumptions.

3. Improved Impact Frame Detection
   The `findImpactFrameIndex` function currently relies on averaging motion
   vectors across frames to infer the moment of ball contact. While functional,
   this approach is inefficient and not fully reliable in scenes where other
   motion (hands, hips, club, etc) dominates. A more robust method could
   incorporate references to static frames where the ball is known not to be
   moving, and compare them against motion and color masks to reject
   unrelated motion. This would greatly improve both speed and accuracy.

5. More Generalized Tracking
   The existing tracking pipeline was designed specifically to compensate for
   the limitations of the test video: low resolution, low framerate, and only
   four frames of visible ball flight. Traditional trackers such as CSRT or MIL
   struggled under these conditions, leading to a custom mask based implementation.
   With higher quality footage, future versions could integrate:
   - CSRT, KCF, or MIL tracking
   - Optical flow clustering or background subtraction
   - Training a small model for ball segmentation instead of color + motion masks
   - A GUI that shows the trajectory plot

6. Broader Dataset and Calibration
   Additional videos captured at higher framerates with clearer ball motion would
   allow for more generalizable logic and better parameter tuning. With more diverse
   example footage, the project could be built to produce more reliable speed, distance,
   and angle estimates across different swing/footage conditions.


Disclaimer
----------
This project analyzes locally provided video files and does not include any
copyrighted or proprietary clips. Users are responsible for ensuring they have
rights to any videos they analyze.
