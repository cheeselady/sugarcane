# Automated Sugarcane Billet Detection and Quality Control System - User Manual
## Overview
The automated sugarcane billet detection system was designed to allow sugarcane factory staff to easily monitor the quality control of each load of sugarcane billets delivered from farmers. The system can connect to a camera that monitors the billets transmitted from a conveyor. The system has two stages of detection. Firstly, it will detect each single billet on the conveyor and calculate the average length of the billets for each load. The individual billets will be cropped and saved as the input source for the second stage of detection. Users can choose which quality control they want to assess, such as the quality assessment of good or bad billets or check the number of eyes and nodes of each billet for plater.

This is a prototype version of the system, which also includes a general object detection function that detects people, cups, bags, chairs, etc. in real-time. This manual will provide an overview of the application and instructions on how to use it. 

There is a short video demonstration available at https://youtu.be/BZWiIOdxyVw. Please feel free to provide any further feedback.

## Installation
The prototype version of the system does not require installation. However, your computer must have a GPU, and the Python and CUDA environments must be installed. Instructions on how to install these requirements are included in the package.

## Getting Started
1.	Run the detect5.py under any IDE.
2.	The main window of the application will appear.
3.	You can start beginning the detection tasks.

## Using the System
### First Stage detection
1.	Select “billets” weights.
2.	Select input source:
a.	Image/Video: If this option is selected, source path needs to be provided. Path can be copied from the File Explore and pasted in the path field. 
b.	Camera: If the camera installed above the billet conveyor is connected, no need to provide the source path. 
3.	Selection of "Save Labels": To calculate the average billet length, ensure that the "Save Labels" option is checked. Otherwise, it is optional.
4.	Selection of “Save crops”: To continue performing the second stage detection, “Save crops” must be checked. Otherwise, it is optional.
5.	Click the 'Start detect' button to start the detection. Only need to click 'Stop detect' if the camera was chosen. 

This marks the end of the first stage detection, which can be used independently for tasks such as general object detection using the laptop camera, provided that the weights are set to 'yolov7'. Alternatively, it can be used as the initial step in a more complex detection task, followed by the second stage detection.

### Second Stage Detection 
Continuing from the previous step, to perform the second stage detection for quality control assessment, make sure to check the "Save crops" option. Otherwise, there will be no input source for the second stage detection.
6.	Check "Crop detect": This option indicates the system to perform the second stage detection automatically after the first stage is completed. Make sure to enable this option to obtain the quality control assessment.
7.	Selection of “Quality Assess” and/or “Eyes&Nodes”: Any or both of these two options need to be checked for system to perform the second stage detection. 
8.	Selection of “Save labels” and “Save images” are optional. 


### Expected Output
The detection results will be displayed in a text box on the window and will also be saved in a text logbook. Additionally, resulting images and labels will be saved in a folder that will pop up when detection is finished or the "Stop detect" button is clicked. The second stage detection will have its own label and image folders, as well as a text logbook, and will be saved in a folder with the quality control task as part of the folder name. This folder will be saved within the "crops" folder of the first detection.
This is a tree list of all directories that have saved the results for the two-stage detection task, with one image of billets as input used as an example to demonstrate.

![alt text](https://github.com/cheeselady/sugarcane/blob/master/um1.jpg)

### The main window:

![alt text](https://github.com/cheeselady/sugarcane/blob/master/um2.jpg)

### The main folder: 

![alt text](https://github.com/cheeselady/sugarcane/blob/master/um3.jpg)

### The crop folder: 

![alt text](https://github.com/cheeselady/sugarcane/blob/master/um4.jpg) 

### Logbooks examples: 

![alt text](https://github.com/cheeselady/sugarcane/blob/master/um5.jpg)
