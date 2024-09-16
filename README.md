# Minsight: A Fingertip-Sized Vision-Based Tactile Sensor for Robotic Manipulation
<p align="right">
  *This is the only official content release regarding the design of Minsight.*
</p>

This project shows the principle desing of [Minsight](https://onlinelibrary.wiley.com/doi/full/10.1002/aisy.202300042). It is a soft fingertip-sized tactile sensor that can enbable robots feel directional force distributions all over its 3D surface.

The paper is titled "[Minsight: A Fingertip-Sized Vision-Based Tactile Sensor for Robotic Manipulation](https://onlinelibrary.wiley.com/doi/full/10.1002/aisy.202300042)", published at *Advanced Intelligent Systems*, 2023. 

<p align="center"><img src="Pics/HumanThumb.png" width="182.265" height="200">
  
This work was done by Iris Andrussow, Huanbo Sun, Katherine J. Kuchenbecker, Georg Martius at the Max Planck Institute for Intelligent Systems.
  
If you make used of our work, please use the citation information [below](#Citation).
  
# Abstract
Intelligent interaction with the physical world requires perceptual abilities beyond vision and hearing; vibrant tactile sensing is essential for autonomous robots to dexterously manipulate unfamiliar objects or safely contact humans. Therefore, robotic manipulators need high-resolution touch sensors that are compact, robust, inexpensive, and efficient. The soft vision-based haptic sensor presented herein is a miniaturized and optimized version of the previously published sensor Insight. Minsight has the size and shape of a human fingertip and uses machine learning methods to output high-resolution maps of 3D contact force vectors at 60 Hz. Experiments confirm its excellent sensing performance, with a mean absolute force error of 0.07 N and contact location error of 0.6 mm across its surface area. Minsight's utility is shown in two robotic tasks on a 3-DoF manipulator. First, closed-loop force control enables the robot to track the movements of a human finger based only on tactile data. Second, the informative value of the sensor output is shown by detecting whether a hard lump is embedded within a soft elastomer with an accuracy of 98\%. These findings indicate that Minsight can give robots the detailed fingertip touch sensing needed for dexterous manipulation and physical human–robot interaction.

# Working Mechanism
## Hardware Design
Insight is a vision-based haptic sensor that uses a camera and a structured lighting system to detect external contact happened on a soft hollow elastomer.
  
<p align="center"><img src="Pics/InsightStructure.png" width="395" height="180.5">
  
## Software Design
Insight constantly records images from inside using a camera. Feeding these images and a reference image into a trained machine learning model makes it possible to estimate the force distribution all over a surface. Each pixel in the force prediction map has three values that indicate the force strength in three directions (x, y, and z). In a force visualization on the right, each point corresponding to each pixel in the force map shows the force distribution of the contact in both normal and shear directions.
  
  <p align="center"><img src="Pics/DataFlow.png" width="472.125" height="222">
    
# Key Components
Four key design components make Insight work properly: the mechanical design of its shell, the tailored imaging system inside, automatic data collection, and cutting-edge deep learning are used behind the scenes to create a cheap, robust, and high-fidelity device.
     
  <p align="center"><img src="Pics/KeyComponents.png" width="405.75" height="201">
       
## Mechanics
Our hybrid structure of a soft elastomer shell enclosing a stiff metal skeleton ensures high sensitivity and robustness. The soft elastomer is for detecting light contact with high sensitivity. The metal skeleton is for holding the elastomer’s global shape and sustaining high-impact forces.

  <p align="center"><img src="Pics/Mechanics.png" width="430.325" height="262.675">

### Elastomer
- 3D Printer: [Formlabs Form 3](https://formlabs.com/eu/3d-printers/form-3/)
- 3D Printing Material: [Tough Resin FLTOTL05](https://formlabs.com/store/tough-2000-resin/). *Note: The material is updated.*
- Mold Design: 
  - [Mold 1](Solidworks/Elastomer_Mold1_In_Fingerprint_NailFlat.SLDPRT)
  - [Mold 2](Solidworks/Elastomer_Mold2_Out1.SLDPRT)
  - [Mold 3](Solidworks/Elastomer_Mold2_Out2.SLDPRT)
- Elastomer Material:
  - [EcoFlex 00-30](https://www.smooth-on.com/products/ecoflex-00-30/)
  - [Aluminum Powder 65 Micrometer, 99% Pure](https://www.amazon.de/Aluminumpulver-Aluminium-Pulver-Alupulver-Zus%C3%A4tze/dp/B06WRTGP2Y)
  - [Aluminum Flake 75 Micrometer](https://www.metallpulver24.de/de/aluminiumpulver-flaky-silber.html)
- [Vacuum Chamber VP1100, 5 Pa](https://www.silikonfabrik.de/vakuumtechnik/komplettsysteme/vakuum-komplettsystem-vks27/vp1200-vakuumkammer-und-pumpe.html). *Note: The Pump is updated.*
       
### Metal Skeleton
- 3D Printer: [ExOne X1 25 Pro](https://www.exone.com/en-US/Resources/News/X1-25PRO)
- 3D Printing Material: [AlSi10Mg-0403 Alluminum Alloy](https://www.shapeways.com/materials/aluminum) *Note: The material is updated.*
- Geometry Design: [Skeleton](Solidworks/Skeleton.SLDPRT)
- Printing Service: [Shapeways](https://www.shapeways.com/)
       
### Connector and Supporter
- 3D Printer: [ExOne X1 25 Pro](https://www.exone.com/en-US/Resources/News/X1-25PRO)
- 3D Printing Material: [AlSi10Mg-0403 Alluminum Alloy](https://www.shapeways.com/materials/aluminum) *Note: The material is updated.*
- Geometry Design: 
  - [Connector](Solidworks/Connector.SLDPRT)
  - [Supportor](Solidworks/Supporter.SLDPRT)
- Printing Service: [Shapeways](https://www.shapeways.com/)
      
## Imaging System
We use a LED ring and a collimator to create structured light. The structured light forms a light cone inside the enclosed volume. Each voxel inside this volume has a specific color code with 3D information. External contact moves the elastomer from one color zone to another. Together with shading effects the camera can detect 3D deformations of the surface from a single 2D image. As shown on the right figure, mimicking human fingerprint, we also add ridges to the surface to help the system recognize the directions of the surface extension.

  <p align="center"><img src="Pics/Imaging.png" width="482.3" height="246.4">
    
### Camera
- [Maker Hawk Raspberry Pi Camera Module 8 MP (Raspberry Pi camera V.2.0)](https://www.amazon.co.uk/MakerHawk-Raspberry-Compatible-Supporting-Resolution/dp/B07HL3Q58Z)

### LED Ring
- [Neopixel Ring with eight pieces of WS2812 5050](https://www.amazon.de/gp/product/B019ZL6724/ref=ppx_yo_dt_b_asin_title_o05_s00?ie=UTF8&psc=1)       

### Collimator
- 3D Printer: [Formlabs Form 3](https://formlabs.com/eu/3d-printers/form-3/)
- Material: [Standard Black](https://formlabs.com/de/shop/black-resin/)
- Geometry Design: [Collimator](Solidworks/Collimator.SLDPRT)

### DAQ
- [Raspberry Pi 4 Model B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) 
- [Python Code Runs in Raspberry Pi](Code/Python-Rasberry.md)

## Testbed
We design a test bed to stimulate the sensor in a controlled way, with both normal and shear forces, over the entire 3D sensing surface.
   <p align="center"><img src="Pics/Testbed.png" width="500.5" height="246.4"> 

### Linear Guide for 3 DOF Translation
- [Linear Guide](https://www.barchmotion.com/)
   - [Linear Guide Geometry](Pics/BarchMotionGeometry.png)
   - [Linear Guide Order List](Pics/BarchMotionOrder.png)
- [Power Supplier](https://www.boomaudio.de/mean-well-rsp-320-36-netzteil)
   - [RSp-320-36](https://www.boomaudio.de/mean-well-rsp-320-36-netzteil)
- [Arduino for Testbed](https://store.arduino.cc/products/arduino-micro)
   - [Arduino Micro](https://store.arduino.cc/products/arduino-micro)
   - [Arduino Code](Code/Testbed-Arduino.md)

### Motors for 2 DOF Rotation
- [Dynamixel Mx28-AT](https://emanual.robotis.com/docs/en/dxl/mx/mx-28/)
- [Dynamixel Mx64-AT](https://emanual.robotis.com/docs/en/dxl/mx/mx-64/)
- [Connector Mx64-AT-Testbed-Base](Solidworks/Mx64-AT-Testbed-Base.SLDPRT)
- [Connector Mx28-AT-Mx64-AT](Solidworks/Mx28-AT-Mx64-AT.SLDPRT)

### Force/Torque Sensor
- [ATI-MINI 40](https://www.ati-ia.com/products/ft/ft_models.aspx?id=mini40)
- [DAQ](https://www.ni.com/de-de/support/model.usb-6210.html)
       
# Software
## [Data Collection](Code/00_Experiment.ipynb)
### Communication
### Coordinate Transfer
### Data Collection
## Data Processing
### Direct Single Contact Prediction
### Force Map Predcition
### Mapping/Visualization
       
# Citation
Please use the following citation if you make use of our work:
## BibTex
``` citation
@article{https://doi.org/10.1002/aisy.202300042,
author = {Andrussow, Iris and Sun, Huanbo and Kuchenbecker, Katherine J. and Martius, Georg},
title = {Minsight: A Fingertip-Sized Vision-Based Tactile Sensor for Robotic Manipulation},
journal = {Advanced Intelligent Systems},
volume = {5},
number = {8},
pages = {2300042},
doi = {https://doi.org/10.1002/aisy.202300042},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/aisy.202300042},
year = {2023}
}
```
### RIS
TY  - JOUR
T1  - Minsight: A Fingertip-Sized Vision-Based Tactile Sensor for Robotic Manipulation
AU  - Andrussow, Iris
AU  - Sun, Huanbo
AU  - Kuchenbecker, Katherine J.
AU  - Martius, Georg
PY  - 2023
DO  - https://doi.org/10.1002/aisy.202300042
JO  - Advanced Intelligent Systems
JA  - Adv. Intell. Syst.
SP  - 2300042
VL  - 5
IS  - 8
PB  - John Wiley & Sons, Ltd
SN  - 2640-4567
UR  - https://doi.org/10.1002/aisy.202300042
Y2  - 2024/09/16
       
       
       
 
       
    