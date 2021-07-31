# Face Recognition using OpenCV Python3

## Installation

1. Install OpenCV
   
   on Ubuntu can use this tutorial from [OpenCV Documentation](https://docs.opencv.org/4.5.2/d2/de6/tutorial_py_setup_in_ubuntu.html)
   
2. Install PyQt and PyQt-tools for GUI

   Install from terminal
   ```
   pip3 install --user pyqt5  
   sudo apt-get install python3-pyqt5  
   sudo apt-get install pyqt5-dev-tools
   sudo apt-get install qttools5-dev-tools
   ```

## How to use

- Train with Manual Dataset

   Train model with manual dataset on `faces-train.py`. Using Cascade Classifier which is provide and include from OpenCV4 or the latest OpenCV Version.
   
   Because the model only detect face, so the preTrained file using `haarcascade_frontalface_alt2.xml` on `haarcascades` folder.
   
   To change other detection, make sure to setting label while recognize on `faces.py`.
   
   Output of `faces-train.py` is yml file as model and labels.pickle as labelled dataset, so it can be used for `faces.py` to recognize.
   
- Run face recognition file or `faces.py` on terminal

- Using GUI App

   run Qt Designer from terminal to edit the GUI.

   `$ qtchooser -run-tool=designer -qt=5`

