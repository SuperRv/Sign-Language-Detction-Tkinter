# Sign-Language-Detction-Tkinter
Created sign language detection using keras, tensorflow, openCV and Mediapipe

**I've created this project in virtual environment using Anaconda**

Step to complete this project:

1. Open collectdata.py to collect dat for each letter. Alternatively you can download the dataset from https://www.kaggle.com/datasets/ayuraj/asl-dataset. But you have to change the file name as:
 Image -> <alphabet in caps eg. A,B.C> -> <O to 69 for each image>
2. Run data.py file to extract key points.
3. Run trainmodel.py to create or train the ml model.
4. Run tkinter_app.py to start the application.

   **If you do not want to do any of this already a trained model is saved jsut run the tkinter_app.py file directly** 
