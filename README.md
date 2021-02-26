# Custom_object_classification_with_Yolov5s_and_OpenCV
A starter project to learn the custom object classification using Yolo5s model and OpenCV, 2 custom objects are classified.

--> Only 2 short videos (360 images) are used and detection was successful, obtained accuracy score over 50% <--

The two objects are some kind of nuts which are named as "uzum" and "kabak" in my country, Turkey :)

Steps:

1. Recording multiple short videos with nuts on basic but different colored backgrounds. eg. white paper, blue paper, one colored table
   
   First step is required since we can not manually label thousands of nut images, instead we use automatic video frame labelling, with this way, we obtain at least 15 labels for    each frame. (ie. 15 nuts on each frame)
   
2. Automatically labelling the "uzum"s and "kabak"s positions on the frames of the videos via OpenCV functions.
   
   HSV colorspace conversion + masking + corner detection is applied to detect the positions of the nuts on each frame, then position info's are saved in xml format. Xml format is used because in case of any mismatches, we can delete/add labels manually by uploading frames and xml files to "LabelImg", an image labelling tool.
   
3. Training the Yolo5s model w/ labelled images. 
   
   At this step, the position information of the objects (had been saved as xml files) are converted to yolo5s label.txt file format via a converter code. Then, https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data steps are followed for training.
   A pdf file is available to see how I executed training code on Google Colab.
   
4. Testing
   
   Result was satisfactory, however, increasing the number of videos to be trained could increase the accuracy.
   
For further information on how to detect your own custom objects, contact me: nssoyuslu@gmail.com
