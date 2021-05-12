# TopGuard

TopGuard is a human-focused car safety monitoring and report system that uses computer vision and AI to enable rapid crash response.  
  
![TopGuard Demo](https://media.giphy.com/media/UBmzNONbR5Wkz72K4j/giphy.gif)  
[shown: TopGuard has detected a passenger's eyes have been closed for a prolonged period after an accident.]

## Concept  

Events that indicate passengers in a vehicle are injured or in danger are detected by TopGuard. Then, TopGuard can relay that information to the relevant services, such as police or medical units.

See the TopGuard proof of concept demonstration here: [https://www.youtube.com/watch?v=MauU6TrVLGM](https://www.youtube.com/watch?v=MauU6TrVLGM)

## Required Packages


```bash
pip install cmake
pip install dlib
pip install cv2
pip install imutils
pip install scipy
```

## Additional Requirements

[haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/tree/master/data/haarcascades)
[shape_predictor_68_face_landmarks.dat](https://github.com/davisking/dlib-models)


## Sources and Research for This Project

[Eye blink detection with OpenCV, Python and dlib](https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)

[Real-time face liveness detection with Python, Keras and OpenCV](https://towardsdatascience.com/real-time-face-liveness-detection-with-python-keras-and-opencv-c35dc70dafd3)

Mafi, S., AbdelRazig, Y., & Doczy, R. (2018). Machine Learning Methods to Analyze Injury Severity of Drivers from Different Age and Gender Groups. Transportation Research Record, 2672(38), 171–183. [https://doi.org/10.1177/0361198118794292](https://doi.org/10.1177/0361198118794292)
