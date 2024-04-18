# face-recognition-system
Interview Task - Face Recognition System

## Requirements
First, Install requirementss
```
pip install -r requirements
```

### How to run
To run face recognition task, run the below code in Terminal:

```
python main.py --video_path [video path] --detector_backend [detector]
```

detector : [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]
### Face detection
----

Benchmark on Laptop CPU: Intel Core i7-12650H, GPU : NVIDIA RTX 3070

| Algorithm    | Speed `each frame(s)/FPS` | Accuracy `mean detection in each frame`
| --------  | :-------:       |---------- |
| OpenCV    | 0.082/12.19     | 1.1
| SSD       | 0.026/38.46     | 0
| Dlib      | 0.32/3.12       | 0.52
| MTCNN     | 0.75/1.33       | 2.25
|FastMTCNN  | 0.113/8.85      | 2.45
|RetinaFace | 4.8/0.2         | 5
|MediaPipe  | 0.0094/106.38   | 0
|**YoloV8** | **0.043/23.25** | **3.8**
|YuNet      | 0.0177/56.49    | 0.06
|CenterFace | 0.25/4          | 1.05


Although `mean detection in each frame` isn't a good metric to evaluate detection accuracy, But considering that there aren't any groud truth for this video we can use this for algorithm camparision assuming that all of the detections are correct.

It's obvious that **YoloV8** has a good trade-off between speed and accuracy!

### Face Recognition
---