import cv2
import argparse
from deepface import DeepFace

def StreamVideo(args):
    """stream video

    Args:
        args (Arguments): arguments of stream video including
            
        - video_path (str) : path to the input video
        - detector_backend (str) : backend name of detector
    """

    # Open the video file
    video = cv2.VideoCapture(args.video_path)

    while True:
        # Read the video frame by frame
        ret, frame = video.read()

        # If the frame was not retrieved, then we have reached the end of the video
        if not ret:
            break
        
        # Call Face Detection Service
        FaceDetection(frame, args.detector_backend)

        # Press 'q' on the keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the video capture object
    video.release()
    
    # Destroy all the windows
    cv2.destroyAllWindows()

def FaceDetection(frame, backend):
    """detect faces in the frame

    Args:
        frame (ndarray): an input frame to detect faces
        backend (str): model name to use for detection
    """
    try:
     face_objs = DeepFace.extract_faces(img_path = frame,
                                        detector_backend = backend
     )
    except:
        print("No Detection")
        face_objs = 0
        

def main():
    parser = argparse.ArgumentParser(description="Run script with video path")
    parser.add_argument("--video_path", "-vp", help="Input path to mp4 video")
    parser.add_argument("--detector_backend", "-db", help="Backend model of face detection [opencv | ssd | dlib | mtcnn | fastmtcnn | retinaface | mediapipe | yolov8 | yunet | centerface]")
    args = parser.parse_args()

    backends = [
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

    if not args.detector_backend in backends:
        args.detector_backend = None
    
    
    StreamVideo(args)


if __name__=="__main__":
    main()