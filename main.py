import cv2
import argparse
from deepface import DeepFace
import os

def DrawFaces(frame, face_objs, show):
    """draw given faces on the input frame and show it

    Args:
        frame (ndarray): input frame to show with faces
        face_objs (list): list of dictionaries that contain face information
        show (bool): whether show result frame or not

    Returns:
        ndarray: result frame with rectangles around faces
    """
    
    image = frame.copy()

    if isinstance(face_objs, list) and len(face_objs) > 0:
      for i in range(len(face_objs)):
        
        # Extract face coordinates in the frame
        x1 = face_objs[i]['facial_area']['x']
        y1 = face_objs[i]['facial_area']['y']
        x2 = x1 + face_objs[i]['facial_area']['w']
        y2 = y1 + face_objs[i]['facial_area']['h']

        # represents the top left corner of rectangle
        start_point = (x1, y1)

        # represents the bottom right corner of rectangle 
        end_point = (x2, y2) 

        # Blue color in BGR
        color = (255, 0, 0) 

        # Line thickness of 2 px 
        thickness = 2

        # Draw a rectangle with blue line borders of thickness of 2 px 
        image = cv2.rectangle(image, start_point, end_point, color, thickness) 
        
    if show:
        cv2.imshow("Frame", image)
    
    return image


def DrawRecognized(frame, face_objs, obj_distances, show):
    """Draw recognized faces with their distances on the input frame and show it

    Args:
        frame (ndarray): input frame to show with recognized faces
        face_objs (list): list of dictionaries that contain face information
        obj_distances (list): list of distances of each face to its recognized face in database
        show (bool): whether show result frame or not
    """
    image = frame.copy()
    
    if isinstance(face_objs, list) and len(face_objs) > 0:
      for i in range(len(obj_distances)):

        if not obj_distances[i].empty:

          # Extract face coordinates in the frame  
          x1 = face_objs[i]['facial_area']['x']
          y1 = face_objs[i]['facial_area']['y']
          x2 = x1 + face_objs[i]['facial_area']['w']
          y2 = y1 + face_objs[i]['facial_area']['h']

          # represents the top left corner of rectangle
          start_point = (x1, y1)

          # represents the bottom right corner of rectangle 
          end_point = (x2, y2) 

          # Green color in BGR 
          color = (0, 255, 0)

          # Line thickness of 2 px 
          thickness = 2

          # Draw a rectangle with blue line borders of thickness of 2 px 
          image = cv2.rectangle(image, start_point, end_point, color, thickness) 

          # Font 
          font = cv2.FONT_HERSHEY_SIMPLEX 
          
          # Create face message with its distances to recognized face in database and name of that recognized face
          message = str(round(obj_distances[i]['distance'].values[0], 3)) + " " +\
          os.path.basename(obj_distances[i]['identity'].values[0])

          # Put text above bounding box
          image = cv2.putText(image, message, (x1, y1 - 10), font,  
                     0.5, color, thickness, cv2.LINE_AA)

    if show:
        cv2.imshow("Frame", image)
        
    return image

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
        FaceDetection(frame, args)

        # Press 'q' on the keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the video capture object
    video.release()
    
    # Destroy all the windows
    cv2.destroyAllWindows()

def FaceDetection(frame, args):
    """detect faces in the frame

    Args:
        frame (ndarray): an input frame to detect faces
        args (args): user input arguments
    """
    try:
     face_objs = DeepFace.extract_faces(img_path = frame,
                                        detector_backend = args.detector_backend
     )
     
     if isinstance(face_objs, list) and len(face_objs) > 0:
        FaceRecognition(frame, face_objs, args)
    except:
        print("No Detection")
        face_objs = 0
        

def FaceRecognition(frame, faces, args):
    """recognize detected faces with matching them to the database faces

    Args:
        frame (ndarray): current frame to recognize faces of it
        faces (list): list of a dictionary which contain detected face information
        args (args): user input arguments
    """
    dfs_list = []
    
    for i in range(len(faces)):
        
      # Extract face from frame
      x1 = faces[i]['facial_area']['x']
      y1 = faces[i]['facial_area']['y']
      x2 = x1 + faces[i]['facial_area']['w']
      y2 = y1 + faces[i]['facial_area']['h']

      face_img = frame[y1:y2, x1:x2, :]

      # Face recognition
      dfs = DeepFace.find(img_path = face_img, 
                db_path = args.database_path,
                detector_backend=args.recongition_detector_backend,
                distance_metric = args.distance_metric, 
                model_name = args.recognition_model,
                threshold = args.threshold,
                enforce_detection=False
      )
      
      if isinstance(dfs, list) and len(dfs) > 0:
          dfs_list.append(dfs[0])

    frame_with_faces = DrawFaces(frame, faces, False)
    DrawRecognized(frame_with_faces, faces, dfs_list, True)
      
        

def main():
    parser = argparse.ArgumentParser(description="Run script with video path")
    parser.add_argument("--video_path", "-vp", help="Input path to mp4 video", type=str, required=True)
    parser.add_argument("--database_path", "-db", help="Path to the face database", type=str, required=True)
    parser.add_argument("--detector_backend", "-b", help="Backend model of face detection [opencv | ssd | dlib | mtcnn | fastmtcnn | retinaface | mediapipe | yolov8 | yunet | centerface]", type=str)
    parser.add_argument("--recongition_detector_backend", "-rb", help="Backend model of face detection in recognition phase [opencv | ssd | dlib | mtcnn | fastmtcnn | retinaface | mediapipe | yolov8 | yunet | centerface]", type=str)
    parser.add_argument("--distance_metric", "-dm", help="Distance metric for face recognition [cosine | euclidean | euclidean_l2]", type=str)
    parser.add_argument("--recognition_model", "-rm", help="Model name for face recognition [VGG-Face | Facenet | Facenet512 | OpenFace | DeepFace | DeepID | ArcFace | Dlib | SFace | GhostFaceNet]", type=str)
    parser.add_argument("--threshold", "-th", help="Threshold for distance of face recognition", type=float)
    
    
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
    
    metrics = ["cosine", "euclidean", "euclidean_l2"]
    
    models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]

    if not args.detector_backend in backends:
        args.detector_backend = 'opencv'
    if not args.distance_metric in metrics:
        args.distance_metric = 'cosine'
    if not args.recognition_model in models:
        args.recognition_model = 'VGG-Face'
    if not args.recongition_detector_backend in backends:
        args.recongition_detector_backend = 'opencv'
    
    
    StreamVideo(args)


if __name__=="__main__":
    main()