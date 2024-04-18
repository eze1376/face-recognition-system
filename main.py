import cv2
import argparse

def StreamVideo(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    while True:
        # Read the video frame by frame
        ret, frame = video.read()

        # If the frame was not retrieved, then we have reached the end of the video
        if not ret:
            break

        # Display the resulting frame
        cv2.imshow("Frame", frame)

        # Press 'q' on the keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the video capture object
    video.release()
    
    # Destroy all the windows
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Run script with video path")
    parser.add_argument("--video_path", "-i", help="Input path to mp4 video")
    args = parser.parse_args()

    StreamVideo(args.video_path)


if __name__=="__main__":
    main()