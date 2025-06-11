import cv2

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path

    def process_video(self, frame_callback, squat_readiness_checker=None):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {self.video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Callback to process the frame (e.g., for detecting landmarks)
            frame_callback(frame)

            # If a readiness checker is passed, display the current state
            if squat_readiness_checker:
                state = squat_readiness_checker.get_current_state()
                # Display the state on the frame
                cv2.putText(frame, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the frame
            cv2.imshow("Video Processor", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
