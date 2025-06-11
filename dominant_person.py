# dominant_person_detector.py
import cv2
import mediapipe as mp
import numpy as np

class DominantPersonDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self._get_pose_model()
        self.drawing_utils = mp.solutions.drawing_utils

    def _get_pose_model(self):
        """Initialize a new MediaPipe pose model."""
        return self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def find_dominant_person(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        return results.pose_landmarks

    def draw_landmarks(self, frame, landmarks):
        """Draw pose landmarks on the frame."""
        self.drawing_utils.draw_landmarks(
            frame, landmarks, self.mp_pose.POSE_CONNECTIONS
        )

    def draw_bounding_box(self, frame, landmarks, h, w):
        """Draw bounding box around detected dominant person."""
        landmarks_list = landmarks.landmark

        # Calculate visibility score
        visibility_score = np.mean([landmark.visibility for landmark in landmarks_list])

        # Get bounding box coordinates
        xs = [landmark.x for landmark in landmarks_list]
        ys = [landmark.y for landmark in landmarks_list]

        min_x, max_x = int(min(xs) * w), int(max(xs) * w)
        min_y, max_y = int(min(ys) * h), int(max(ys) * h)

        # Compute bounding box area
        bbox_area = (max_x - min_x) * (max_y - min_y)

        # Combine visibility score and size into a dominance score
        dominance_score = 0.7 * visibility_score + 0.3 * (bbox_area / (h * w))

        # Draw bounding box and score
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        cv2.putText(frame, f"Score: {dominance_score:.2f}",
                    (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
        cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)
