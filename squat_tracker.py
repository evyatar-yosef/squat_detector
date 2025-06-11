import cv2
import mediapipe as mp
import numpy as np
from squat_assesor import SquatAssessor

class SquatTracker:
    NOTHING = 1
    READY = 0
    IN_REP = 0
    FINISHED = 0
    squat_started = False

    def __init__(self, min_arm_angle=60, max_arm_angle=120, min_knee_angle=170):

        self.squat_assesor = SquatAssessor()

        self.min_arm_angle = min_arm_angle  # Minimum angle for arms holding the bar
        self.max_arm_angle = max_arm_angle  # Maximum angle for arms holding the bar
        self.min_knee_angle = min_knee_angle  # Minimum angle for knees to be considered extended
        self.NOTHING = 1
        self.previous_knee_angle = None  # Track the previous knee angle for detecting movement

    def calculate_angle(self, joint1, joint2, joint3):
        """Calculate the angle between three joints."""
        x1, y1 = joint1.x, joint1.y
        x2, y2 = joint2.x, joint2.y
        x3, y3 = joint3.x, joint3.y

        vector1 = np.array([x1 - x2, y1 - y2])
        vector2 = np.array([x3 - x2, y3 - y2])

        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        angle = np.arccos(dot_product / (magnitude1 * magnitude2)) * (180.0 / np.pi)
        return angle


    def is_ready_to_squat(self, landmarks):

        if self.IN_REP:
            return True
        if self.READY:
            return True

        """Check if the person is ready to squat."""
        shoulder = landmarks.landmark[11]  # Left shoulder
        elbow = landmarks.landmark[13]    # Left elbow
        wrist = landmarks.landmark[15]    # Left wrist
        hip = landmarks.landmark[23]      # Left hip
        knee = landmarks.landmark[25]     # Left knee
        ankle = landmarks.landmark[27]    # Left ankle

        # Check arm angle
        arm_angle = self.calculate_angle(shoulder, elbow, wrist)
        if not (self.min_arm_angle <= arm_angle <= self.max_arm_angle):
            return False

        if (wrist.y > elbow.y):  # Y-coordinates should decrease as we go up
            print("Arms not facing upwards")
            return False

        # Check knee angle
        knee_angle = self.calculate_angle(hip, knee, ankle)
        if knee_angle < self.min_knee_angle:
            return False



        # Ensure shoulders and hips are vertically aligned
        if abs(shoulder.x - hip.x) > 0.1:  # Allow a small deviation
            return False

        if self.NOTHING:
            print("Ready to squat")
            self.READY = 1
            self.NOTHING = 0


        return True

    def squat_rep_position(self, landmarks):
        """Check if the person is starting a squat by detecting knee bending."""
        if self.READY == 0 and self.IN_REP == 0:
            print("Not ready yet")
            return

        hip = landmarks.landmark[23]  # Left Hip
        knee = landmarks.landmark[25]  # Left Knee
        ankle = landmarks.landmark[27]  # Left Ankle
        knee_angle = self.calculate_angle(hip, knee, ankle)

        squat_start_threshold = 110  # Below this angle means squat started
        squat_end_threshold = 170    # Above this angle means squat ended

        # If squat is not started and knee angle is below start threshold, start squat
        if not self.squat_started and knee_angle < squat_start_threshold:
            self.squat_started = True
            self.READY = 0
            self.IN_REP = 1
            print("Squat started")

        # If squat is started and knee angle is above end threshold, end squat
        elif self.squat_started and knee_angle > squat_end_threshold:
            self.squat_started = False
            self.READY = 1
            self.IN_REP =    0
            print("Squat ended")


        return True  # Squat is in progress and posture is acceptable

    def calculate_angle(self, joint1, joint2, joint3):
        """Calculate the angle between three joints (hip, knee, and ankle)"""
        x1, y1 = joint1.x, joint1.y
        x2, y2 = joint2.x, joint2.y
        x3, y3 = joint3.x, joint3.y

        vector1 = np.array([x1 - x2, y1 - y2])
        vector2 = np.array([x3 - x2, y3 - y2])

        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        angle = np.arccos(dot_product / (magnitude1 * magnitude2)) * (180.0 / np.pi)
        return angle

    def check_finished_transition(self, landmarks):
        """
        Check if the person transitions from READY to FINISHED state by putting arms down.
        """
        if self.READY == 1:  # Only check this when in READY state
            shoulder = landmarks.landmark[11]  # Left shoulder
            elbow = landmarks.landmark[13]    # Left elbow
            wrist = landmarks.landmark[15]    # Left wrist

            # Check if arms are now facing downward
            if wrist.y > elbow.y:  # Y-coordinates increase downwards
                self.READY = 0
                self.FINISHED = 1
                print("Transitioned to FINISHED state")
                return True
        return False


    def get_ready(self):
        return self.READY

    def get_inrep(self):
        return self.IN_REP

    def get_finish(self):
        return self.FINISHED

    def get_current_state(self):
        """Return the current state of the squatting process."""
        if self.FINISHED:
            return "Finished"
        elif self.IN_REP:
            return "In Rep"
        elif self.READY:
            return "Ready"
        else:
            return "Not Ready"



