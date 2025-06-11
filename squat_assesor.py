import math
import numpy as np


class SquatAssessor:
    def __init__(self, min_score=0, max_score=20):

        self.min_score = min_score
        self.max_score = max_score

        self.shoulder_scores = []
        self.knee_alignment_scores = []
        self.score_knee_angle = 0
        self.score_knee_deviation = self.max_score
        self.overall_score = 0
        self.score_shoulder = 0
        self.all_rep_overall_scores = []
        self.all_rep_shoulder_scores = []
        self.all_rep_depth_scores = []
        self.all_rep_knee_tracking_scores = []
        self.all_rep_knee_alignment_scores = []


    def assess_squat(self, landmarks):

        shoulder_avg = self.get_average_shoulder_score()
        self.all_rep_shoulder_scores.append(shoulder_avg)

        depth_score = self.get_depth(landmarks)
        self.all_rep_depth_scores.append(depth_score)

        knee_score = self.score_knee_deviation
        self.all_rep_knee_tracking_scores.append(knee_score)

        knee_alignment_score = self.get_average_knee_alignment_score()
        self.all_rep_knee_alignment_scores.append(knee_alignment_score)

        overall_score = self.calculate_overall_score(shoulder_avg,depth_score,knee_score,knee_alignment_score)
        self.all_rep_overall_scores.append(overall_score)


        print(f"✅ Final Squat Scores:")
        print(f"   Shoulder Alignment Score: {shoulder_avg:.2f}")
        print(f"   Depth Score: {depth_score:.2f}")
        print(f"   Knee Tracking Score: {knee_score:.2f}")
        print(f" knee alinment score:{knee_alignment_score:.2f}")
        print(f" overall_score:{overall_score:.2f}")

        return {
            "shoulders": shoulder_avg,
            "depth": depth_score,
            "knees": knee_score,
            "knees_alinment": knee_alignment_score,
            "overall_score": overall_score
        }


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

    ### Functions for depth score ###

    def get_depth(self, landmarks):
        curr_score_knee_angle = self.evaluate_depth_with_angle(landmarks)
       # print("This is current score knee angle: " + str(curr_score_knee_angle))
       # print("This is self score knee angle: " + str(self.score_knee_angle))
        if curr_score_knee_angle > self.score_knee_angle :
            self.score_knee_angle = curr_score_knee_angle
        return self.score_knee_angle

    def evaluate_depth_with_angle(self, landmarks):
        hip = landmarks.landmark[23]  # Left hip
        knee = landmarks.landmark[25]  # Left knee
        ankle = landmarks.landmark[27]  # Left ankle

        angle = self.calculate_angle(hip, knee, ankle)

        max_angle = 180  # Maximum angle (standing straight)
        min_angle = 50  # Minimum angle (deep squat)

        # Normalize the angle to a score between 0 and 20
        if angle > max_angle:
            score = self.min_score  # Above max angle, no depth achieved
        elif angle < min_angle:
            score = self.max_score  # Below min angle, max depth achieved
        else:
            score = self.max_score * (1 - (angle - min_angle) / (max_angle - min_angle))

        return score

    def reset(self):
        self.score_knee_angle = 0
        self.score_knee_deviation = 20
        self.shoulder_scores.clear()

    ### Functions for Knee allignment ###

    def evaluate_knee_tracking(self, landmarks):
        """
        Evaluates if the knee is tracking correctly on the x-axis relative to the toes,
        with stricter penalties for side-to-side deviation as the squat deepens.
        """
        left_hip = landmarks.landmark[23]  # Left hip
        left_knee = landmarks.landmark[25]  # Left knee
        left_ankle = landmarks.landmark[27]  # Left ankle
        left_toe = landmarks.landmark[31]  # Left toe

        # Right Side
        right_hip = landmarks.landmark[24]  # Right hip
        right_knee = landmarks.landmark[26]  # Right knee
        right_ankle = landmarks.landmark[28]  # Right ankle
        right_toe = landmarks.landmark[32]  # Right toe

        left_squat_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_squat_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

        # Calculate lateral deviation for both knees (positive if knee is ahead, negative if behind)
        left_lateral_deviation = left_knee.x - left_toe.x
        right_lateral_deviation = right_toe.x - right_knee.x


        # Map squat depth to a factor between 0 and 1 based on the squat angle (deeper squat -> higher factor)
        left_depth_factor = max(0, min(1, (135 - left_squat_angle) / 45))
        right_depth_factor = max(0, min(1, (135 - right_squat_angle) / 45))

        # Initialize penalty for both sides
        left_penalty = 0.0
        right_penalty = 0.0

        # Left side penalty calculation
        if left_lateral_deviation > 0:  # If knee is ahead of toe
            normalized_deviation = min(1.0, abs(left_lateral_deviation))
            left_penalty = (1 - math.exp(-normalized_deviation * 5)) * left_depth_factor

            # Right side penalty calculation
        if right_lateral_deviation > 0:  # If knee is ahead of toe

            normalized_deviation = min(1.0, abs(right_lateral_deviation))
            right_penalty = (1 - math.exp(-normalized_deviation * 5)) * right_depth_factor


        # Calculate the score for each side
        left_score = self.max_score * (1 - left_penalty)
        right_score = self.max_score * (1 - right_penalty)

        if left_penalty > 0 or right_penalty > 0:
           print("")
        #    print(f"Left Knee Score: {left_score:.2f}, Right Knee Score: {right_score:.2f}")
        # Select the lowest score from both sides

        lowest_score = min(left_score, right_score)

        # Track the lowest knee deviation score
        if lowest_score < self.score_knee_deviation:
            self.score_knee_deviation = lowest_score
            # print(f"New Lowest Knee Deviation Score: {self.score_knee_deviation:.2f}")

        return lowest_score

    def evaluate_shoulder_alignment(self, landmarks):
        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]

        shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)

        threshold = 0.002
        max_tilt = 0.01

        if shoulder_tilt <= threshold:
            score = self.max_score
        elif shoulder_tilt >= max_tilt:
            score = self.min_score
        else:
            penalty_ratio = (shoulder_tilt - threshold) / (max_tilt - threshold)
            score = self.max_score - penalty_ratio * (self.max_score - self.min_score)

        self.shoulder_scores.append(score)
        return score

    def get_average_shoulder_score(self):
        if not self.shoulder_scores:
            return self.max_score  # Assume perfect if no data

        return sum(self.shoulder_scores) / len(self.shoulder_scores)

    def evaluate_knee_alignment(self, landmarks):
        left_knee = landmarks.landmark[25]
        right_knee = landmarks.landmark[26]

        knee_tilt = abs(left_knee.y - right_knee.y)

        threshold = 0.005
        max_tilt = 0.025

        if knee_tilt <= threshold:
            score = self.max_score
        elif knee_tilt >= max_tilt:
            score = self.min_score
        else:
            penalty_ratio = (knee_tilt - threshold) / (max_tilt - threshold)
            score = self.max_score - penalty_ratio * (self.max_score - self.min_score)

        self.knee_alignment_scores.append(score)

        return score


    def get_average_knee_alignment_score(self):
        if not self.knee_alignment_scores:
            return self.max_score  # Assume perfect if no data

        return sum(self.knee_alignment_scores) / len(self.knee_alignment_scores)


    def calculate_overall_score(self,shoulder_score, depth_score, knee_tracking_score, knee_alignment_score):
        weights = {
            "shoulders": 0.25,
            "depth": 0.35,
            "knee_tracking": 0.25,
            "knee_alignment": 0.15
        }
        overall_score = (
                weights["shoulders"] * shoulder_score +
                weights["depth"] * depth_score +
                weights["knee_tracking"] * knee_tracking_score +
                weights["knee_alignment"] * knee_alignment_score
        )
        return overall_score

    def get_total_shoulder_score(self):
        if not self.all_rep_shoulder_scores:
            return self.max_score  # default
        return sum(self.all_rep_shoulder_scores) / len(self.all_rep_shoulder_scores)

    def get_total_depth_score(self):
        if not self.all_rep_depth_scores:
            return self.max_score  # default
        return sum(self.all_rep_depth_scores) / len(self.all_rep_depth_scores)

    def get_total_knee_tracking_score(self):
        if not self.all_rep_knee_tracking_scores:
            return self.max_score  # default
        return sum(self.all_rep_knee_tracking_scores) / len(self.all_rep_knee_tracking_scores)

    def get_total_knee_alignment_score(self):
        if not self.all_rep_knee_alignment_scores:
            return self.max_score  # default
        return sum(self.all_rep_knee_alignment_scores) / len(self.all_rep_knee_alignment_scores)





    def get_total_exercise_score(self):
        if not self.all_rep_overall_scores:
            return self.max_score
        return sum(self.all_rep_overall_scores) / len(self.all_rep_overall_scores)



    def generate_feedback(self,shoulders, depth, knee_tracking, knee_alignment):
        """
        Receives the average scores for each squat parameter and returns a list of personalized feedback messages.
        """
        feedback = []

        if depth < 15:
            feedback.append("🔻 Your squat depth was insufficient – try to lower yourself more while keeping your heels grounded.")

        if shoulders < 16:
            feedback.append("↔️ Your shoulder alignment was off – work on stabilizing your upper body and keeping the bar centered.")

        if knee_tracking < 14:
            feedback.append("🦵 Your knees moved too far forward – control the descent and focus on activating your posterior chain.")

        if knee_alignment < 16:
            feedback.append("🦵 Your knees were not level – ensure you're distributing your weight evenly between both legs.")

        if not feedback:
            feedback.append("✅ Great job! All your technique metrics were within the acceptable range. Keep it up!")

        return feedback
