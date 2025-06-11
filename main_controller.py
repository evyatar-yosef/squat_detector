from video_processor import VideoProcessor
from dominant_person import DominantPersonDetector
from squat_tracker import SquatTracker
from squat_assesor import SquatAssessor

class MainController:
    def __init__(self, video_path):
        self.video_processor = VideoProcessor(video_path)
        self.detector = DominantPersonDetector()
        self.squat_readiness_checker = SquatTracker()
        self.squat_assesor = SquatAssessor()
        self.frames_processed = 0
        self.rep_scored = False



    def run(self):
        def process_frame(frame):
            self.frames_processed += 1
            if self.frames_processed % 200 == 0:
                print("Reinitializing dominant person detector...")
                self.detector = DominantPersonDetector()

            landmarks = self.detector.find_dominant_person(frame)
            if not landmarks:
                return

            h, w, _ = frame.shape
            self.detector.draw_landmarks(frame, landmarks)
            self.detector.draw_bounding_box(frame, landmarks, h, w)

            if self.squat_readiness_checker.get_finish() == 1:
                self.handle_finished()
                return

            if self.squat_readiness_checker.get_ready() == 1 or self.squat_readiness_checker.get_inrep() == 1:
                self.handle_rep_progress(landmarks)
            else:
                self.squat_readiness_checker.is_ready_to_squat(landmarks)

        self.video_processor.process_video(process_frame, self.squat_readiness_checker)

    def handle_finished(self):
        final_score = self.squat_assesor.get_total_exercise_score()
   #     print(f"final score:{final_score:.2f}")
        self.print_final_scores()
        print("Workout Done")

    def handle_rep_progress(self, landmarks):
        self.squat_readiness_checker.squat_rep_position(landmarks)
        self.squat_readiness_checker.check_finished_transition(landmarks)

        if self.squat_readiness_checker.get_inrep() == 1:
            self.squat_assesor.evaluate_shoulder_alignment(landmarks)
            self.squat_assesor.evaluate_knee_alignment(landmarks)
            self.squat_assesor.get_depth(landmarks)
            self.squat_assesor.evaluate_knee_tracking(landmarks)
            self.rep_scored = False

        elif self.squat_readiness_checker.get_ready() == 1 and not self.rep_scored:
            self.squat_assesor.assess_squat(landmarks)
            self.squat_assesor.reset()
            self.rep_scored = True

    def print_final_scores(self):
        shoulder_avg = self.squat_assesor.get_total_shoulder_score()
        depth_avg = self.squat_assesor.get_total_depth_score()
        knee_track_avg = self.squat_assesor.get_total_knee_tracking_score()
        knee_align_avg = self.squat_assesor.get_total_knee_alignment_score()
        overall = self.squat_assesor.get_total_exercise_score()

        print("\n📊 Final Scores Summary:")
        print(f"   🟪 Shoulder Alignment:     {shoulder_avg:.2f}/20")
        print(f"   🟦 Depth Score:            {depth_avg:.2f}/20")
        print(f"   🟩 Knee Tracking:          {knee_track_avg:.2f}/20")
        print(f"   🟨 Knee Alignment:         {knee_align_avg:.2f}/20")
        print(f"   🟫 Overall Exercise Score: {overall:.2f}/20")

        feedback = self.squat_assesor.generate_feedback(shoulder_avg, depth_avg, knee_track_avg, knee_align_avg)

        print("\n💬 personal feedback:")
        for line in feedback:
            print(f"  - {line}")


# Run the application
if __name__ == "__main__":
    controller = MainController(video_path="squat_bad.mp4")
    controller.run()
