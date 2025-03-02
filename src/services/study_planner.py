from src.models.study_plan import StudyPlanRequest


class StudyPlannerService:
    """
    Service class to generate study plan based on user input
    """
    @staticmethod
    def generate_plan(request: StudyPlanRequest) -> dict:
        """
        Generate study plan based on user input
        :param request:  StudyPlanRequest object
        :return:  dict
        """
        # Simple algorithm to generate study plan
        # In a real application, this would be more sophisticated
        adhd_severity = int(request.adhd_severity)

        # Adjust session time based on ADHD severity
        base_session_time = 45 - (adhd_severity * 5)
        optimal_session_time = max(15, min(45, base_session_time))

        # Calculate splits
        total_study_time = request.study_duration
        splits = total_study_time // optimal_session_time
        remaining_time = total_study_time % optimal_session_time

        return {
            "optimal_session_time": float(optimal_session_time),
            "break_interval": float(request.break_duration),
            "splits": splits,
            "remaining_time": float(remaining_time)
        }
