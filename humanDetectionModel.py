import logging

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG

class HumanDetectionModel:
    def __init__(self, model_path, threshold):
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
        )
        self.detector = FaceDetector.create_from_options(options)

        self.threshold = threshold

    def detect_faces(self, image_path):
        image = mp.Image.create_from_file(image_path)
        detection_result = self.detector.detect(image)

        for detection in detection_result.detections:
            for category in detection.categories:
                logger.info("possible human face in {} with score {}".format(image_path, category.score))
                if category.score > self.threshold:
                    return True

        return False
