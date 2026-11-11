import unittest

import numpy as np

from scout.detectors import COCO_NAMES, TensorRTDetector, letterbox_geometry


class Yolo26DecodeTest(unittest.TestCase):
    def test_4k_input_preserves_pixels_with_stride_padding(self):
        self.assertEqual(letterbox_geometry(3840, 2160, 3840, 2176),
                         (1.0, 3840, 2160, 0, 8))

    def test_end_to_end_rows_are_filtered_and_unpadded(self):
        detector = TensorRTDetector.__new__(TensorRTDetector)
        detector.class_names = COCO_NAMES
        detector.class_confidences = {}
        detector.confidence = .25
        detector.iou = .45
        detector.max_box_aspect_ratio = 0
        output = np.zeros((1, 300, 6), dtype=np.float32)
        output[0, 0] = [100, 108, 300, 408, .91, 0]
        output[0, 1] = [102, 110, 298, 406, .80, 0]
        output[0, 2] = [500, 108, 700, 408, .78, 2]
        output[0, 3] = [0, 0, 10, 10, .05, 0]
        detections = detector._decode(output, 1.0, 0, 8, 1280, 720)
        self.assertEqual(len(detections), 2)
        self.assertEqual([d.class_id for d in detections], [0, 2])
        self.assertEqual((detections[0].x1, detections[0].y1, detections[0].x2, detections[0].y2),
                         (100, 100, 300, 400))
        self.assertAlmostEqual(detections[0].confidence, .91, places=2)


if __name__ == '__main__':
    unittest.main()
