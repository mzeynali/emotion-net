from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort
from scipy.special import softmax


class EmotionNet:
    """
    This class takes in a checkpoint path and initializes the inference
    session and use it to calculate landmarks, emotion, valence, and arousal
    of a face.

    :param checkpoint_path: The path to the model checkpoint
    :type checkpoint_path: str
    """

    def __init__(self, checkpoint_path: str):
        self.emotions_list = {
            0: "Neutral",
            1: "Happiness",
            2: "Sadness",
            3: "Surprise",
            4: "Fear",
            5: "Disgust",
            6: "Anger",
            7: "Contempt",
        }
        self.emonet = ort.InferenceSession(
            checkpoint_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.emonet.get_inputs()[0].name
        self.landmarks_name = self.emonet.get_outputs()[0].name
        self.emotion_name = self.emonet.get_outputs()[1].name
        self.valence_name = self.emonet.get_outputs()[2].name
        self.arousal_name = self.emonet.get_outputs()[3].name

    def __call__(
        self, frame: np.ndarray, bounding_box: Tuple[int, int, int, int]
    ) -> Tuple[Tuple[str, float], float, float]:
        """
        The function takes in a frame and a bounding box, and returns a tuple
        of the predicted emotion, valence, and arousal.

        :param frame: the frame to be analyzed
        :type frame: np.ndarray
        :param bounding_box: The bounding box of the face in the image
            in format xyxy [x1, y1, x2, y2]
        :type bounding_box: Tuple[int, int, int, int]
        """

        # emotion analysis
        if isinstance(bounding_box, np.ndarray):
            bounding_box = bounding_box.flatten().tolist()
        emonet_input = preprocess_emonet(bounding_box, frame)
        onnx_pred = self.emonet.run(
            [
                self.landmarks_name,
                self.emotion_name,
                self.valence_name,
                self.arousal_name,
            ],
            {self.input_name: emonet_input},
        )
        landmarks_out, emotion_out, valence_out, arousal_out = onnx_pred
        pred_emo = [
            self.emotions_list[np.argmax(emotion_out)],
            np.max(softmax(emotion_out)),
        ]
        valence = np.clip(valence_out, -1.0, 1.0).item()
        arousal = np.clip(arousal_out, -1.0, 1.0).item()
        return pred_emo, valence, arousal


def get_transform(
    center: Tuple[int, int], scale: float, output_size: Tuple[int, int], rot=0
):
    """
    It takes in a center, scale, and rotation, and returns a transformation
    matrix that can be used to transform the image.

    :param center: The center of the image
    :type center: Tuple[int, int]
    :param scale: The scale of the image
    :type scale: float
    :param res: the resolution of the image
    :type res: Tuple[int, int]
    :param rot: rotation angle, defaults to 0 (optional)
    :return: A transformation matrix
    """
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(output_size[1]) / h
    t[1, 1] = float(output_size[0]) / h
    t[0, 2] = output_size[1] * (-float(center[0]) / h + 0.5)
    t[1, 2] = output_size[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1

    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 200
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -output_size[1] / 2
        t_mat[1, 2] = -output_size[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))

    return t


def get_scale_center(
    bounding_box: Tuple[int, int, int, int]
) -> Tuple[float, Tuple[int, int]]:
    """
    It takes a bounding box and returns the scale and center of the bounding
    box based on height and width.

    :param bounding_box: The bounding box of the person in the image
        in format xyxy [x1, y1, x2, y2]
    :type bounding_box: Tuple[int, int, int, int]
    :return: The scale and center of the bounding box.
    """
    center = np.array(
        [
            bounding_box[2] - (bounding_box[2] - bounding_box[0]) / 2,
            bounding_box[3] - (bounding_box[3] - bounding_box[1]) / 2,
        ]
    )
    scale = (
        bounding_box[2] - bounding_box[0] + bounding_box[3] - bounding_box[1]
    ) / 220.0
    return scale, center


def transform_image_shape(
    image: np.ndarray, bounding_box: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    It takes an image and a bounding box, and returns a new image that
    is cropped to the bounding box, warped to 256x256 based on center
    and scale.

    :param image: the image to be transformed
    :type image: np.ndarray
    :param bounding_box: bounding box
    :type bounding_box: Tuple[int, int, int, int]
    :return: The image is being transformed to a 256x256 image.
    """
    scale, center = get_scale_center(bounding_box)
    mat = get_transform(center, scale, (256, 256), 0)[:2]
    image = cv2.warpAffine(image, mat, (256, 256))

    return image


def preprocess_emonet(
    bounding_box: Tuple[int, int, int, int], image: np.ndarray
) -> np.ndarray:
    """
    We take the image, crop it to the bounding box, resize it to 256x256,
    and then transpose it to the format that the model expects.

    :param bounding_box: Tuple[int, int, int, int] in xyxy format
        [x1, y1, x2, y2]
    :type bounding_box: Tuple[int, int, int, int]
    :param image: The image to be processed
    :type image: np.ndarray
    """
    h, w = image.shape[:2]

    if bounding_box:
        image = transform_image_shape(image, bounding_box=bounding_box)
    else:
        image = transform_image_shape(image, bounding_box=[0, 0, w, h])

    image = cv2.resize(image, (256, 256)) / 255
    image = image.transpose(2, 0, 1)
    image = image[np.newaxis, :]
    return image.astype(np.float32)
