import os
from imageai.Detection import ObjectDetection
from typing import List


exec_path = os.path.dirname(__file__)


def find_img() -> str:
    return _find_first_file(['.jpg', '.jpeg', '.png'])


def find_model() -> str:
    return _find_first_file(['.h5'])
    

def _find_first_file(file_exts: List[str]) -> str:
    for file in os.listdir(exec_path):
        ext = os.path.splitext(file)[1]
        if ext.lower() in file_exts:
            return os.path.join(exec_path, file)


def start_obj_detection_from_img(input_img_path: str, model_path: str) -> None:
    if not input_img_path or not os.path.exists(input_img_path):
        print('Input image doesn\'t exist! Please provide one.')
        return
    img_name, img_ext = os.path.splitext(input_img_path)

    if not os.path.exists(model_path):
        print('Model doesn\'t exist! Make sure to download it. (https://imageai.readthedocs.io/en/latest/detection/index.html)')
        return

    output_image_path = os.path.join(exec_path, f'{img_name}_result{img_ext}')

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path)
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=input_img_path, output_image_path=output_image_path)

    for obj in detections:
        print(obj['name'], ' : ', obj['percentage_probability'])


if __name__ == '__main__':
    start_obj_detection_from_img(find_img(), find_model())
