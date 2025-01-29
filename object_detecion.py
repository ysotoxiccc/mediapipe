import cv2
import mediapipe as mp
import numpy as np


MARGIN = 10  # пиксели
ROW_SIZE = 10  # пиксели
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

counter = 0

def visualize(
    image,
    detection_result) -> np.ndarray:

  for detection in detection_result.detections:
    # Отрисовка рамки
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height


    # Подпись
    category = detection.categories[0]
    category_name = category.category_name

    cv2.rectangle(image, start_point, end_point, (255, 0, 0), 3)

    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image
#Захват видеопотока и его параметров
cam = cv2.VideoCapture(0)
timestamps = [cam.get(cv2.CAP_PROP_POS_MSEC)]
fps = cam.get(cv2.CAP_PROP_FPS)
calc_timestamps = [0.0]


# Получаем ширину и высоту кадра
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

#инициализация детектора объектов
BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.components.containers.DetectionResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

detection_result_list = []

#параметры медиапайп
options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='efficientdet.tflite'),
    running_mode=VisionRunningMode.VIDEO,
    score_threshold=0.5,
    max_results=5,
    category_allowlist=['person']

    )

with ObjectDetector.create_from_options(options) as detector:
    while True:
        ret, frame = cam.read()
        timestamps.append(cam.get(cv2.CAP_PROP_POS_MSEC))
        calc_timestamps.append(calc_timestamps[-1] + 1000 / fps)
        image = cv2.flip(frame, 1)
        counter+=1

        # Перевод из брг в ргб
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        frame_timestamp_ms = 1000 * counter / fps

        # Запуск процесса детекции
        detection_result = detector.detect_for_video(mp_image, counter)
        current_frame = mp_image.numpy_view()
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
        # отображение кадра

        vis_image = visualize(current_frame, detection_result)
        cv2.imshow('object_detector', vis_image)
        detection_result_list.clear()


        if cv2.waitKey(1) == 27:
            break


        if cv2.waitKey(1) == ord('q'):
            break


    cam.release()

    cv2.destroyAllWindows()
