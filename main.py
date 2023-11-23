import cv2
import onnxruntime as ort
import numpy as np

# 모델 파일과 입력 이미지 크기 설정
model_path = './best.onnx'
input_size = (640, 640)

# ONNX 런타임 세션 초기화
ort_session = ort.InferenceSession(model_path)



# 이미지 전처리 함수
def preprocess(frame):
    # 이미지 크기 조정 및 정규화
    image = cv2.resize(frame, input_size)
    image = image.astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def draw_boxes(frame, boxes, confidences, class_ids, classes):
    for (box, confidence, class_id) in zip(boxes, confidences, class_ids):
        (x1, y1, x2, y2) = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{classes[class_id]}: {confidence:.2f}"
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def postprocess(frame, results):
    for result in results:
       print(result.shape)
       print(result)


    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9","10","11","12","13","14","15","16","17","18","19","20","21","22","23",
                   "24","25","26","27","28"]

# 카메라 초기화
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 전처리
    preprocessed = preprocess(frame)
    # ONNX 모델 실행
    outputs = ort_session.run(None, {'images': preprocessed})

    # 후처리 및 결과 시각화
    postprocess(frame, outputs)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
