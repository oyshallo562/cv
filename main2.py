from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO('./best.onnx')  # 사전 훈련된 YOLOv8n 모델

# 카메라 열기
cap = cv2.VideoCapture(0)

try:
    while True:
        # 카메라에서 이미지 캡처
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 BGR에서 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 모델에 이미지 입력
        results = model([frame_rgb])

        # 결과 처리 및 바운딩 박스 그리기
        for result in results:
            for (x1, y1, x2, y2, conf, cls) in result.boxes.data:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # 처리된 이미지 표시
        cv2.imshow('YOLO Detection', frame)

        # 0.1초 대기하고 'q'를 누르면 종료
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
finally:
    # 카메라 자원 해제 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()
