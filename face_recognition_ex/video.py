import dlib, cv2
import numpy as np

# 얼굴 감지기, 랜드마크 예측기, 얼굴 인식 모델 초기화
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models\shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

# 이미 저장된 얼굴 디스크립터 로드
descs = np.load('img/descs.npy', allow_pickle=True)[()]

# 얼굴을 인코딩하는 함수 정의
def encode_face(img):
  dets = detector(img, 1)

  if len(dets) == 0:
    return np.empty(0)

  for k, d in enumerate(dets):
    shape = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)

    return np.array(face_descriptor)

# 웹캠 또는 비디오 입력을 캡처
cap = cv2.VideoCapture(0)

# 캡처가 열리지 않으면 종료
if not cap.isOpened():
  exit()

# 초기 프레임 읽기
_, img_bgr = cap.read() # (800, 1920, 3)

# 비디오 화면 크기 조정을 위한 변수 설정
padding_size = 0
resized_width = 1920//3  # 화면 크기를 1/3로 줄임
video_size = (resized_width, int(img_bgr.shape[0] * resized_width // img_bgr.shape[1]))
output_size = (resized_width, int(img_bgr.shape[0] * resized_width // img_bgr.shape[1] + padding_size * 2))

# while 루프 시작
while True:
  # 프레임 읽기
  ret, img_bgr = cap.read()
  if not ret:
    break

  # 비디오 프레임 크기 조정
  img_bgr = cv2.resize(img_bgr, video_size)
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

  # 얼굴 감지
  dets = detector(img_bgr, 1)

  # 감지된 얼굴에 대해 루프 수행
  for k, d in enumerate(dets):
    # 얼굴 랜드마크 예측
    shape = sp(img_rgb, d)

    # 얼굴 디스크립터 추출
    face_descriptor = facerec.compute_face_descriptor(img_rgb, shape)

    # 최근 발견된 얼굴 초기화
    last_found = {'name': 'unknown', 'dist': 0.6, 'color': (0,0,255)}

    # 저장된 디스크립터와 현재 얼굴 디스크립터 간의 유사성 계산
    for name, saved_desc in descs.items():
      dist = np.linalg.norm([face_descriptor] - saved_desc, axis=1)

      # 가장 유사한 얼굴 정보 업데이트
      if dist < last_found['dist']:
        last_found = {'name': name, 'dist': dist, 'color': (255,255,255)}

    # 감지된 얼굴 주변에 사각형 그리고 이름 표시
    cv2.rectangle(img_bgr, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=last_found['color'], thickness=2)
    cv2.putText(img_bgr, last_found['name'], org=(d.left(), d.top()), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=last_found['color'], thickness=2)

  # 화면에 프레임 출력
  cv2.imshow('img', img_bgr)

  # 'q' 키를 누르면 종료
  if cv2.waitKey(1) == ord('q'):
    break

# 캡처 해제
cap.release()
