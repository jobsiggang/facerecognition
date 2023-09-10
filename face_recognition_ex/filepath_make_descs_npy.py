import dlib
import cv2
import numpy as np
import os

# 얼굴 감지기, 랜드마크 예측기, 얼굴 인식 모델 초기화
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models\shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

def find_faces(img):
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=int)
    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d)

        # convert dlib shape to numpy array
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)

    return rects, shapes, shapes_np

def encode_faces(img, shapes):
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)

# 이미지 파일이 있는 디렉토리 설정
image_directory = 'img'

# 이미지 파일 디렉토리에서 jpg 확장자만 있는 파일 목록 읽기
image_files = []
for filename in os.listdir(image_directory):
    if filename.endswith('.jpg'):
        image_files.append(filename)

# 이미지 파일 이름을 얼굴 디스크립터에 매핑할 딕셔너리 초기화
descs = {}

# 각 이미지 파일에 대해 얼굴 디스크립터 계산
for image_file in image_files:
    # 확장자를 제외한 파일명 추출
    person_name = os.path.splitext(image_file)[0]

    # 이미지 파일 읽기
    img_bgr = cv2.imread(os.path.join(image_directory, image_file))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 이미지에서 얼굴 찾기
    _, img_shapes, _ = find_faces(img_rgb)
    print(person_name)
    # 얼굴 디스크립터 생성 및 저장
    if len(img_shapes) > 0:
        descs[person_name] = encode_faces(img_rgb, img_shapes)[0]
    else:
        descs[person_name] = None

# 생성된 디스크립터를 파일에 저장
np.save('img/descs.npy', descs)

# 생성된 디스크립터 출력
print(descs)
