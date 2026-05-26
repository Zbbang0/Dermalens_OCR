from paddleocr import PaddleOCR
import numpy as np
import cv2

ocr = PaddleOCR(lang='korean', use_angle_cls=True, show_log=False)

img_path = r'C:\Users\User\Desktop\univ\2026\2026 -1\캡스톤\Dermalens_ai\images\tlqkf.jpg'
img_array = np.fromfile(img_path, dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# 전성분 구간만 크롭 (하단 40~70% 구간)
h, w = img.shape[:2]
crop = img[int(h*0.4):int(h*0.7), :]

# 이미지 반전 (금색 배경 → 흰 배경)
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
inverted = cv2.bitwise_not(gray)

# 확대
scale = 3.0
enlarged = cv2.resize(inverted, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
bgr = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)

result = ocr.ocr(bgr, cls=True)
print("=== 전성분 구간 OCR ===")
for line in result[0]:
    print(line[1][0], '|', round(line[1][1], 2))