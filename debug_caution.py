# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, ".")
from src.ocr.section_detector import OCRSectionDetector

d = OCRSectionDetector()
line6 = "+활성-TECA'30, 000 ppm 함유 붉은 반점 부어오름 또는 가려움증 등의 이상증상이나 부작용이 있는 경우에는 전문의 등과 상담할것 2.상처가 있는 부위 등에는사용을자제할것3.보관 및 취급 시 주의사항"
line7 = "+병풀단백질추출물 함유 가) 어린이의 손이 닿지 않는곳에 보관할 것 나) 직사광선을 피해서 보관할 것 4.화장품이 눈에 들어갔을 때에는 물로씻어내고 이상이있는 경우에는 전문의와상담할것"
for i, t in enumerate([line6, line7], start=6):
    print(f"--- line {i} ---")
    print("strong caution:", d._has_strong_caution_signal(t))
    print("imperative matches:", d.CAUTION_IMPERATIVE_PATTERN.findall(t))
    print("manufacturer/meta:", d._is_manufacturer_or_meta_text(t))
    print("advertising:", d._looks_like_advertising_copy(t))
    print("implicit:", d._detect_implicit_section({"text": t}))
