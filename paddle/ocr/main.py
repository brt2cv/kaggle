#!/usr/bin/env python3
# @Date    : 2021-12-07
# @Author  : Bright (brt2@qq.com)
# @Link    : https://gitee.com/brt2

# 参考: https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/paddleocr.py
# 模型下载: https://github.com/brt2cv/kaggle/releases/download/PP-OCRv2.1/models.7z

from tools.infer.predict_system import TextSystem
from tools.infer.utility import parse_args

import cv2

if __name__ == "__main__":
    # version 2.1
    args = parse_args()

    args.use_gpu = False
    args.cls_model_dir = "models/ch_PP-OCRv2/ch_ppocr_mobile_v2.0_cls_infer"
    args.det_model_dir = "models/ch_PP-OCRv2/ch_PP-OCRv2_det_infer"
    args.rec_model_dir = "models/ch_PP-OCRv2/ch_PP-OCRv2_rec_infer"

    engine = TextSystem(args)
    im = cv2.imread("test.jpg")
    result = engine(im)  # det & rec
    if result is not None:
        dt_box, tuple_rec = result
        for text, score in tuple_rec:
            # print(text)
            print(f"[{score:.2f}] {text}")
