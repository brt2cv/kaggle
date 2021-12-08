#!/bin/bash

paddle2onnx --model_dir models/ch_PP-OCRv2/ch_PP-OCRv2_det_infer  --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file models/ch_onnx_v2.1/det.onnx --opset_version 11 --enable_onnx_checker True
paddle2onnx --model_dir models/ch_PP-OCRv2/ch_ppocr_mobile_v2.0_cls_infer  --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file models/ch_onnx_v2.1/cls.onnx --opset_version 11 --enable_onnx_checker True
paddle2onnx --model_dir models/ch_PP-OCRv2/ch_PP-OCRv2_rec_infer  --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file models/ch_onnx_v2.1/rec.onnx --opset_version 11 --enable_onnx_checker True
