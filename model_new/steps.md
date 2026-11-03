I. ocr.py : use PP-OCRv6 det + rec from 'third_party' to pipeline extraction text  & bounding box norm [0,1000]
    1. PIl image 
    2. Text detection pp-ocrv6 det -> output : polygon 4 points [(x,y),(x,y),(x,y),(x,y)]
    3. transfer polygon to Bounding box [x_min, y_min, x_max, y_max] and  Reading order sorting 
    4. Text Recognition & CTC Decoding (đọc kí tự)
        - với mỗi bouding box, ta crop ảnh 
        - resize crop về chuẩn của Rec (chiều cao cố định 48px, chiều rọng tỉ lệ)
        - Rec dự đoán chuỗi phân phối xác suát kí tự
        - CTC Greedy Decoder -> chuyển chuỗi xác suất thành từ hoàn chỉnh (loại duplicate và token blank)
    5. chuẩn hóa về dải [0.1000] , vì layoutlmv3 sử dụng 2d trên thang cố định [1000]
    
    [configs/pp_ocrv6_small_det] -> PPOCRV6SmallDetConfig
                                        |
                                        v
    [checkpoints/pp_ocrv6_small_det] -> PPOCRV6SmallDetForObjectDetection -> Bounding Boxes
                                                                                | (Cắt ảnh)
                                                                                v
    [configs/pp_ocrv6_small_rec] -> PPOCRV6SmallRecImageProcessor (Resize 48px) -> Crop Tensors
                                        |                                           |
                                        v                                           v
    [configs/pp_ocrv6_small_rec] -> PPOCRV6SmallRecConfig                              |
                                        |                                           |
                                        v                                           v
    [checkpoints/pp_ocrv6_small_rec] -> PPOCRV6SmallRecForTextRecognition <------------+
                                        |
                                        v (Logits)
                                    PPOCRV6SmallRecImageProcessor (CTC Greedy Decode)
                                        |
                                        v
                                    Output Text (str)