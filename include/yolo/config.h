#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>


const int kGpuId = 0;
const int kNumClass = 80;
const int kNumMasks = 32;
const int kInputH = 640;
const int kInputW = 640;
const int kLetterboxValue = 114;
const float kNmsThresh = 0.45f;
const float kConfThresh = 0.25f;
const int kMaxNumOutputBbox = 1000;  // assume the box outputs no more than kMaxNumOutputBbox boxes that conf >= kNmsThresh;
const int kNumBoxElement = 7 + kNumMasks;  // bbox, confidence, class, keepflag, mask coefficients

// Run from the project root. Use raw one-to-many segmentation outputs:
// output0 [1, 4 + kNumClass + kNumMasks, N], output1 [1, kNumMasks, H, W].
const std::string onnxFile = "./model/yolo26m-seg.onnx";
const std::string kYoloEngineFile = "./model/yolo26m-seg.engine";

// for FP16 mode
const bool bFP16Mode = false;
// for INT8 mode
const bool bINT8Mode = false;
const std::string cacheFile = "./model/yolo26m-seg-int8.cache";
const std::string calibrationDataPath = "../calibrator";  // 存放用于 int8 量化校准的图像

const std::vector<std::string> vClassNames {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

#endif  // CONFIG_H
