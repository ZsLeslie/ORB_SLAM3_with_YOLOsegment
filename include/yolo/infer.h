#ifndef INFER_H
#define INFER_H

#include <opencv2/opencv.hpp>
#include "public.h"
#include "types.h"
#include "config.h"

using namespace nvinfer1;


class YoloDetector
{
public:
    YoloDetector(const std::string trtFile);
    ~YoloDetector();
    YoloDetector(const YoloDetector&) = delete;
    YoloDetector& operator=(const YoloDetector&) = delete;
    std::vector<Detection> inference(cv::Mat& img);
    static void draw_image(cv::Mat& img, std::vector<Detection>& inferResult, bool drawBbox=true);

private:
    void get_engine();
    void release() noexcept;
    static void process_mask(
        float* protoDevice, Dims32 protoOutDims, std::vector<Detection>& vDetections, 
        int kInputH, int kInputW, cv::Mat& img, cudaStream_t stream
    );

private:
    Logger              gLogger;
    std::string         trtFile_;

    ICudaEngine *       engine = nullptr;
    IRuntime *          runtime = nullptr;
    IExecutionContext * context = nullptr;

    cudaStream_t        stream = nullptr;

    float *             outputData = nullptr;
    std::vector<void *> vBufferD;
    float *             transposeDevice = nullptr;
    float *             decodeDevice = nullptr;

    int                 inputBindingIndex = -1;
    int                 detectionBindingIndex = -1;
    int                 protoBindingIndex = -1;

    int                 OUTPUT_CANDIDATES = 0;  // 8400 at 640 x 640
    Dims32              protoOutDims{};  // proto shape [1 32 160 160]
};

#endif  // INFER_H
