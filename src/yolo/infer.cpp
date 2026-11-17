#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>

#include <NvOnnxParser.h>

#include "infer.h"
#include "preprocess.h"
#include "postprocess.h"
#include "calibrator.h"
#include "utils.h"
#include "draw.h"

using namespace nvinfer1;

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

size_t tensor_elements(const Dims32& dims) {
    size_t elements = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("Unresolved tensor shape: " + shapeToString(dims));
        }
        elements *= static_cast<size_t>(dims.d[i]);
    }
    return elements;
}

}  // namespace


YoloDetector::YoloDetector(const std::string trtFile): trtFile_(trtFile)
{
    gLogger = Logger(ILogger::Severity::kERROR);
    try {
        check_cuda(cudaSetDevice(kGpuId), "Select CUDA device");
        check_cuda(cudaStreamCreate(&stream), "Create CUDA stream");
        get_engine();

        // TensorRT assigns binding indices when building the engine. Resolve by name.
        inputBindingIndex = engine->getBindingIndex("images");
        detectionBindingIndex = engine->getBindingIndex("output0");
        protoBindingIndex = engine->getBindingIndex("output1");
        if (engine->getNbBindings() != 3 || inputBindingIndex < 0 ||
            detectionBindingIndex < 0 || protoBindingIndex < 0) {
            throw std::runtime_error("Expected one input 'images' and two outputs 'output0', 'output1'.");
        }
        if (!engine->bindingIsInput(inputBindingIndex) ||
            engine->bindingIsInput(detectionBindingIndex) || engine->bindingIsInput(protoBindingIndex)) {
            throw std::runtime_error("Unexpected input/output roles for images, output0 or output1.");
        }
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            if (engine->getBindingDataType(i) != DataType::kFLOAT ||
                engine->getBindingFormat(i) != TensorFormat::kLINEAR) {
                throw std::runtime_error(std::string("Expected FP32 linear I/O for tensor '") +
                                         engine->getBindingName(i) + "'.");
            }
        }

        context = engine->createExecutionContext();
        if (!context) {
            throw std::runtime_error("Failed creating TensorRT execution context.");
        }
        Dims32 inputDims = engine->getBindingDimensions(inputBindingIndex);
        bool dynamicInput = false;
        for (int i = 0; i < inputDims.nbDims; ++i) {
            dynamicInput = dynamicInput || inputDims.d[i] == -1;
        }
        if (dynamicInput && !context->setBindingDimensions(
                inputBindingIndex, Dims32 {4, {1, 3, kInputH, kInputW}})) {
            throw std::runtime_error("Failed setting TensorRT input dimensions.");
        }
        inputDims = context->getBindingDimensions(inputBindingIndex);
        if (inputDims.nbDims != 4 || inputDims.d[0] != 1 || inputDims.d[1] != 3 ||
            inputDims.d[2] != kInputH || inputDims.d[3] != kInputW) {
            throw std::runtime_error("Unexpected images shape " + shapeToString(inputDims) +
                                     "; expected (1, 3, " + std::to_string(kInputH) + ", " +
                                     std::to_string(kInputW) + ").");
        }

        Dims32 outDims = context->getBindingDimensions(detectionBindingIndex);
        if (outDims.nbDims != 3 || outDims.d[0] != 1 ||
            outDims.d[1] != 4 + kNumClass + kNumMasks || outDims.d[2] <= 0) {
            throw std::runtime_error("Unsupported output0 shape " + shapeToString(outDims) +
                "; expected (1, " + std::to_string(4 + kNumClass + kNumMasks) +
                ", N) with xywh, class scores and mask coefficients. "
                "Export raw one-to-many outputs (Ultralytics 8.4.211: nms=None); "
                "end-to-end/embedded-NMS outputs are not supported.");
        }
        protoOutDims = context->getBindingDimensions(protoBindingIndex);
        if (protoOutDims.nbDims != 4 || protoOutDims.d[0] != 1 ||
            protoOutDims.d[1] != kNumMasks || protoOutDims.d[2] <= 0 || protoOutDims.d[3] <= 0) {
            throw std::runtime_error("Unsupported output1 shape " + shapeToString(protoOutDims) +
                                     "; expected (1, " + std::to_string(kNumMasks) + ", H, W).");
        }
        OUTPUT_CANDIDATES = outDims.d[2];

        vBufferD.resize(engine->getNbBindings(), nullptr);
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            Dims32 dims = context->getBindingDimensions(i);
            std::cout << "Tensor " << engine->getBindingName(i) << ": binding " << i
                      << ", FP32 " << shapeToString(dims) << std::endl;
            check_cuda(cudaMalloc(&vBufferD[i], tensor_elements(dims) * sizeof(float)), "Allocate tensor buffer");
        }
        outputData = new float[1 + kMaxNumOutputBbox * kNumBoxElement];
        check_cuda(cudaMalloc(&transposeDevice, tensor_elements(outDims) * sizeof(float)), "Allocate transpose buffer");
        check_cuda(cudaMalloc(&decodeDevice, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float)), "Allocate decode buffer");
    } catch (...) {
        release();
        throw;
    }
}

void YoloDetector::get_engine(){
    runtime = createInferRuntime(gLogger);
    if (!runtime) {
        throw std::runtime_error("Failed creating TensorRT runtime.");
    }
    if (access(trtFile_.c_str(), F_OK) == 0){
        std::ifstream engineFile(trtFile_, std::ios::binary | std::ios::ate);
        if (!engineFile) {
            throw std::runtime_error("Failed opening engine file: " + trtFile_);
        }
        const std::streamsize fsize = engineFile.tellg();
        if (fsize <= 0) {
            throw std::runtime_error("Empty or unreadable engine file: " + trtFile_);
        }
        engineFile.seekg(0, std::ios::beg);
        std::vector<char> engineString(static_cast<size_t>(fsize));
        if (!engineFile.read(engineString.data(), fsize)) {
            throw std::runtime_error("Failed reading engine file: " + trtFile_);
        }
        engine = runtime->deserializeCudaEngine(engineString.data(), engineString.size());
        if (!engine) {
            throw std::runtime_error("Failed loading engine: " + trtFile_ + ". Remove the stale engine file and rebuild it.");
        }
        std::cout << "Loaded engine: " << trtFile_ << std::endl;
    } else {
        std::unique_ptr<IBuilder> builder(createInferBuilder(gLogger));
        if (!builder) {
            throw std::runtime_error("Failed creating TensorRT builder.");
        }
        std::unique_ptr<INetworkDefinition> network(
            builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
        std::unique_ptr<Int8EntropyCalibrator2> calibrator;
        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
        if (!network || !config) {
            throw std::runtime_error("Failed creating TensorRT network or builder configuration.");
        }
        config->setMaxWorkspaceSize(1 << 30);
        if (bFP16Mode){
            config->setFlag(BuilderFlag::kFP16);
        }
        if (bINT8Mode){
            config->setFlag(BuilderFlag::kINT8);
            int batchSize = 8;
            calibrator.reset(new Int8EntropyCalibrator2(batchSize, kInputW, kInputH, calibrationDataPath.c_str(), cacheFile.c_str()));
            config->setInt8Calibrator(calibrator.get());
        }

        std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger));
        if (!parser) {
            throw std::runtime_error("Failed creating ONNX parser.");
        }
        if (!parser->parseFromFile(onnxFile.c_str(), int(gLogger.reportableSeverity))){
            std::cout << std::string("Failed parsing .onnx file!") << std::endl;
            for (int i = 0; i < parser->getNbErrors(); ++i){
                auto *error = parser->getError(i);
                std::cout << std::to_string(int(error->code())) << std::string(":") << std::string(error->desc()) << std::endl;
            }
            throw std::runtime_error("Failed parsing ONNX model: " + onnxFile);
        }
        std::cout << std::string("Succeeded parsing .onnx file!") << std::endl;

        if (network->getNbInputs() != 1 || network->getNbOutputs() != 2) {
            throw std::runtime_error("Expected one image input and two segmentation outputs.");
        }
        ITensor* inputTensor = network->getInput(0);
        Dims32 inputDims = inputTensor->getDimensions();
        bool dynamicInput = false;
        for (int i = 0; i < inputDims.nbDims; ++i) {
            dynamicInput = dynamicInput || inputDims.d[i] == -1;
        }
        if (dynamicInput) {
            // The builder owns the optimization profile.
            IOptimizationProfile* profile = builder->createOptimizationProfile();
            const Dims32 dims {4, {1, 3, kInputH, kInputW}};
            if (!profile || !profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, dims) ||
                !profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, dims) ||
                !profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, dims) ||
                config->addOptimizationProfile(profile) < 0) {
                throw std::runtime_error("Failed configuring TensorRT optimization profile.");
            }
        }

        std::cout << "Building engine from: " << onnxFile << std::endl;
        std::unique_ptr<IHostMemory> engineString(builder->buildSerializedNetwork(*network, *config));
        if (!engineString) {
            throw std::runtime_error("Failed building TensorRT engine from: " + onnxFile);
        }
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (!engine) {
            throw std::runtime_error("Failed deserializing the newly built engine.");
        }
        std::cout << "Succeeded building engine!" << std::endl;

        std::ofstream engineFile(trtFile_, std::ios::binary);
        if (!engineFile.write(static_cast<const char *>(engineString->data()), engineString->size()) ||
            !engineFile.flush()) {
            throw std::runtime_error("Failed saving engine file: " + trtFile_);
        }
        std::cout << "Saved engine: " << trtFile_ << std::endl;
    }
}

YoloDetector::~YoloDetector(){
    release();
}

void YoloDetector::release() noexcept {
    if (stream) {
        CHECK(cudaStreamSynchronize(stream));
    }
    for (void* buffer : vBufferD) {
        if (buffer) CHECK(cudaFree(buffer));
    }
    vBufferD.clear();
    if (transposeDevice) CHECK(cudaFree(transposeDevice));
    if (decodeDevice) CHECK(cudaFree(decodeDevice));
    transposeDevice = nullptr;
    decodeDevice = nullptr;
    delete [] outputData;
    outputData = nullptr;
    delete context;
    delete engine;
    delete runtime;
    context = nullptr;
    engine = nullptr;
    runtime = nullptr;
    if (stream) CHECK(cudaStreamDestroy(stream));
    stream = nullptr;
}

std::vector<Detection> YoloDetector::inference(cv::Mat& img){
    if (img.empty()) return {};

    // put input on device, then letterbox、bgr to rgb、hwc to chw、normalize.
    preprocess(img, (float*)vBufferD[inputBindingIndex], kInputH, kInputW, stream);

    // tensorrt inference
    if (!context->enqueueV2(vBufferD.data(), stream, nullptr)) {
        throw std::runtime_error("TensorRT inference failed.");
    }

    // transpose [116 8400] convert to [8400 116]
    transpose((float*)vBufferD[detectionBindingIndex], transposeDevice, OUTPUT_CANDIDATES, 4 + kNumClass + kNumMasks, stream);
    // convert [8400 116] to [39001, ], 39001 = 1 + 1000 * (4bbox + cond + cls + keepflag + 32masks)
    decode(transposeDevice, decodeDevice, OUTPUT_CANDIDATES, kNumClass, kNumMasks, kConfThresh, kMaxNumOutputBbox, kNumBoxElement, stream);
    // cuda nms
    nms(decodeDevice, kNmsThresh, kMaxNumOutputBbox, kNumBoxElement, stream);
    check_cuda(cudaGetLastError(), "Launch detection postprocessing");
    check_cuda(cudaMemcpyAsync(outputData, decodeDevice, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float), cudaMemcpyDeviceToHost, stream), "Copy detections to host");
    check_cuda(cudaStreamSynchronize(stream), "Finish detection postprocessing");

    std::vector<Detection> vDetections;
    int count = std::min((int)outputData[0], kMaxNumOutputBbox);
    for (int i = 0; i < count; i++){
        int pos = 1 + i * kNumBoxElement;
        int keepFlag = (int)outputData[pos + 6];
        if (keepFlag == 1){
            Detection det;
            memcpy(det.bbox, &outputData[pos], 4 * sizeof(float));
            det.conf = outputData[pos + 4];
            det.classId = (int)outputData[pos + 5];
            memcpy(det.mask, &outputData[pos + 7], kNumMasks * sizeof(float));
            vDetections.push_back(det);
        }
    }

    process_mask((float*)vBufferD[protoBindingIndex], protoOutDims, vDetections, kInputH, kInputW, img, stream);
    check_cuda(cudaStreamSynchronize(stream), "Finish mask postprocessing");

    for (size_t j = 0; j < vDetections.size(); j++){
        scale_bbox(img, vDetections[j].bbox);
    }

    return vDetections;
}

void YoloDetector::process_mask(
    float* protoDevice, Dims32 protoOutDims, std::vector<Detection>& vDetections,
    int kInputH, int kInputW, cv::Mat& img, cudaStream_t stream
){
    int protoC = protoOutDims.d[1];  // default 32
    int protoH = protoOutDims.d[2];  // default 160
    int protoW = protoOutDims.d[3];  // default 160

    int n = vDetections.size();  // number of bboxes
    if (n == 0) return;

    // prepare n x 32 length mask coef space on device
    float* maskCoefDevice = nullptr;
    CHECK(cudaMalloc(&maskCoefDevice, n * protoC * sizeof(float)));
    // prepare n x 160 x 160 mask space on device
    float* maskDevice = nullptr;
    CHECK(cudaMalloc(&maskDevice, n * protoH * protoW * sizeof(float)));

    float* bboxDevice = nullptr;  // x1,y1,x2,y2,x1,y1,x2,y2,...x1,y1,x2,y2
    CHECK(cudaMalloc(&bboxDevice, n * 4 * sizeof(float)));

    for (size_t i = 0; i < n; i++){
        CHECK(cudaMemcpyAsync(&maskCoefDevice[i * protoC], vDetections[i].mask, protoC * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(&bboxDevice[i * 4], vDetections[i].bbox, 4 * sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    // mask = sigmoid(mask coef x proto)
    matrix_multiply(maskCoefDevice, n, protoC, protoDevice, protoC, protoH * protoW, maskDevice, stream, true);

    // down sample bbox from 640x640 to 160x160
    float heightRatio = (float)protoH / (float)kInputH;  // 160 / 640 = 0.25
    float widthRatio = (float)protoW / (float)kInputW;  // 160 / 640 = 0.25
    downsample_bbox(bboxDevice, n * 4, heightRatio, widthRatio, stream);

    // set 0 where mask out of bbox
    crop_mask(maskDevice, n, protoH, protoW, bboxDevice, stream);

    // scale mask from 160x160 to original resolution
    // 1. cut mask
    float r_w = protoW / (img.cols * 1.0);
    float r_h = protoH / (img.rows * 1.0);
    float r = std::min(r_w, r_h);
    float pad_h = (protoH - r * img.rows) / 2;
    float pad_w = (protoW - r * img.cols) / 2;
    int cutMaskLeft = (int)pad_w;
    int cutMaskTop = (int)pad_h;
    int cutMaskRight = (int)(protoW - pad_w);
    int cutMaskBottom = (int)(protoH - pad_h);
    int cutMaskWidth = cutMaskRight - cutMaskLeft;
    int cutMaskHeight = cutMaskBottom - cutMaskTop;
    float* cutMaskDevice = nullptr;
    CHECK(cudaMalloc(&cutMaskDevice, n * cutMaskHeight * cutMaskWidth * sizeof(float)));
    cut_mask(maskDevice, n, protoH, protoW, cutMaskDevice, cutMaskTop, cutMaskLeft, cutMaskHeight, cutMaskWidth, stream);

    // 2. bilinear resize mask
    float* scaledMaskDevice = nullptr;
    CHECK(cudaMalloc(&scaledMaskDevice, n * img.rows * img.cols * sizeof(float)));
    resize(cutMaskDevice, n, cutMaskHeight, cutMaskWidth, scaledMaskDevice, img.rows, img.cols, stream);

    for (size_t i = 0; i < n; i++){
        vDetections[i].maskMatrix.resize(img.rows * img.cols);
        CHECK(cudaMemcpyAsync(vDetections[i].maskMatrix.data(), &scaledMaskDevice[i * img.rows * img.cols], img.rows * img.cols * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }

    CHECK(cudaFree(maskCoefDevice));
    CHECK(cudaFree(maskDevice));
    CHECK(cudaFree(bboxDevice));
    CHECK(cudaFree(cutMaskDevice));
    CHECK(cudaFree(scaledMaskDevice));
}


void YoloDetector::draw_image(cv::Mat& img, std::vector<Detection>& inferResult, bool drawBbox){
    // draw inference result on image
    for (size_t i = 0; i < inferResult.size(); i++){
        // draw bboxes
        if (drawBbox){
            cv::Scalar bboxColor(get_random_int(), get_random_int(), get_random_int());
            cv::Rect r(
                round(inferResult[i].bbox[0]),
                round(inferResult[i].bbox[1]),
                round(inferResult[i].bbox[2] - inferResult[i].bbox[0]),
                round(inferResult[i].bbox[3] - inferResult[i].bbox[1])
            );
            cv::rectangle(img, r, bboxColor, 2);

            std::string className = vClassNames[(int)inferResult[i].classId];
            std::string labelStr = className + " " + std::to_string(inferResult[i].conf).substr(0, 4);

            cv::Size textSize = cv::getTextSize(labelStr, cv::FONT_HERSHEY_PLAIN, 1.2, 2, NULL);
            cv::Point topLeft(r.x, r.y - textSize.height - 3);
            cv::Point bottomRight(r.x + textSize.width, r.y);
            cv::rectangle(img, topLeft, bottomRight, bboxColor, -1);
            cv::putText(img, labelStr, cv::Point(r.x, r.y - 2), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }

        draw_mask(img, inferResult[i].maskMatrix.data());
    }
}
