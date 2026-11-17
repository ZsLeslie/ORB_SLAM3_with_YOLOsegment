# YOLO26 实例分割接入

默认模型为 `yolo26m-seg`。推理源码来自
`/home/leslie/code/TensorRT-YOLO11/C++/segment`，已放入 `src/yolo`，
由当前项目编译 `lib/libyolo_infer.so`，避免预编译库与头文件不匹配。

## 模型文件

从项目根目录运行程序，模型路径统一配置在 `include/yolo/config.h`：

- `kYoloEngineFile`：`./model/yolo26m-seg.engine`。
- `onnxFile`：`./model/yolo26m-seg.onnx`，引擎不存在时用于构建。

本机已从参考目录复制这两份模型。模型与校准缓存不纳入 Git；在其他机器上
需要单独准备模型，并使用目标 GPU 和 TensorRT 环境重新构建引擎。
更换 ONNX 后应移走对应旧引擎，程序优先加载现有引擎。

当前支持 TensorRT 8.x API，本机使用 TensorRT 8.6.1.6。输入输出必须为
FP32、线性内存布局；引擎内部可以使用 FP16，张量类型以实际引擎为准。

| 张量 | 本次模型形状 | 内容 |
| --- | --- | --- |
| `images` | `[1, 3, 640, 640]` | RGB、CHW、除以 255，Letterbox 填充值 114 |
| `output0` | `[1, 116, 8400]` | xywh + 80 类分数 + 32 个掩膜系数 |
| `output1` | `[1, 32, 160, 160]` | 掩膜原型 |

推理按名称查找张量绑定，校验输入尺寸、输出布局和数据类型，并在 CPU
读取检测结果前同步 CUDA 流。后处理继续使用转置、置信度过滤、NMS 和
掩膜重建，向 Tracking 返回原图坐标框及原图尺寸的概率掩膜。掩膜缩放时
钳制采样坐标，避免图像边缘外插产生超出 0–1 的概率值。

当前 ONNX 的元数据为 Ultralytics 8.4.211、Opset 16、`end2end=False`、
`nms=None`。该版本可使用下面的方式重新导出原始输出模型：

```python
from ultralytics import YOLO

YOLO("yolo26m-seg.pt").export(
    format="onnx", imgsz=640, batch=1, dynamic=False,
    opset=16, simplify=False, nms=None,
)
```

不同 Ultralytics 版本的导出选项可能不同，应核对实际张量形状和数据类型。
`[1, 300, 38]` 等端到端或内置 NMS 输出会被明确拒绝，需要重新导出为上表
中的原始输出布局。相同布局的 YOLO11 分割模型仍可使用。

## 编译

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DTENSORRT_ROOT=/home/leslie/TensorRT-8.6.1.6
cmake --build build -j2
```

编译需要 CMake 3.17 或更新版本，以及原项目的 CUDA、OpenCV 等依赖。
必须一并重编译推理库与 SLAM 程序；不能搭配旧的 `YoloDetector` 头文件使用
新的动态库。Tracking 的人体运动判断、动态特征剔除和语义点云流程保持原样。
