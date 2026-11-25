# mnet

mcnn 的独立 CPU 推理 runtime。只依赖 mcnn 自己的模块，不包含、不链接 ncnn：

- `mformat`：param/bin 读取、`ParamDict`、`ModelBin`
- `mgraph`：拓扑解析、`Layer` 基类、`ExecutionBackend`、`Extractor`
- `mmat` / `mmemory` / `mplatform`：`Mat` 存储与分配
- `mlayer`：concrete layer 实现

## 用法

```cpp
#include <mnet/net.hpp>

mnet::Net net;
net.opt().num_threads = 4;
net.load("model.param", "model.bin");

mnet::Mat in(224, 224, 3);
// ... fill input ...

mnet::Net::Extractor ex = net.create_extractor();
ex.input("data", in);

mnet::Mat out;
ex.extract("output", out);
```

构建选项：

- `-DCMAKE_BUILD_TYPE=Release`：Benchmark 必须用 Release；Debug 会慢 3–6 倍。
- `-Dmnet_enable_avx2=ON`：为 mnet 对象和测试打开 AVX2/FMA 快路径。

## 平台支持

CPU 路径已在四个平台构建并跑通（数值 smoke：`Scale` 2x+1、1x1 卷积 probe）：

| 平台 | 工具链 | OpenMP 后端 | 验证 |
| --- | --- | --- | --- |
| Linux x86_64（含 WSL2） | GCC/Clang + 系统 OpenMP | 系统 libgomp/libomp | `ctest` 6/6；AVX2 |
| macOS arm64 | CommandLineTools clang + CLT SDK | `mplatform::simpleomp` | 构建 + smoke + probe |
| Windows x86_64 | llvm-mingw clang（交叉编译，`-static`） | `mplatform::simpleomp` | 构建 + smoke + probe（Windows 原生运行） |
| Android arm64 | NDK r27c clang（API 26，`c++_static`） | `mplatform::simpleomp` | 构建 + smoke + probe（真机运行） |

无系统 OpenMP 的平台使用：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -Dmnet_use_simple_omp=ON
```

该选项会把 `mplatform/src/omp/simpleomp.cpp` 编入 `libmnet.a`，并由 mplatform
生成一个 `<omp.h>` 兼容头；Apple clang 自动加 `-Xpreprocessor -fopenmp`。
Android 使用 `-DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26
-DANDROID_STL=c++_static`；Windows 交叉编译需要 Windows 版 CMake 运行测试或把
产物拷到 Windows 执行。

## 当前状态

- `Net::load_param` 按 magic（`7767517`）自动识别文本/二进制 param，分别走
  `mgraph::Graph::load_param` / `load_param_bin`；二进制按 ncnn type index 建层，
  已覆盖全部有稳定 ncnn type index 的已注册层（70 个），仅有
  `YoloV5Focus` / `ShortConv` / `GatedDeltaRule` 因上游枚举未包含而只支持文本名。
- `Net::load_model` 使用 `mformat::ModelBinFromDataReader` 顺序读取权重。
- `Extractor` 通过 `mgraph::Extractor` 懒执行 producer，再复制输出 blob。
- 已注册层：`Input`、`ReLU`、`Sigmoid`、`TanH`、`Swish`、`Flatten`、`Dropout`、
  `Threshold`、`Power`、`GELU`、`Softmax`、`Bias`、`Scale`、`Convolution`、
  `Pooling`、`InnerProduct`、`Concat`、`Split`、`ConvolutionDepthWise`、
  `BatchNorm`、`ShuffleChannel`、`Slice`、`RMSNorm`、`BinaryOp`、`Permute`、
  `Reshape`、`MatMul`、`Interp`、`Embed`、`Gemm`、`RotaryEmbed`、`SDPA`（KV cache）、
  `AbsVal`、`BNLL`、`Clip`、`Exp`、`Log`、`HardSigmoid`、`HardSwish`、`GroupNorm`、
  `LayerNorm`、`PReLU`、`Normalize`、`Deconvolution`、`DeconvolutionDepthWise`、
  `YoloV5Focus`、`MemoryData`、`PriorBox`、`DetectionOutput`、`Eltwise`、`Noop`、
  `UnaryOp`、`ExpandDims`、`Reduction`、`Einsum`、`Crop`、`Squeeze`、
  `YoloDetectionOutput`、`Yolov3DetectionOutput`、`ShortConv`、`GatedDeltaRule`、
  `MultiHeadAttention`（fp32 与量化：weight-block int4/int6/int8、legacy int8；
  自注意力/交叉注意力、可选 mask、KV cache）、`Proposal`、
  `ROIPooling` / `PSROIPooling` / `ROIAlign`（Faster-RCNN / R-CNN 的 ROI 池化）、
  `Proposal`、`Padding`、`Tile`、`GridSample`（2D 双线性/最近邻）、`ELU`、`SELU`、
  `Erf`、`Mish`、`Softplus`、`PixelShuffle`、`Reorg`、`Convolution1D`、`Convolution3D`、
  `LSTM` / `GRU`（单向与双向，LSTM 含 hidden→output 投影）。
  `Convolution` / `ConvolutionDepthWise` / `InnerProduct` 支持 ncnn 的 int8 推理
  （`8=int8_scale_term` 1/2/101/102，含 requantize 链）。
- `tests/unit/model_tests.cpp` 已端到端跑通：
  - `squeezenet_v1.1`（227x227x3 → 1000 类 softmax，概率和校验）
  - `shufflenet_v2_x0.5`（224x224x3 → 1000 维输出，覆盖 ConvDW/BN/Slice/ShuffleChannel）
  - `yolo11n`（640x640x3 → `out0`，覆盖 Reshape/Permute/Slice/BinaryOp/MatMul/Interp；
    Release+AVX2 4 线程 29.5–30.3 ms，峰值 RSS 73 MB；1x1 与 3x3 卷积走
    8 像素×4 输出通道的 pack8 im2col-GEMM 微内核）
  - 另外通过 `mnet_probe` 验证 35 个真实模型（AVX2 Release，含峰值 RSS），覆盖
    pose、scrfd、yolo11n_seg/pose/obb、yolov8n/seg/pose/obb/cls、yolov5s/5s.ncnn/
    5s_6.0、yolov6n、yolov7-tiny、nanodet_m、yolact、yolov8s_worldv2、pelee、
    mnet.25、mobilenet_ssd_voc/ssdlite、PP_OCRv5_mobile_det、yolov4-tiny、
    mobilenet_yolo、mobilenetv2_yolov3、PP_OCRv5_mobile_rec、rfcn_end2end、
    yolov4-opt 等。ncnn-assets 中所有带 `.bin` 的模型均已跑通。
    `ZF_faster_rcnn_final` 上游只有 param、没有权重，明确不在支持范围内。
  - Qwen3 0.6B int8 单 token：embed → decoder → proj → argmax
  - Qwen3 0.6B int8 两 token decode：把 28 层 KV cache 回灌，验证 cache 复用
  - 文本生成：BPE tokenizer + 23-token chat prompt → 3 个新 token → `Hello! How can`
  - `tools/mnet_llm`：读取 `model.json` 的通用 LLM runner（rope 参数 / tokenizer /
    KV、conv、gated-delta-rule 状态缓存），已跑通 MiniCPM5 1B/2B 与 Qwen3.5 4B int8：

| 模型 | 线程 | Prefill | Decode | 峰值 RSS |
| --- | --- | --- | --- | --- |
| MiniCPM5 1B int8 | 4 | 32.5 ms | 32.6 ms/tok | 3.03 GB |
| MiniCPM5 2B int8 | 4 | 156.1 ms | 95.6 ms/tok | 5.26 GB |
| Qwen3.5 4B int8 (int8 LM head) | 4 | 128.2 ms | 129.7 ms/tok | 7.55 GB |
| Qwen3 0.6B int8 | 4 | 126 ms | 18 ms/tok | 1.44 GB |

  同一台机器上 ncnn 基线（1B 4 线程 41.3 ms/tok / 3.21 GB，2B 4 线程
  103.6 ms/tok / 6.25 GB），mnet 的 decode 更快、峰值内存更低。

  LM head 量化流程已脚本化：`mcnn/tools/quantize/quantize_lm_head.sh` 调用
  ncnn 的 `ncnnllm2int` 把 `model.json` 里的 `proj_out` 权重压成 8-bit
  （block 64），其余文件以相对软链复用，并改写 `model.json` 指向新权重；
  对绑定词表的模型会保留共享的 embedding 权重文件。

所有已发布 `.bin` 的 param 现在都能找到对应 layer 实现；`rfcn_end2end` 与
`yolov4-opt` 的权重在上游以分卷 `.partNN` 发布，拼接后可端到端运行。
`mnet_probe --allow-empty` 用于随机输入下检测头输出为空的模型。

数值一致性：`mcnn/tools/parity` 用同一份 param/bin 同时跑上游 ncnn 与 mnet，
已覆盖 46 类算子：3x3 卷积、1x1 卷积、ConvolutionDepthWise、
Deconvolution、DeconvolutionDepthWise、Gemm（常量转置权重）、InnerProduct、
Interp、SDPA、Pooling、Softmax、Concat、BatchNorm、GroupNorm、LayerNorm、
LRN、Permute、Slice、Split、Crop、Squeeze、MatMul、Reshape、
DetectionOutput、Yolov3DetectionOutput、YoloDetectionOutput、PriorBox、
Eltwise、BinaryOp（张量与标量）、PReLU、Clip、ExpandDims、Reduction、
Einsum、UnaryOp、Swish、ReLU、Sigmoid、HardSigmoid、HardSwish、Dropout、
Noop、Flatten、ShuffleChannel、MemoryData、MultiHeadAttention（fp32 prefill/
KV-cache 解码 + int8/int4 权重量化 + legacy int8）、ROIPooling、
PSROIPooling、Proposal、ROIAlign、Padding、Tile、GridSample，以及 int8 卷积/
depthwise/InnerProduct/requantize、LSTM/GRU，共 86 组对比，最大绝对误差 ≤ 2e-5。
`MNET_NCNN_PARITY=1` 运行 46 类层对齐，`MNET_NCNN_MODELS=1` 运行 34 个模型
的端到端对齐（模型清单见 `mcnn/tools/parity/models.txt`，可用参数替换）。
另提供 `tools/parity/model_compare`：同一模型文件与输入同时跑 ncnn 与 mnet，
可比较任意 blob 或多个输出。34 个带权重模型端到端对齐，判定按每个通道取
`max(1e-3, 1e-3 * 通道幅值)`：除 worldv2 外最大绝对误差 ≤ 2.4e-4
（yolo11n 8.0e-5、yolov8n 7.4e-5、PP-OCRv5 rec 1.4e-4、yolact 1.4e-5、
nanodet_m 3.6e-6、squeezenet 4.5e-7）；worldv2 3.9e-2，但其通道幅值约 200，
相对误差 1.8e-4，与 ncnn 自身 Winograd/SGEMM 两种算法之间的差异
（同一 blob 上 max 1.26e-2）同一量级。
该工具先后定位并修复了 Interp 双线性边界越界外插和 Reduction 4D→3D 形状映射
两个 bug，并确认 `yolov4-tiny-opt` 只能在导出尺寸 416 下运行（固定 Crop）。

尚未完成：

- SSD 系列在随机输入下无检测框（`DetectionOutput` 返回空，行为与 ncnn 一致）；
- graph 级 elempack 传播（把数据在整个图里保持 pack8）尚未实现：当前只在卷积
  微内核内部打包，层与层之间仍是 1 通道布局；
- `GridSample` 目前只有 2D（双线性/最近邻），3D 与 bicubic 未实现；
- `LSTM` / `GRU` 的 int8 路径未移植（int8 模型会在 load_param 返回 -1）；
- 与 ncnn 参考输出的逐层数值对比测试；
- 安装包导出已支持（`cmake --install` + `find_package(mnet)` 消费测试通过）。
