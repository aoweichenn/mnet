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
  `MultiHeadAttention`（fp32、自注意力/交叉注意力、可选 mask）、`Proposal`、
  `PSROIPooling`（Faster-RCNN / RFCN 的 ROI 生成与池化）、`ELU`、`SELU`、`Erf`、
  `Mish`、`Softplus`、`PixelShuffle`、`Reorg`、`Convolution1D`、`Convolution3D`。
- `tests/unit/model_tests.cpp` 已端到端跑通：
  - `squeezenet_v1.1`（227x227x3 → 1000 类 softmax，概率和校验）
  - `shufflenet_v2_x0.5`（224x224x3 → 1000 维输出，覆盖 ConvDW/BN/Slice/ShuffleChannel）
  - `yolo11n`（640x640x3 → `out0`，覆盖 Reshape/Permute/Slice/BinaryOp/MatMul/Interp；
    Release+AVX2 约 47 ms，峰值 RSS 98 MB；3x3 stride1 按尺寸选 Winograd
    F(2x2,3x3) 或 F(4x4,3x3)）
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

所有已发布 `.bin` 的 param 现在都能找到对应 layer 实现；`rfcn_end2end` 与
`yolov4-opt` 的权重在上游以分卷 `.partNN` 发布，拼接后可端到端运行。
`mnet_probe --allow-empty` 用于随机输入下检测头输出为空的模型。

数值一致性：`mcnn/tools/parity` 用同一份 param/bin 同时跑上游 ncnn 与 mnet，
已覆盖 14 类算子：3x3 Winograd 卷积、1x1 卷积、ConvolutionDepthWise、Gemm
（常量转置权重）、Interp（nearest）、SDPA（多输入注意力）、Pooling、Softmax、
Concat、BatchNorm、Permute、Slice、MatMul、Reshape，最大绝对误差均 ≤ 9.5e-7。
另提供 `tools/parity/model_compare`：同一模型文件与输入同时跑 ncnn 与 mnet，
可比较任意 blob 或多个输出。26 个模型端到端对齐（max_abs_diff / 相对误差）：
yolo11n 1.2e-4、yolov8n 1.2e-4、PP-OCRv5 rec 1.3e-4（含 MHA）、yolact 1.3e-5、
nanodet_m 3.3e-6、worldv2 8.5e-3（相对 4e-5）、squeezenet 4.7e-7。
该工具先后定位并修复了 Interp 双线性边界越界外插和 Reduction 4D→3D 形状映射
两个 bug。

尚未完成：

- SSD 系列在随机输入下无检测框（`DetectionOutput` 返回空，行为与 ncnn 一致）；
- yolo11n 约 1.45 倍慢于 ncnn（46–47 ms vs 32 ms），热点仍是 3x3 Convolution
  （Winograd 后约 31 ms）与 1x1 Convolution（约 12 ms）；
  `MultiHeadAttention` 的 KV-cache 与量化路径尚未移植（遇到时返回 -1）；
- 与 ncnn 参考输出的逐层数值对比测试；
- 安装包导出已支持（`cmake --install` + `find_package(mnet)` 消费测试通过）。
