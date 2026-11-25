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

- `Net::load_param` 使用 `mgraph::Graph` 解析文本 param。
- `Net::load_model` 使用 `mformat::ModelBinFromDataReader` 顺序读取权重。
- `Extractor` 通过 `mgraph::Extractor` 懒执行 producer，再复制输出 blob。
- 已注册层：`Input`、`ReLU`、`Sigmoid`、`TanH`、`Swish`、`Flatten`、`Dropout`、
  `Threshold`、`Power`、`GELU`、`Softmax`、`Bias`、`Scale`、`Convolution`、
  `Pooling`、`InnerProduct`、`Concat`、`Split`、`ConvolutionDepthWise`、
  `BatchNorm`、`ShuffleChannel`、`Slice`、`RMSNorm`、`BinaryOp`、`Permute`、
  `Reshape`、`MatMul`、`Interp`、`Embed`、`Gemm`、`RotaryEmbed`、`SDPA`（KV cache）、
  `AbsVal`、`BNLL`、`Clip`、`Exp`、`Log`、`HardSigmoid`、`HardSwish`、`GroupNorm`、
  `LayerNorm`、`PReLU`、`Normalize`、`Deconvolution`、`YoloV5Focus`。
- `tests/unit/model_tests.cpp` 已端到端跑通：
  - `squeezenet_v1.1`（227x227x3 → 1000 类 softmax，概率和校验）
  - `shufflenet_v2_x0.5`（224x224x3 → 1000 维输出，覆盖 ConvDW/BN/Slice/ShuffleChannel）
  - `yolo11n`（640x640x3 → `out0`，覆盖 Reshape/Permute/Slice/BinaryOp/MatMul/Interp；
    Release+AVX2 约 84 ms，峰值 RSS 96 MB）
  - 另外通过 `mnet_probe` 验证 16 个模型：pose、scrfd、yolo11n_seg/pose/obb、
    yolov8n/seg/pose/obb/cls、yolov5s/5s.ncnn/5s_6.0、yolov6n、yolov7-tiny、nanodet_m
  - Qwen3 0.6B int8 单 token：embed → decoder → proj → argmax
  - Qwen3 0.6B int8 两 token decode：把 28 层 KV cache 回灌，验证 cache 复用
  - 文本生成：BPE tokenizer + 23-token chat prompt → 3 个新 token → `Hello! How can`

尚未完成：

- 二进制 param 已支持（按 ncnn type index 注册）；仍缺完整 layer type 表；
- 更广的层覆盖与二进制 param；
- 安装包导出已支持（`cmake --install` + `find_package(mnet)` 消费测试通过）；
- 与 ncnn 参考输出的数值对比测试。
