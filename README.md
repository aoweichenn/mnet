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

## 当前状态

- `Net::load_param` 使用 `mgraph::Graph` 解析文本 param。
- `Net::load_model` 使用 `mformat::ModelBinFromDataReader` 顺序读取权重。
- `Extractor` 通过 `mgraph::Extractor` 懒执行 producer，再复制输出 blob。
- 已注册层：`Input`、`ReLU`、`Sigmoid`、`TanH`、`Swish`、`Flatten`、`Dropout`、
  `Threshold`、`Power`、`GELU`、`Softmax`、`Bias`、`Scale`、`Convolution`、
  `Pooling`、`InnerProduct`、`Concat`、`Split`、`ConvolutionDepthWise`、
  `BatchNorm`、`ShuffleChannel`、`Slice`、`RMSNorm`、`BinaryOp`、`Permute`、
  `Reshape`、`MatMul`、`Interp`、`Embed`、`Gemm`、`RotaryEmbed`、`SDPA`（KV cache）。
- `tests/unit/model_tests.cpp` 已端到端跑通：
  - `squeezenet_v1.1`（227x227x3 → 1000 类 softmax，概率和校验）
  - `shufflenet_v2_x0.5`（224x224x3 → 1000 维输出，覆盖 ConvDW/BN/Slice/ShuffleChannel）
  - `yolo11n`（640x640x3 → `out0`，覆盖 Reshape/Permute/Slice/BinaryOp/MatMul/Interp）
  - Qwen3 0.6B 单 token：embed → decoder → proj → argmax（覆盖 Gemm/RotaryEmbed/SDPA/KV cache）

尚未完成：

- 二进制 param（`load_param_bin`）与完整 layer type 表；
- 更多层（BinaryOp/Reshape/Permute/Slice/RMSNorm/SDPA 等）；
- 安装包导出（当前 `add_subdirectory` 消费兄弟模块）；
- 与 ncnn 参考输出的数值对比测试。
