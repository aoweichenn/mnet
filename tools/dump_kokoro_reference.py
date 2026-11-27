"""Dump a deterministic Kokoro (StyleTTS2) reference with the stage tensors
tools/mnet_tts validates against.

The model takes (input_ids, style, speed); no random sources are present, so the
same inputs always give the same waveform.
"""
import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ort.set_default_logger_severity(3)

TARGETS = [
    r"^/encoder/bert/embeddings/LayerNorm/LayerNormalization$",
    r"^/encoder/bert/encoder/albert_layer_groups\.0/albert_layers\.0/attention/LayerNorm(_\d+)?/LayerNormalization$",
    r"^/encoder/bert/encoder/albert_layer_groups\.0/albert_layers\.0/full_layer_layer_norm(_\d+)?/LayerNormalization$",
    r"^/encoder/bert/encoder/albert_layer_groups\.0/albert_layers\.0/ffn/Add(_\d+)?$",
    r"^/encoder/bert/encoder/albert_layer_groups\.0/albert_layers\.0/ffn_output/MatMul(_\d+)?$",
    r"^/encoder/bert_encoder.*$",
    r"^/encoder/text_encoder/embedding/Gather$",
    r"^/encoder/text_encoder/cnn\.\d/cnn\.\d\.0/Conv$",
    r"^/encoder/text_encoder/cnn\.\d/cnn\.\d\.1/LayerNormalization$",
    r"^/encoder/text_encoder/lstm/LSTM$",
    r"^/encoder/predictor/text_encoder/lstms\.\d/LSTM$",
    r"^/encoder/predictor/text_encoder/lstms\.\d/Add_2$",
    r"^/encoder/predictor/lstm/LSTM$",
    r"^/encoder/predictor/duration_proj/linear_layer/Add$",
    r"^/encoder/predictor/Sigmoid$",
    r"^/encoder/Round$",
    r"^/decoder/decoder/Concat$",
    r"^/decoder/decoder/(encode|decode\.\d)/(conv1|conv2)/Conv$",
    r"^/decoder/decoder/generator/conv_pre/Conv$",
    r"^/decoder/decoder/generator/ups\.\d/ConvTranspose$",
    r"^/decoder/decoder/generator/conv_post/Conv$",
    r"^/decoder/decoder/generator/resblocks\.0/(convs1|convs2)\.\d/Conv$",
    r"^/decoder/decoder/generator/resblocks\.0/(adain1|adain2)\.\d/Mul_3$",
    r"^/decoder/decoder/generator/resblocks\.0/(Mul|Mul_1)$",
    r"^/decoder/decoder/generator/resblocks\.0/adain1\.0/(fc/Gemm|ReduceMean|Div_1|Add_1|Slice_1|Add_3)$",
    r"^/decoder/decoder/generator/resblocks\.0/Add$",
    r"^/decoder/decoder/generator/resblocks\.[0-5]/Add_8$",
    r"^/decoder/decoder/generator/noise_res\.[01]/Add_8$",
    r"^/decoder/decoder/generator/noise_res\.0/(adain1|adain2)\.\d/Add_3$",
    r"^/decoder/decoder/generator/noise_res\.0/(Add(_\d+)?|convs1\.\d/Conv|convs2\.\d/Conv)$",
    r"^/decoder/decoder/generator/noise_convs\.[01]/Conv$",
    r"^/decoder/decoder/generator/Add_[2-7]$",
    r"^/decoder/decoder/generator/Div_[24]$",
    r"^/decoder/decoder/generator/LeakyRelu(_1|_2)?$",
    r"^/decoder/decoder/generator/reflection_pad/Pad$",
    r"^/decoder/decoder/generator/Concat_3$",
    r"^/decoder/decoder/generator/f0_upsamp/Resize$",
    r"^/decoder/decoder/generator/m_source/l_sin_gen/(Mul_6|Sin)$",
    r"^/decoder/decoder/generator/m_source/l_sin_gen/(Mul_4|Resize_1|CumSum)$",
    r"^/decoder/decoder/generator/m_source/l_tanh/Tanh$",
    r"^/decoder/decoder/generator/istft/stft/(ConvTranspose|Mul_4|Slice_4|ScatterND)$",
    r"^/decoder/decoder/(encode|decode\.\d)/Div_\d+$",
    r"^/decoder/decoder/F0_conv/Conv$",
    r"^/decoder/decoder/N_conv/Conv$",
    r"^/encoder/F0\.\d/(Div_2|Div_4|Div_6)$",
    r"^/encoder/N\.\d/(Div_2|Div_4|Div_6)$",
    r"^/encoder/shared/LSTM$",
    r"^/encoder/F0_proj/Conv$",
    r"^/encoder/N_proj/Conv$",
    r"^/encoder/F0\.\d/conv1/Conv$",
    r"^/encoder/F0\.\d/conv2/Conv$",
    r"^/encoder/F0\.\d/norm1/Mul_3$",
    r"^/encoder/F0\.\d/norm2/Mul_3$",
    r"^/encoder/bert/encoder/embedding_hidden_mapping_in/Add$",
]

parser = argparse.ArgumentParser(description="Dump a deterministic Kokoro reference")
parser.add_argument("model")
parser.add_argument("--ids", required=True, help="comma separated phoneme ids, e.g. 43,60,66")
parser.add_argument("--style", required=True, help=".bin with the 256-value style row")
parser.add_argument("--speed", default="1.0")
parser.add_argument("--out", default=None)
args = parser.parse_args()

model = onnx.load(args.model)
graph = model.graph
inf = onnx.shape_inference.infer_shapes(model)
shape_of = {}
for vi in list(inf.graph.value_info) + list(inf.graph.output) + list(inf.graph.input):
    if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
        shape_of[vi.name] = [d.dim_value if d.HasField("dim_value") else -1 for d in vi.type.tensor_type.shape.dim]

wanted = []
for node in graph.node:
    for pattern in TARGETS:
        import re

        if re.match(pattern, node.name):
            wanted.append((re.sub(r"[^A-Za-z0-9.]+", "_", node.name).strip("_"), node.output[0], node.op_type))
            break
wanted.append(("output", graph.output[0].name, "output"))

produced = {o.name for o in graph.output}
seen = set()
unique = []
for label, tensor, op in wanted:
    if tensor in seen:
        continue
    seen.add(tensor)
    unique.append((label, tensor))
    if tensor not in produced:
        graph.output.append(helper.make_tensor_value_info(tensor, TensorProto.FLOAT, shape_of.get(tensor)))

out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.model)), "ref")
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "kokoro_ref.onnx")
onnx.save(model, path)

ids = np.array([[int(v) for v in args.ids.split(",")]], dtype=np.int64)
style = np.fromfile(args.style, dtype=np.float32).reshape(1, 256)
speed = np.array([float(args.speed)], dtype=np.float32)
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
names = [o.name for o in sess.get_outputs()]
values = sess.run(None, {"input_ids": ids, "style": style, "speed": speed})
got = dict(zip(names, values))

with open(os.path.join(out_dir, "manifest.txt"), "w") as f:
    f.write(f"ids {ids.size} {'x'.join(str(v) for v in ids[0])}\n")
    for label, tensor in unique:
        arr = np.asarray(got[tensor], dtype=np.float32)
        f.write(f"{label} {arr.size} " + "x".join(str(s) for s in arr.shape) + "\n")
        arr.ravel().tofile(os.path.join(out_dir, label + ".bin"))
print(f"dumped {len(unique)} tensors to {out_dir}")
for label, tensor in unique[:6]:
    print("  ", label, np.asarray(got[tensor]).shape)
