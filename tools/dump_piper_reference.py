"""Dump a deterministic Piper/VITS reference (zero noise) with the stage tensors
that a pure-mnet implementation is validated against."""
import os
import re

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ort.set_default_logger_severity(3)

import argparse

args_parser = argparse.ArgumentParser(description="Dump a deterministic Piper/VITS reference with stage tensors")
args_parser.add_argument("model", help="path to the .onnx model")
args_parser.add_argument("--ids", required=True, help="phoneme ids (float32 or int64 binary)")
args_parser.add_argument("--scales", required=True, help="three float32 scales (noise, length, noise_w)")
args_parser.add_argument("--out", default=None, help="reference directory (default: <model dir>/ref)")
ARGS = args_parser.parse_args()
SRC = ARGS.model
DIR = os.path.dirname(os.path.abspath(SRC))
OUT = ARGS.out or os.path.join(DIR, "ref")

TARGETS = [
    r"^/enc_p/emb/Gather$",
    r"^/enc_p/proj/Conv$",
    r"^/enc_p/encoder/attn_layers\.\d+/conv_[qkvo]/Conv$",
    r"^/enc_p/encoder/ffn_layers\.\d+/conv_[12]/Conv$",
    r"^/enc_p/encoder/norm_layers_[12]\.\d+/Mul$",
    r"^/dp/Split$",
    r"^/Exp$",
    r"^/dp/pre/Conv$",
    r"^/dp/proj/Conv$",
    r"^/dp/convs/convs_1x1\.\d+/Conv$",
    r"^/dp/convs/convs_sep\.\d+/Conv$",
    r"^/dp/flows\.\d+/pre/Conv$",
    r"^/dp/flows\.\d+/proj/Conv$",
    r"^/flow/flows\.\d+/pre/Conv$",
    r"^/flow/flows\.\d+/post/Conv$",
    r"^/flow/flows\.\d+/enc/in_layers\.\d+/Conv$",
    r"^/dec/conv_pre/Conv$",
    r"^/dec/ups\.\d+/ConvTranspose$",
    r"^/dec/resblocks\.\d+/convs\.\d+/Conv$",
    r"^/dec/conv_post/Conv$",
    r"^/dec/Tanh$",
]


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9.]+", "_", name).strip("_")


def main():
    model = onnx.load(SRC)
    graph = model.graph

    # deterministic noise: RandomNormalLike(x) -> Mul(x, 0)
    nodes = []
    extra = []
    for node in graph.node:
        if node.op_type == "RandomNormalLike":
            zero = numpy_helper.from_array(np.zeros(1, dtype=np.float32), "zero_" + sanitize(node.name))
            extra.append(zero)
            nodes.append(helper.make_node("Mul", [node.input[0], zero.name], [node.output[0]], name=node.name))
        else:
            nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    graph.initializer.extend(extra)

    inf = onnx.shape_inference.infer_shapes(model)
    shape_of = {}
    for vi in list(inf.graph.value_info) + list(inf.graph.output) + list(inf.graph.input):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            dims = [d.dim_value if d.HasField("dim_value") else -1 for d in vi.type.tensor_type.shape.dim]
            shape_of[vi.name] = dims

    wanted = []
    for node in graph.node:
        for pat in TARGETS:
            if re.match(pat, node.name):
                wanted.append((sanitize(node.name), node.output[0]))
                break
    wanted.append(("output", graph.output[0].name))
    # de-duplicate by tensor name, keep the first label
    seen = set()
    unique = []
    for label, tensor in wanted:
        if tensor in seen:
            continue
        seen.add(tensor)
        unique.append((label, tensor))

    produced = {o.name for o in graph.output}
    for label, tensor in unique:
        if tensor in produced:
            continue
        dims = shape_of.get(tensor)
        graph.output.append(helper.make_tensor_value_info(tensor, TensorProto.FLOAT, dims))

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "piper_zero_ref.onnx")
    onnx.save(model, path)

    # phoneme ids come either as float32 or as int64 tensors
    as_float = np.fromfile(ARGS.ids, dtype=np.float32)
    if as_float.size > 0 and np.all(np.isfinite(as_float)) and np.all(as_float == np.round(as_float)) and \
            as_float.min() >= 0 and as_float.max() < 65536:
        ids = as_float.astype(np.int64)[None, :]
    else:
        ids = np.fromfile(ARGS.ids, dtype=np.int64)[None, :]
    lengths = np.array([ids.shape[1]], dtype=np.int64)
    scales = np.fromfile(ARGS.scales, dtype=np.float32)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    values = sess.run(None, {"input": ids, "input_lengths": lengths, "scales": scales})
    got = dict(zip(out_names, values))

    with open(os.path.join(OUT, "manifest.txt"), "w") as f:
        f.write(f"ids {ids.size}\n")
        f.write(f"length {int(lengths[0])}\n")
        for label, tensor in unique:
            arr = np.asarray(got[tensor], dtype=np.float32).ravel()
            shape = list(np.asarray(got[tensor]).shape)
            f.write(f"{label} {arr.size} {'x'.join(str(s) for s in shape)}\n")
            arr.tofile(os.path.join(OUT, label + ".bin"))
    print(f"dumped {len(unique)} tensors to {OUT}")
    for label, tensor in unique[:8]:
        print("  ", label, np.asarray(got[tensor]).shape)
    print("  ...")


if __name__ == "__main__":
    main()
