"""Flatten every parameter of an ONNX TTS model into one blob plus a text manifest.

torch exports any weight-normalised convolution as a folded initializer with a
generic name (`onnx::Conv_8168`), so those are re-registered under the name of
the convolution they feed (`flow.flows.6.enc.in_layers.0.weight`). Constants the
exporter folds into standalone tensors (e.g. a frozen exp(-logs)) keep their own
name and are exported too.
"""
import argparse
import os

import numpy as np
import onnx

parser = argparse.ArgumentParser(description="Flatten an ONNX TTS model into mnet_tts's weight blob")
parser.add_argument("model", help="path to the .onnx model")
parser.add_argument("--blob", default="piper_weights", help="blob/manifest basename")
parser.add_argument("--out-dir", default=None, help="output directory (default: the model directory)")
args = parser.parse_args()
DIR = args.out_dir or os.path.dirname(os.path.abspath(args.model))
SRC = args.model

model = onnx.load(SRC)
graph = model.graph
init_names = {init.name for init in graph.initializer}

# generic folded weights -> the owning convolution's canonical parameter name
renames = {}
for node in graph.node:
    if node.op_type in ("Conv", "ConvTranspose"):
        slots = ((1, ".weight"), (2, ".bias"))
    elif node.op_type == "MatMul":
        # MatMul(x, W): the weight is the second operand
        slots = ((1, ".weight"),)
    elif node.op_type == "Gemm":
        slots = ((1, ".weight"), (2, ".bias"))
    elif node.op_type == "LSTM":
        # ONNX LSTM inputs are (x, W, R, B, sequence_lens, initial_h, initial_c)
        slots = ((1, ".weight_ih"), (2, ".weight_hh"), (3, ".bias"))
    else:
        continue
    base = node.name.strip("/").replace("/", ".")
    # the trailing path component is the op name itself
    for suffix in (".ConvTranspose", ".Conv", ".MatMul", ".Gemm", ".LSTM"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    for index, suffix in slots:
        if index >= len(node.input):
            continue
        name = node.input[index]
        if name in init_names and (name.startswith("onnx::") or name.startswith("/")):
            renames.setdefault(name, set()).add(base + suffix)

entries = []
aliases = []
blob = []
offset = 0
seen = set()
for init in graph.initializer:
    arr = np.ascontiguousarray(onnx.numpy_helper.to_array(init).astype(np.float32))
    names = sorted(n for n in renames.get(init.name, {init.name}) if n not in seen)
    if not names:
        continue
    primary = names[0]
    seen.add(primary)
    entries.append((primary, arr.size, tuple(arr.shape), offset))
    blob.append(arr.ravel())
    for alias in names[1:]:
        if alias not in seen:
            seen.add(alias)
            aliases.append((alias, arr.size, tuple(arr.shape), offset))
    offset += arr.size

flat = np.concatenate(blob) if blob else np.zeros(0, dtype=np.float32)
flat.tofile(os.path.join(DIR, args.blob + ".bin"))
with open(os.path.join(DIR, args.blob + ".txt"), "w") as f:
    f.write(f"{len(entries)}\n")
    for name, count, shape, off in entries + aliases:
        f.write(f"{name} {count} {off} {'x'.join(str(s) for s in shape)}\n")
print(f"wrote {len(entries)} params, {flat.size} values ({flat.nbytes / 1e6:.1f} MB)")
