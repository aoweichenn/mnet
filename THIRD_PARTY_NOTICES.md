# Third-party notices

## BPE tokenizer

`src/tokenizer/bpe_tokenizer.cpp` and `include/mnet/detail/bpe_tokenizer.h` /
`include/mnet/detail/tokenizer_types.h` are derived from the ncnn_llm project
(https://github.com/futz12/ncnn_llm), which is distributed under the Apache
License 2.0. The files are used for byte-level BPE tokenization in the optional
LLM runner.

Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

## GGML quantized formats

`include/mnet/detail/iq1s_grid.hpp`, `include/mnet/detail/iq_grids.hpp` and the
IQ1_S/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/Q4_K/Q5_K/Q6_K/IQ4_XS dequantization
routines in `include/mnet/gguf.hpp` are derived from llama.cpp
(https://github.com/ggml-org/llama.cpp), which is distributed under the MIT
License. The codebook table and the block layouts follow
`ggml/src/ggml-common.h` and `ggml/src/ggml-quants.c`.

MIT License: https://opensource.org/licenses/MIT

## Unicode tables

`include/mnet/detail/unicode_flags.hpp` (codepoint category and whitespace
ranges) is derived from llama.cpp (MIT License), `src/unicode-data.cpp`.

The BPE pre-tokenizer structure in `include/mnet/bpe.hpp` follows llama.cpp's
`unicode_regex_split_custom_qwen35` and the GPT4O pattern.
