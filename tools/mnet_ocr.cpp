// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// End-to-end PP-OCRv5 runner on top of mnet:
//
//   BMP image -> det (DB segmentation) -> connected components -> text boxes
//             -> crop/resize -> rec (CTC) -> UTF-8 text
//
// The postprocess mirrors ncnn/examples/ppocrv5.cpp (threshold + component
// score + rectangular unclip expansion) so latency and outputs are comparable.
// No OpenCV dependency: the tool reads uncompressed 24/32-bit BMP files.

#include <mnet/net.hpp>

#include <mplatform/cpu.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr std::int32_t kDetStride = 32;
constexpr std::int32_t kDetPadValue = 114;
constexpr std::int32_t kDetDefaultSize = 640;
constexpr float kDetThreshold = 0.3F;
constexpr float kBoxScoreThreshold = 0.6F;
constexpr float kBoxEnlargeWidth = 1.95F;
constexpr float kBoxEnlargeHeight = 0.95F;
constexpr float kMinBoxSizePx = 3.0F;
constexpr std::int32_t kMaxBoxes = 1000;
constexpr std::int32_t kRecHeight = 48;
constexpr std::int32_t kRecMaxWidth = 640;
constexpr std::int32_t kRecMinWidth = 8;

// PP-OCRv5 preprocessing (PaddleOCR NormalizeImage, RGB order).
constexpr float kDetMeanR = 0.485F * 255.0F;
constexpr float kDetMeanG = 0.456F * 255.0F;
constexpr float kDetMeanB = 0.406F * 255.0F;
constexpr float kDetNormR = 1.0F / 0.229F / 255.0F;
constexpr float kDetNormG = 1.0F / 0.224F / 255.0F;
constexpr float kDetNormB = 1.0F / 0.225F / 255.0F;

// PP-OCRv5 rec preprocessing: (x - 127.5) / 127.5.
constexpr float kRecMean = 127.5F;
constexpr float kRecNorm = 1.0F / 127.5F;

constexpr std::int32_t kBytesPerPixel = 3;

struct Image {
    std::int32_t w = 0;
    std::int32_t h = 0;
    std::vector<std::uint8_t> rgb;
};

struct Box {
    float x = 0.0F;
    float y = 0.0F;
    float w = 0.0F;
    float h = 0.0F;
    float score = 0.0F;
};

// Byte offset of pixel (x, y) in an interleaved RGB buffer of the given stride.
std::size_t rgb_offset(std::int32_t x, std::int32_t y, std::int32_t stride) {
    return (static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) + static_cast<std::size_t>(x)) *
           static_cast<std::size_t>(kBytesPerPixel);
}

std::uint32_t read_le32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

std::int32_t read_le32s(const std::uint8_t* p) {
    return static_cast<std::int32_t>(read_le32(p));
}

std::uint16_t read_le16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(static_cast<std::uint16_t>(p[0]) |
                                      (static_cast<std::uint16_t>(p[1]) << 8));
}

bool load_bmp(const std::string& path, Image& img) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return false;
    }

    std::uint8_t header[54] = {};
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    if (file.gcount() != static_cast<std::streamsize>(sizeof(header)) || header[0] != 'B' || header[1] != 'M') {
        std::fprintf(stderr, "%s: not a BMP file\n", path.c_str());
        return false;
    }

    const std::uint32_t data_offset = read_le32(header + 10);
    const std::int32_t width = read_le32s(header + 18);
    const std::int32_t signed_height = read_le32s(header + 22);
    const std::uint16_t bits = read_le16(header + 28);
    const std::uint32_t compression = read_le32(header + 30);
    if (compression != 0 || (bits != 24 && bits != 32) || width <= 0 || signed_height == 0) {
        std::fprintf(stderr, "%s: unsupported BMP (must be uncompressed 24/32-bit)\n", path.c_str());
        return false;
    }

    const bool bottom_up = signed_height > 0;
    const std::int32_t height = bottom_up ? signed_height : -signed_height;
    const std::int32_t bytes_per_pixel = bits / 8;
    const std::int32_t row_stride = ((width * bytes_per_pixel + 3) / 4) * 4;

    std::vector<std::uint8_t> row(static_cast<std::size_t>(row_stride));
    img.w = width;
    img.h = height;
    img.rgb.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * kBytesPerPixel);

    file.seekg(static_cast<std::streamoff>(data_offset), std::ios::beg);
    for (std::int32_t y = 0; y < height; ++y) {
        file.read(reinterpret_cast<char*>(row.data()), row_stride);
        if (file.gcount() != static_cast<std::streamsize>(row_stride)) {
            std::fprintf(stderr, "%s: truncated pixel data\n", path.c_str());
            return false;
        }
        const std::int32_t dst_y = bottom_up ? height - 1 - y : y;
        std::uint8_t* dst = img.rgb.data() + rgb_offset(0, dst_y, width);
        for (std::int32_t x = 0; x < width; ++x) {
            const std::uint8_t* src = row.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(bytes_per_pixel);
            const std::size_t base = static_cast<std::size_t>(x) * static_cast<std::size_t>(kBytesPerPixel);
            dst[base + 0] = src[2];  // BMP is BGR(A)
            dst[base + 1] = src[1];
            dst[base + 2] = src[0];
        }
    }
    return true;
}

// Bilinear resize of an interleaved RGB buffer.
void resize_rgb(const std::uint8_t* src, std::int32_t sw, std::int32_t sh, std::int32_t dw, std::int32_t dh,
                std::vector<std::uint8_t>& dst) {
    dst.resize(static_cast<std::size_t>(dw) * static_cast<std::size_t>(dh) * kBytesPerPixel);
    const float x_ratio = dw > 1 ? static_cast<float>(sw - 1) / static_cast<float>(dw - 1) : 0.0F;
    const float y_ratio = dh > 1 ? static_cast<float>(sh - 1) / static_cast<float>(dh - 1) : 0.0F;
    for (std::int32_t y = 0; y < dh; ++y) {
        const float fy = static_cast<float>(y) * y_ratio;
        const auto y0 = static_cast<std::int32_t>(fy);
        const std::int32_t y1 = std::min(y0 + 1, sh - 1);
        const float wy = fy - static_cast<float>(y0);
        for (std::int32_t x = 0; x < dw; ++x) {
            const float fx = static_cast<float>(x) * x_ratio;
            const auto x0 = static_cast<std::int32_t>(fx);
            const std::int32_t x1 = std::min(x0 + 1, sw - 1);
            const float wx = fx - static_cast<float>(x0);
            for (std::int32_t c = 0; c < kBytesPerPixel; ++c) {
                const float v00 = static_cast<float>(src[rgb_offset(x0, y0, sw) + static_cast<std::size_t>(c)]);
                const float v01 = static_cast<float>(src[rgb_offset(x1, y0, sw) + static_cast<std::size_t>(c)]);
                const float v10 = static_cast<float>(src[rgb_offset(x0, y1, sw) + static_cast<std::size_t>(c)]);
                const float v11 = static_cast<float>(src[rgb_offset(x1, y1, sw) + static_cast<std::size_t>(c)]);
                const float top = v00 + (v01 - v00) * wx;
                const float bottom = v10 + (v11 - v10) * wx;
                const float value = top + (bottom - top) * wy;
                dst[rgb_offset(x, y, dw) + static_cast<std::size_t>(c)] =
                    static_cast<std::uint8_t>(std::clamp(value + 0.5F, 0.0F, 255.0F));
            }
        }
    }
}

struct DetInput {
    mnet::Mat blob;
    float scale = 1.0F;
    std::int32_t resized_w = 0;
    std::int32_t resized_h = 0;
    std::int32_t pad_x = 0;
    std::int32_t pad_y = 0;
};

DetInput make_det_input(const Image& img, std::int32_t target_size) {
    DetInput out;
    const float scale = target_size > 0 ? static_cast<float>(target_size) /
                                              static_cast<float>(std::max(img.w, img.h))
                                        : 1.0F;
    std::int32_t rw = std::max(1, static_cast<std::int32_t>(std::lround(static_cast<float>(img.w) * scale)));
    std::int32_t rh = std::max(1, static_cast<std::int32_t>(std::lround(static_cast<float>(img.h) * scale)));
    rw = std::min(rw, target_size > 0 ? target_size : rw);
    rh = std::min(rh, target_size > 0 ? target_size : rh);

    const std::int32_t pad_w = (rw + kDetStride - 1) / kDetStride * kDetStride - rw;
    const std::int32_t pad_h = (rh + kDetStride - 1) / kDetStride * kDetStride - rh;
    const std::int32_t pw = rw + pad_w;
    const std::int32_t ph = rh + pad_h;

    std::vector<std::uint8_t> resized;
    resize_rgb(img.rgb.data(), img.w, img.h, rw, rh, resized);

    out.blob.create(pw, ph, kBytesPerPixel);
    const std::int32_t pad_left = pad_w / 2;
    const std::int32_t pad_top = pad_h / 2;
    for (std::int32_t c = 0; c < kBytesPerPixel; ++c) {
        float* dst = out.blob.channel(c);
        const float mean = c == 0 ? kDetMeanR : (c == 1 ? kDetMeanG : kDetMeanB);
        const float norm = c == 0 ? kDetNormR : (c == 1 ? kDetNormG : kDetNormB);
        for (std::int32_t y = 0; y < ph; ++y) {
            for (std::int32_t x = 0; x < pw; ++x) {
                float value = static_cast<float>(kDetPadValue);
                if (x >= pad_left && x < pad_left + rw && y >= pad_top && y < pad_top + rh) {
                    const std::int32_t sx = x - pad_left;
                    const std::int32_t sy = y - pad_top;
                    value = static_cast<float>(resized[rgb_offset(sx, sy, rw) + static_cast<std::size_t>(c)]);
                }
                dst[static_cast<std::size_t>(y) * static_cast<std::size_t>(pw) + static_cast<std::size_t>(x)] =
                    (value - mean) * norm;
            }
        }
    }

    out.scale = scale;
    out.resized_w = rw;
    out.resized_h = rh;
    out.pad_x = pad_left;
    out.pad_y = pad_top;
    return out;
}

std::vector<Box> det_postprocess(const mnet::Mat& prob, const DetInput& in, std::int32_t img_w, std::int32_t img_h) {
    const std::int32_t w = prob.w;
    const std::int32_t h = prob.h;
    const float* p = prob.channel(0);

    std::vector<std::uint8_t> mask(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
    for (std::int32_t i = 0; i < w * h; ++i) {
        mask[static_cast<std::size_t>(i)] = p[i] >= kDetThreshold ? 1 : 0;
    }

    std::vector<Box> boxes;
    std::vector<std::int32_t> stack;
    for (std::int32_t y = 0; y < h; ++y) {
        for (std::int32_t x = 0; x < w; ++x) {
            const std::int32_t start = y * w + x;
            if (mask[static_cast<std::size_t>(start)] == 0) {
                continue;
            }
            stack.clear();
            stack.push_back(start);
            mask[static_cast<std::size_t>(start)] = 0;

            std::int32_t min_x = x;
            std::int32_t max_x = x;
            std::int32_t min_y = y;
            std::int32_t max_y = y;
            double score_sum = 0.0;
            std::int32_t count = 0;

            while (!stack.empty()) {
                const std::int32_t index = stack.back();
                stack.pop_back();
                const std::int32_t cx = index % w;
                const std::int32_t cy = index / w;
                score_sum += static_cast<double>(p[index]);
                ++count;
                min_x = std::min(min_x, cx);
                max_x = std::max(max_x, cx);
                min_y = std::min(min_y, cy);
                max_y = std::max(max_y, cy);

                for (std::int32_t dy = -1; dy <= 1; ++dy) {
                    for (std::int32_t dx = -1; dx <= 1; ++dx) {
                        const std::int32_t nx = cx + dx;
                        const std::int32_t ny = cy + dy;
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h) {
                            continue;
                        }
                        const std::int32_t ni = ny * w + nx;
                        if (mask[static_cast<std::size_t>(ni)] != 0) {
                            mask[static_cast<std::size_t>(ni)] = 0;
                            stack.push_back(ni);
                        }
                    }
                }
            }

            if (count <= 2) {
                continue;
            }
            const float score = static_cast<float>(score_sum / count);
            if (score < kBoxScoreThreshold) {
                continue;
            }

            float bw = static_cast<float>(max_x - min_x + 1);
            float bh = static_cast<float>(max_y - min_y + 1);
            if (std::max(bw, bh) < kMinBoxSizePx * in.scale) {
                continue;
            }

            // Rectangular stand-in for DB unclip. DB expands a text region
            // along its normal by ~half the text thickness, so the short side
            // is scaled by the enlarge ratio and the long side grows by the
            // original thickness (the ncnn example expresses the same idea
            // with minAreaRect, where "width" ends up as the short side).
            const float cx = (static_cast<float>(min_x) + static_cast<float>(max_x)) * 0.5F;
            const float cy = (static_cast<float>(min_y) + static_cast<float>(max_y)) * 0.5F;
            const bool horizontal = bw >= bh;
            const float thickness = horizontal ? bh : bw;
            const float length = horizontal ? bw : bh;
            const float new_thickness = thickness * kBoxEnlargeWidth;
            const float new_length = length + thickness * kBoxEnlargeHeight;
            bw = horizontal ? new_length : new_thickness;
            bh = horizontal ? new_thickness : new_length;

            Box box;
            box.x = (cx - bw * 0.5F - static_cast<float>(in.pad_x)) / in.scale;
            box.y = (cy - bh * 0.5F - static_cast<float>(in.pad_y)) / in.scale;
            box.w = bw / in.scale;
            box.h = bh / in.scale;
            box.score = score;

            box.x = std::clamp(box.x, 0.0F, static_cast<float>(img_w - 1));
            box.y = std::clamp(box.y, 0.0F, static_cast<float>(img_h - 1));
            box.w = std::clamp(box.w, 1.0F, static_cast<float>(img_w) - box.x);
            box.h = std::clamp(box.h, 1.0F, static_cast<float>(img_h) - box.y);
            boxes.push_back(box);
            if (static_cast<std::int32_t>(boxes.size()) >= kMaxBoxes) {
                break;
            }
        }
    }

    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        if (std::fabs(a.y - b.y) > 8.0F) {
            return a.y < b.y;
        }
        return a.x < b.x;
    });
    return boxes;
}

mnet::Mat make_rec_input(const Image& img, const Box& box) {
    const std::int32_t x0 = std::max(0, static_cast<std::int32_t>(box.x));
    const std::int32_t y0 = std::max(0, static_cast<std::int32_t>(box.y));
    const std::int32_t x1 = std::min(img.w, static_cast<std::int32_t>(box.x + box.w));
    const std::int32_t y1 = std::min(img.h, static_cast<std::int32_t>(box.y + box.h));
    const std::int32_t cw = std::max(1, x1 - x0);
    const std::int32_t ch = std::max(1, y1 - y0);

    std::vector<std::uint8_t> crop(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch) * kBytesPerPixel);
    for (std::int32_t y = 0; y < ch; ++y) {
        std::memcpy(crop.data() + rgb_offset(0, y, cw), img.rgb.data() + rgb_offset(x0, y0 + y, img.w),
                    static_cast<std::size_t>(cw) * static_cast<std::size_t>(kBytesPerPixel));
    }

    const std::int32_t rw = std::clamp(static_cast<std::int32_t>(std::lround(static_cast<float>(cw) * kRecHeight /
                                                                           static_cast<float>(ch))),
                                       kRecMinWidth, kRecMaxWidth);
    std::vector<std::uint8_t> resized;
    resize_rgb(crop.data(), cw, ch, rw, kRecHeight, resized);

    mnet::Mat blob(rw, kRecHeight, kBytesPerPixel);
    for (std::int32_t c = 0; c < kBytesPerPixel; ++c) {
        float* dst = blob.channel(c);
        for (std::int32_t y = 0; y < kRecHeight; ++y) {
            for (std::int32_t x = 0; x < rw; ++x) {
                const float value = static_cast<float>(resized[rgb_offset(x, y, rw) + static_cast<std::size_t>(c)]);
                dst[static_cast<std::size_t>(y) * static_cast<std::size_t>(rw) + static_cast<std::size_t>(x)] =
                    (value - kRecMean) * kRecNorm;
            }
        }
    }
    return blob;
}

std::vector<std::string> load_dict(const std::string& path) {
    std::vector<std::string> dict;
    std::ifstream file(path);
    if (!file) {
        std::fprintf(stderr, "cannot open dict %s\n", path.c_str());
        return dict;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        dict.push_back(line);
    }
    return dict;
}

std::string ctc_decode(const mnet::Mat& out, const std::vector<std::string>& dict) {
    std::string text;
    std::int32_t last = 0;
    for (std::int32_t t = 0; t < out.h; ++t) {
        const float* row = out.row(t);
        std::int32_t best = 0;
        float best_score = -1e30F;
        for (std::int32_t c = 0; c < out.w; ++c) {
            if (row[c] > best_score) {
                best_score = row[c];
                best = c;
            }
        }
        if (best == last) {
            continue;
        }
        last = best;
        if (best <= 0) {
            continue;
        }
        const std::int32_t index = best - 1;
        if (index >= 0 && index < static_cast<std::int32_t>(dict.size())) {
            text += dict[static_cast<std::size_t>(index)];
        }
    }
    return text;
}

}  // namespace

int main(int argc, char** argv) {
    std::string det_param;
    std::string det_bin;
    std::string rec_param;
    std::string rec_bin;
    std::string dict_path;
    std::string image_path;
    std::int32_t threads = 4;
    std::int32_t det_size = kDetDefaultSize;
    std::int32_t iters = 3;

    for (std::int32_t i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        const bool has_next = i + 1 < argc;
        if (std::strcmp(arg, "--det-param") == 0 && has_next) {
            det_param = argv[++i];
        } else if (std::strcmp(arg, "--det-bin") == 0 && has_next) {
            det_bin = argv[++i];
        } else if (std::strcmp(arg, "--rec-param") == 0 && has_next) {
            rec_param = argv[++i];
        } else if (std::strcmp(arg, "--rec-bin") == 0 && has_next) {
            rec_bin = argv[++i];
        } else if (std::strcmp(arg, "--dict") == 0 && has_next) {
            dict_path = argv[++i];
        } else if (std::strcmp(arg, "--image") == 0 && has_next) {
            image_path = argv[++i];
        } else if (std::strcmp(arg, "--threads") == 0 && has_next) {
            threads = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--det-size") == 0 && has_next) {
            det_size = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--iters") == 0 && has_next) {
            iters = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr,
                         "usage: mnet_ocr --image IN.bmp --det-param P --det-bin B --rec-param P2 --rec-bin B2 "
                         "--dict DICT.txt [--threads N] [--det-size N] [--iters N]\n");
            return 2;
        }
    }
    if (image_path.empty() || det_param.empty() || det_bin.empty() || rec_param.empty() || rec_bin.empty() ||
        dict_path.empty()) {
        std::fprintf(stderr, "missing required argument\n");
        return 2;
    }

    Image image;
    if (!load_bmp(image_path, image)) {
        return 1;
    }
    std::printf("image=%s %dx%d\n", image_path.c_str(), image.w, image.h);

    const std::vector<std::string> dict = load_dict(dict_path);
    if (dict.empty()) {
        return 1;
    }
    std::printf("dict=%zu cpus=%d big=%d threads=%d\n", dict.size(), mplatform::get_cpu_count(),
                mplatform::get_big_cpu_count(), threads);

    mnet::Net det_net;
    det_net.opt().num_threads = threads;
    if (det_net.load(det_param.c_str(), det_bin.c_str()) != 0) {
        std::fprintf(stderr, "det load failed\n");
        return 1;
    }
    mnet::Net rec_net;
    rec_net.opt().num_threads = threads;
    if (rec_net.load(rec_param.c_str(), rec_bin.c_str()) != 0) {
        std::fprintf(stderr, "rec load failed\n");
        return 1;
    }

    double det_ms = 0.0;
    double rec_ms = 0.0;
    std::vector<Box> boxes;
    std::vector<std::string> texts;

    for (std::int32_t iter = 0; iter < iters; ++iter) {
        DetInput det_in = make_det_input(image, det_size);
        mnet::Net::Extractor det_ex = det_net.create_extractor();
        if (det_ex.input("in0", det_in.blob) != 0) {
            std::fprintf(stderr, "det input failed\n");
            return 1;
        }
        mnet::Mat prob;
        const auto det_start = std::chrono::steady_clock::now();
        if (det_ex.extract("out0", prob) != 0) {
            std::fprintf(stderr, "det extract failed\n");
            return 1;
        }
        const auto det_end = std::chrono::steady_clock::now();
        const double det_cost = std::chrono::duration<double, std::milli>(det_end - det_start).count();

        if (iter == 0) {
            float min_prob = 1.0F;
            float max_prob = 0.0F;
            double sum_prob = 0.0;
            std::int32_t above = 0;
            const std::int32_t prob_count = prob.w * prob.h;
            for (std::int32_t i = 0; i < prob_count; ++i) {
                const float v = prob.channel(0)[static_cast<std::size_t>(i)];
                min_prob = std::min(min_prob, v);
                max_prob = std::max(max_prob, v);
                sum_prob += static_cast<double>(v);
                above += v >= kDetThreshold ? 1 : 0;
            }
            std::printf("det_stats min=%.3f max=%.3f mean=%.3f above_%.1f=%.2f%%\n", static_cast<double>(min_prob),
                        static_cast<double>(max_prob),
                        sum_prob / static_cast<double>(prob.w * prob.h), static_cast<double>(kDetThreshold),
                        100.0 * static_cast<double>(above) / static_cast<double>(prob.w * prob.h));
        }
        boxes = det_postprocess(prob, det_in, image.w, image.h);

        const auto rec_start = std::chrono::steady_clock::now();
        texts.clear();
        for (const Box& box : boxes) {
            mnet::Mat crop = make_rec_input(image, box);
            mnet::Net::Extractor rec_ex = rec_net.create_extractor();
            if (rec_ex.input("in0", crop) != 0) {
                std::fprintf(stderr, "rec input failed\n");
                return 1;
            }
            mnet::Mat rec_out;
            if (rec_ex.extract("out0", rec_out) != 0) {
                std::fprintf(stderr, "rec extract failed\n");
                return 1;
            }
            texts.push_back(ctc_decode(rec_out, dict));
        }
        const auto rec_end = std::chrono::steady_clock::now();
        const double rec_cost = std::chrono::duration<double, std::milli>(rec_end - rec_start).count();

        det_ms += det_cost;
        rec_ms += rec_cost;
        std::printf("iter=%d det=%.2fms rec=%.2fms total=%.2fms boxes=%zu\n", iter, det_cost, rec_cost,
                    det_cost + rec_cost, boxes.size());
        if (iter == 0) {
            std::printf("det_map=%dx%d boxes=%zu\n", prob.w, prob.h, boxes.size());
            for (std::size_t i = 0; i < boxes.size(); ++i) {
                std::printf("  [%zu] score=%.3f box=(%.0f,%.0f,%.0fx%.0f) text=%s\n", i,
                            static_cast<double>(boxes[i].score), static_cast<double>(boxes[i].x),
                            static_cast<double>(boxes[i].y), static_cast<double>(boxes[i].w),
                            static_cast<double>(boxes[i].h), texts[i].c_str());
            }
        }
    }

    std::printf("avg det=%.2fms rec=%.2fms total=%.2fms\n", det_ms / static_cast<double>(iters),
                rec_ms / static_cast<double>(iters), (det_ms + rec_ms) / static_cast<double>(iters));
    return 0;
}
