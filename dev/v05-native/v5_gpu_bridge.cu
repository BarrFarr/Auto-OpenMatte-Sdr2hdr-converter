#include "v5_gpu_bridge.h"

#include <cuda_runtime.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
}

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

enum V5SlotState : int {
    V5_SLOT_FREE = 0,
    V5_SLOT_DECODER_OWNED = 1,
    V5_SLOT_CONSUMER_OWNED = 2,
    V5_SLOT_GPU_PENDING = 3,
};

struct V5Slot {
    AVFrame *frame = nullptr;
    // Explicit ownership of the decoded hardware surface. The AVFrame ref and
    // this buffer ref both remain live until consumer_done has completed.
    AVBufferRef *surface_ref = nullptr;
    // FFmpeg fills CUDA AVFrame buffers on its default CUDA stream. This event
    // fences that producer copy before our nonblocking conversion stream reads it.
    cudaEvent_t source_ready = nullptr;
    float *rgb = nullptr;
    cudaEvent_t consumer_done = nullptr;
    int surface_id = -1;
    int state = V5_SLOT_FREE;
    uint64_t sequence = 0;
};

struct V5Decoder {
    AVFormatContext *format = nullptr;
    AVCodecContext *codec = nullptr;
    AVBufferRef *hw_device = nullptr;
    AVPacket *packet = nullptr;
    AVFrame *decoded = nullptr;
    std::vector<V5Slot> slots;
    std::string path;
    std::string decoder_name;
    std::string last_error;
    int source_mode = 0;
    int device_index = 0;
    int stream_index = -1;
    int width = 0;
    int height = 0;
    // Emitted working-RGB geometry. It equals the decoded size unless an HDR
    // target size is requested, in which case the pre-PQ code plane is resampled
    // to that size on the GPU. resample_active stays false when the requested
    // size equals the source, so a scale-1.0 source keeps the original single
    // kernel and does exactly the same work as before.
    int output_width = 0;
    int output_height = 0;
    bool resample_active = false;
    int software_format = AV_PIX_FMT_NONE;
    uint64_t next_sequence = 0;
    int64_t target_timestamp = AV_NOPTS_VALUE;
    bool target_reached = false;
    bool demux_eof = false;
    bool decoder_draining = false;
    bool first_frame_reported = false;
    uint64_t acquired_count = 0;
    uint64_t released_count = 0;
    std::mutex lifecycle_mutex;
    // Vertical chroma reconstruction filter for 10-bit 4:2:0 sources. It is a
    // function of frame height only, so the same builder serves the HDR P010
    // source and a 10-bit P010 Open Matte source.
    int16_t *hdr_chroma_filter = nullptr;
    int32_t *hdr_chroma_filter_pos = nullptr;
    int hdr_chroma_filter_size = 0;
    // Exact transfer-function tables. Both source paths feed their transfer
    // decode from a bounded integer code, so the decode is tabulated once at
    // open time instead of evaluated per pixel. See build_transfer_luts.
    float *pq_eotf_lut = nullptr;
    double *bt1886_lut = nullptr;
    // 10-bit Open Matte sources reach BT.1886 through a true 16-bit RGB code
    // instead of swscale's duplicated 8-bit byte, so they need their own table.
    double *bt1886_lut16 = nullptr;
};

struct V5Encoder {
    AVFormatContext *format = nullptr;
    AVCodecContext *codec = nullptr;
    AVStream *stream = nullptr;
    AVBufferRef *hw_device = nullptr;
    AVBufferRef *hw_frames = nullptr;
    AVPacket *packet = nullptr;
    std::string path;
    std::string last_error;
    int width = 0;
    int height = 0;
    int device_index = 0;
    bool header_written = false;
    bool trailer_written = false;
    bool output_surface_reported = false;
    bool first_packet_reported = false;
};

namespace {

constexpr int kHdrMode = 0;
constexpr int kSdrMode = 1;
using Slot = V5Slot;

void bridge_marker(const char *marker, const std::string &details = {}) {
    std::fprintf(
        stderr,
        "[V05_MARKER] %s%s%s\n",
        marker != nullptr ? marker : "UNKNOWN",
        details.empty() ? "" : " ",
        details.c_str()
    );
    std::fflush(stderr);
}

void bridge_error(const char *kind, const std::string &message) {
    std::fprintf(
        stderr,
        "[V05_BRIDGE_ERROR] %s: %s\n",
        kind != nullptr ? kind : "unknown",
        message.c_str()
    );
    std::fflush(stderr);
}

void set_error(V5Decoder *decoder, const char *message) {
    if (decoder != nullptr) {
        decoder->last_error = message != nullptr ? message : "unknown V5 bridge error";
        bridge_error("decoder", decoder->last_error);
    } else {
        bridge_error("decoder", message != nullptr ? message : "unknown V5 bridge error");
    }
}

void set_error(V5Decoder *decoder, const std::string &message) {
    if (decoder != nullptr) {
        decoder->last_error = message;
        bridge_error("decoder", decoder->last_error);
    } else {
        bridge_error("decoder", message);
    }
}

void copy_error(const V5Decoder *decoder, char *buffer, int capacity) {
    if (buffer == nullptr || capacity <= 0) {
        return;
    }
    const char *message = (decoder != nullptr && !decoder->last_error.empty())
        ? decoder->last_error.c_str()
        : "unknown V5 bridge error";
    std::snprintf(buffer, static_cast<size_t>(capacity), "%s", message);
}

std::string ffmpeg_error(int code) {
    char buffer[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(code, buffer, sizeof(buffer));
    return std::string(buffer);
}

bool cuda_ok(V5Decoder *decoder, cudaError_t status, const char *operation) {
    if (status == cudaSuccess) {
        return true;
    }
    set_error(decoder, std::string(operation) + ": " + cudaGetErrorString(status));
    return false;
}

AVPixelFormat choose_cuda_format(AVCodecContext *, const AVPixelFormat *formats) {
    if (formats == nullptr) {
        return AV_PIX_FMT_NONE;
    }
    for (const AVPixelFormat *format = formats; *format != AV_PIX_FMT_NONE; ++format) {
        if (*format == AV_PIX_FMT_CUDA) {
            return AV_PIX_FMT_CUDA;
        }
    }
    return AV_PIX_FMT_NONE;
}

bool frame_is_cuda(V5Decoder *decoder, const AVFrame *frame) {
    if (frame == nullptr || frame->format != AV_PIX_FMT_CUDA) {
        set_error(decoder, "NVDEC did not return an AV_PIX_FMT_CUDA frame");
        return false;
    }
    if (frame->hw_frames_ctx == nullptr) {
        set_error(decoder, "CUDA frame has no AVHWFramesContext");
        return false;
    }
    const AVHWFramesContext *frames = reinterpret_cast<const AVHWFramesContext *>(
        frame->hw_frames_ctx->data
    );
    if (frames == nullptr) {
        set_error(decoder, "CUDA frame has an invalid AVHWFramesContext");
        return false;
    }
    if (frames->sw_format != AV_PIX_FMT_P010LE && frames->sw_format != AV_PIX_FMT_NV12) {
        set_error(
            decoder,
            "Unsupported CUDA software format: " + std::to_string(frames->sw_format)
        );
        return false;
    }
    decoder->software_format = frames->sw_format;
    return true;
}

size_t decoder_in_flight_locked(const V5Decoder *decoder) {
    size_t count = 0;
    for (const Slot &slot : decoder->slots) {
        if (slot.state != V5_SLOT_FREE) {
            ++count;
        }
    }
    return count;
}

void emit_surface_counters(V5Decoder *decoder, const char *phase) {
    std::lock_guard<std::mutex> lock(decoder->lifecycle_mutex);
    const uint64_t in_flight = static_cast<uint64_t>(decoder_in_flight_locked(decoder));
    const bool invariant = decoder->acquired_count == decoder->released_count + in_flight;
    bridge_marker(
        "NVDEC_SURFACE_COUNTERS",
        "phase=" + std::string(phase != nullptr ? phase : "unknown") +
        " decoder=" + decoder->decoder_name +
        " acquired=" + std::to_string(decoder->acquired_count) +
        " released=" + std::to_string(decoder->released_count) +
        " in_flight=" + std::to_string(in_flight) +
        " invariant=" + std::to_string(invariant ? 1 : 0)
    );
}

bool retain_surface_reference(V5Decoder *decoder, Slot &slot, const AVFrame *source) {
    if (source == nullptr) {
        set_error(decoder, "cannot retain a surface from a null AVFrame");
        return false;
    }
    for (int index = 0; index < AV_NUM_DATA_POINTERS; ++index) {
        if (source->buf[index] == nullptr) {
            continue;
        }
        slot.surface_ref = av_buffer_ref(source->buf[index]);
        if (slot.surface_ref == nullptr) {
            set_error(decoder, "av_buffer_ref for CUDA surface failed");
            return false;
        }
        return true;
    }
    set_error(decoder, "CUDA AVFrame has no surface AVBufferRef");
    return false;
}

void transition_slot_to_free_locked(V5Decoder *decoder, Slot &slot) {
    // Drop the explicit surface handle before the AVFrame reference. Both are
    // released only after the slot-specific CUDA completion event is done.
    av_buffer_unref(&slot.surface_ref);
    av_frame_unref(slot.frame);
    if (slot.state != V5_SLOT_FREE) {
        slot.state = V5_SLOT_FREE;
        ++decoder->released_count;
    }
}

bool acquire_slot(V5Decoder *decoder, int *slot_index) {
    int pending_index = -1;
    cudaEvent_t pending_event = nullptr;
    {
        std::lock_guard<std::mutex> lock(decoder->lifecycle_mutex);
        for (size_t index = 0; index < decoder->slots.size(); ++index) {
            Slot &slot = decoder->slots[index];
            if (slot.state == V5_SLOT_FREE) {
                slot.state = V5_SLOT_DECODER_OWNED;
                ++decoder->acquired_count;
                *slot_index = slot.surface_id;
                return true;
            }
        }
        for (size_t index = 0; index < decoder->slots.size(); ++index) {
            Slot &slot = decoder->slots[index];
            if (slot.state == V5_SLOT_GPU_PENDING) {
                pending_index = static_cast<int>(index);
                pending_event = slot.consumer_done;
                break;
            }
        }
    }

    if (pending_index >= 0) {
        if (!cuda_ok(decoder, cudaEventSynchronize(pending_event), "cudaEventSynchronize")) {
            return false;
        }
        std::lock_guard<std::mutex> lock(decoder->lifecycle_mutex);
        Slot &slot = decoder->slots[static_cast<size_t>(pending_index)];
        if (slot.state != V5_SLOT_GPU_PENDING || slot.consumer_done != pending_event) {
            set_error(decoder, "NVDEC surface ownership changed while waiting for completion event");
            return false;
        }
        transition_slot_to_free_locked(decoder, slot);
        slot.state = V5_SLOT_DECODER_OWNED;
        ++decoder->acquired_count;
        *slot_index = slot.surface_id;
        return true;
    }

    set_error(decoder, "GPU surface ring has no reusable slot; release every returned frame");
    return false;
}

int hdr_av_log2(int value) {
    if (value <= 0) {
        return 0;
    }
    int result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

int64_t hdr_bicubic_coeff(
    int64_t distance,
    int64_t fone,
    int64_t b_value,
    int64_t c_value
) {
    if (distance >= (1LL << 31)) {
        return 0;
    }
    const int64_t dd = (distance * distance) >> 30;
    const int64_t ddd = (dd * distance) >> 30;
    int64_t coeff;
    if (distance < (1LL << 30)) {
        coeff =
            (12 * (1LL << 24) - 9 * b_value - 6 * c_value) * ddd
            + (-18 * (1LL << 24) + 12 * b_value + 6 * c_value) * dd
            + (6 * (1LL << 24) - 2 * b_value) * (1LL << 30);
    } else {
        coeff =
            (-b_value - 6 * c_value) * ddd
            + (6 * b_value + 30 * c_value) * dd
            + (-12 * b_value - 48 * c_value) * distance
            + (8 * b_value + 24 * c_value) * (1LL << 30);
    }
    return coeff / ((1LL << 54) / fone);
}

bool build_hdr_chroma_filter(V5Decoder *decoder) {
    const int src_size = (decoder->height + 1) >> 1;
    const int dst_size = decoder->height;
    if (src_size < 3 || dst_size <= 0) {
        set_error(decoder, "HDR chroma filter requires a valid 4:2:0 geometry");
        return false;
    }

    const int64_t x_inc =
        (((static_cast<int64_t>(src_size) << 16) + (dst_size >> 1)) / dst_size);
    const int src_pos = (128 << 1) - 128 + 128 >> 1;
    const int dst_pos = (128 << 0) - 128 + 128 >> 0;
    const int ratio = src_size / dst_size;
    const int64_t fone = 1LL << (54 - std::min(hdr_av_log2(ratio), 8));
    const int64_t b_value = 0;
    const int64_t c_value = 10066329;  // (int64_t)(0.6 * (1 << 24)) in FFmpeg.
    int filter_size;
    if (std::llabs(x_inc - 0x10000) < 10 && src_pos == dst_pos) {
        filter_size = 1;
    } else if (x_inc <= (1LL << 16)) {
        filter_size = 5;
    } else {
        filter_size = 1 + (4 * src_size + dst_size - 1) / dst_size;
    }
    filter_size = std::min(filter_size, src_size - 2);
    filter_size = std::max(filter_size, 1);

    const int64_t initial_x =
        ((static_cast<int64_t>(dst_pos) * x_inc) >> 7)
        - ((static_cast<int64_t>(src_pos) * 0x10000) >> 7);
    int64_t x_dst_in_src = initial_x;
    std::vector<int> positions(static_cast<size_t>(dst_size));
    std::vector<std::vector<int64_t>> rows(
        static_cast<size_t>(dst_size),
        std::vector<int64_t>(static_cast<size_t>(filter_size), 0)
    );
    for (int index = 0; index < dst_size; ++index) {
        if (filter_size == 1 && std::llabs(x_inc - 0x10000) < 10 && src_pos == dst_pos) {
            positions[static_cast<size_t>(index)] = index;
            rows[static_cast<size_t>(index)][0] = fone;
        } else {
            int xx = (x_dst_in_src - (filter_size - 2) * (1LL << 16)) / (1 << 17);
            positions[static_cast<size_t>(index)] = xx;
            for (int tap = 0; tap < filter_size; ++tap) {
                const int64_t distance_base =
                    static_cast<int64_t>(xx) * (1LL << 17) - x_dst_in_src;
                const int64_t distance =
                    (distance_base < 0 ? -distance_base : distance_base) << 13;
                rows[static_cast<size_t>(index)][static_cast<size_t>(tap)] =
                    hdr_bicubic_coeff(distance, fone, b_value, c_value);
                ++xx;
            }
            x_dst_in_src += 2 * x_inc;
        }
    }

    const double cutoff = 0.002 * static_cast<double>(fone);
    int min_filter_size = 0;
    for (int index = dst_size - 1; index >= 0; --index) {
        auto &row = rows[static_cast<size_t>(index)];
        int minimum = filter_size;
        int64_t accumulated = 0;
        for (int tap = 0; tap < filter_size; ++tap) {
            accumulated += row[0] < 0 ? -row[0] : row[0];
            if (static_cast<double>(accumulated) > cutoff) {
                break;
            }
            if (index < dst_size - 1 && positions[static_cast<size_t>(index)] >=
                                             positions[static_cast<size_t>(index + 1)]) {
                break;
            }
            for (int column = 1; column < filter_size; ++column) {
                row[static_cast<size_t>(column - 1)] = row[static_cast<size_t>(column)];
            }
            row[static_cast<size_t>(filter_size - 1)] = 0;
            ++positions[static_cast<size_t>(index)];
        }
        accumulated = 0;
        for (int column = filter_size - 1; column > 0; --column) {
            const int64_t value = row[static_cast<size_t>(column)];
            accumulated += value < 0 ? -value : value;
            if (static_cast<double>(accumulated) > cutoff) {
                break;
            }
            --minimum;
        }
        min_filter_size = std::max(min_filter_size, minimum);
    }

    const int aligned_size = (min_filter_size + 1) & ~1;
    if (aligned_size <= 0) {
        set_error(decoder, "HDR chroma filter reduction produced an empty filter");
        return false;
    }
    std::vector<int64_t> reduced(
        static_cast<size_t>(dst_size) * static_cast<size_t>(aligned_size),
        0
    );
    for (int index = 0; index < dst_size; ++index) {
        for (int tap = 0; tap < std::min(aligned_size, filter_size); ++tap) {
            reduced[static_cast<size_t>(index) * aligned_size + tap] =
                rows[static_cast<size_t>(index)][static_cast<size_t>(tap)];
        }
    }

    for (int index = 0; index < dst_size; ++index) {
        auto *row = reduced.data() + static_cast<size_t>(index) * aligned_size;
        int position = positions[static_cast<size_t>(index)];
        if (position < 0) {
            for (int tap = 1; tap < aligned_size; ++tap) {
                const int left = std::max(tap + position, 0);
                row[left] += row[tap];
                row[tap] = 0;
            }
            position = 0;
        }
        if (position + aligned_size > src_size) {
            const int shift = position + std::min(aligned_size - src_size, 0);
            int64_t accumulated = 0;
            for (int tap = aligned_size - 1; tap >= 0; --tap) {
                if (position + tap >= src_size) {
                    accumulated += row[tap];
                    row[tap] = 0;
                }
            }
            for (int tap = aligned_size - 1; tap >= 0; --tap) {
                if (tap < shift) {
                    row[tap] = 0;
                } else {
                    row[tap] = row[tap - shift];
                }
            }
            position -= shift;
            row[src_size - 1 - position] += accumulated;
        }
        positions[static_cast<size_t>(index)] = position;
    }

    std::vector<int16_t> normalized(
        static_cast<size_t>(dst_size) * static_cast<size_t>(aligned_size),
        0
    );
    constexpr int64_t one = 1LL << 12;
    for (int index = 0; index < dst_size; ++index) {
        const auto *row = reduced.data() + static_cast<size_t>(index) * aligned_size;
        int64_t sum = 0;
        for (int tap = 0; tap < aligned_size; ++tap) {
            sum += row[tap];
        }
        sum = (sum + one / 2) / one;
        if (!sum) {
            sum = 1;
        }
        int64_t error = 0;
        for (int tap = 0; tap < aligned_size; ++tap) {
            const int64_t value = row[tap] + error;
            const int64_t adjusted = value >= 0 ? value + sum / 2 : value - sum / 2;
            const int64_t integer_value = adjusted / sum;
            normalized[static_cast<size_t>(index) * aligned_size + tap] =
                static_cast<int16_t>(integer_value);
            error = value - integer_value * sum;
        }
    }

    decoder->hdr_chroma_filter_size = aligned_size;
    const size_t coefficient_count =
        static_cast<size_t>(dst_size) * static_cast<size_t>(aligned_size);
    if (!cuda_ok(
            decoder,
            cudaMalloc(
                reinterpret_cast<void **>(&decoder->hdr_chroma_filter),
                coefficient_count * sizeof(int16_t)
            ),
            "cudaMalloc HDR chroma filter"
        )) {
        return false;
    }
    if (!cuda_ok(
            decoder,
            cudaMalloc(
                reinterpret_cast<void **>(&decoder->hdr_chroma_filter_pos),
                static_cast<size_t>(dst_size) * sizeof(int32_t)
            ),
            "cudaMalloc HDR chroma filter positions"
        )) {
        return false;
    }
    if (!cuda_ok(
            decoder,
            cudaMemcpy(
                decoder->hdr_chroma_filter,
                normalized.data(),
                coefficient_count * sizeof(int16_t),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy HDR chroma filter"
        )) {
        return false;
    }
    if (!cuda_ok(
            decoder,
            cudaMemcpy(
                decoder->hdr_chroma_filter_pos,
                positions.data(),
                static_cast<size_t>(dst_size) * sizeof(int32_t),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy HDR chroma filter positions"
        )) {
        return false;
    }
    return true;
}

__device__ inline double clamp01(double value) {
    return fmin(fmax(value, 0.0), 1.0);
}

__device__ inline double pq_eotf_normalized(double signal) {
    constexpr double m1 = 2610.0 / 16384.0;
    constexpr double m2 = 2523.0 / 4096.0 * 128.0;
    constexpr double c1 = 3424.0 / 4096.0;
    constexpr double c2 = 2413.0 / 4096.0 * 32.0;
    constexpr double c3 = 2392.0 / 4096.0 * 32.0;
    const double clipped = clamp01(signal);
    const double vp = pow(clipped, 1.0 / m2);
    const double numerator = fmax(vp - c1, 0.0);
    const double denominator = fmax(c2 - c3 * vp, 1e-12);
    return pow(numerator / denominator, 1.0 / m1);
}

__device__ inline double bt1886_eotf(double signal) {
    return pow(clamp01(signal), 2.4);
}

__device__ inline int p010_sample(const uint8_t *plane, int pitch, int x, int y) {
    const auto *row = reinterpret_cast<const uint16_t *>(plane + static_cast<size_t>(y) * pitch);
    return static_cast<int>(row[x] >> 6);
}

__device__ inline int nv12_sample(const uint8_t *plane, int pitch, int x, int y) {
    const auto *row = plane + static_cast<size_t>(y) * pitch;
    return static_cast<int>(row[x]);
}

// FFmpeg 7.1.1 swscale's limited-range BT.709 yuv2rgb RGB48 path uses an
// 8-bit LUT and duplicates each 8-bit result into both RGB48 bytes.  These
// helpers reproduce that LUT for the V4-compatible SDR source path.
__device__ inline int swscale_clip_uint8(int64_t value) {
    return static_cast<int>(value < 0 ? 0 : (value > 255 ? 255 : value));
}

__device__ inline int swscale_luma_table_value(int table_index) {
    constexpr int64_t kYCoeff = 76309;  // (65536 * 255) / 219
    constexpr int64_t kYBase =
        -(static_cast<int64_t>(384) << 16)
        - static_cast<int64_t>(512) * kYCoeff
        - (static_cast<int64_t>(16) << 16);
    const int64_t table_value =
        kYBase + static_cast<int64_t>(table_index) * kYCoeff;
    return swscale_clip_uint8((table_value + (static_cast<int64_t>(1) << 15)) >> 16);
}

__device__ inline int swscale_chroma_offset(int coefficient, int chroma_code) {
    const int clipped = chroma_code < 0 ? 0 : (chroma_code > 255 ? 255 : chroma_code);
    return -(coefficient >> 9)
        + static_cast<int>((static_cast<int64_t>(clipped) * coefficient) >> 16);
}

__device__ inline int swscale_rgb_code(int y_code, int u_code, int v_code, int channel) {
    constexpr int kCrv = 100902;
    constexpr int kCbu = 118894;
    constexpr int kCgu = -12001;
    constexpr int kCgv = -29993;
    constexpr int kYOffset = 326 + 512;
    if (channel == 0) {
        return swscale_luma_table_value(
            kYOffset + y_code + swscale_chroma_offset(kCrv, v_code)
        );
    }
    if (channel == 1) {
        return swscale_luma_table_value(
            kYOffset + y_code
            + swscale_chroma_offset(kCgu, u_code)
            + swscale_chroma_offset(kCgv, v_code)
        );
    }
    return swscale_luma_table_value(
        kYOffset + y_code + swscale_chroma_offset(kCbu, u_code)
    );
}

__device__ inline int clip_rgb16(int64_t value) {
    return value < 0 ? 0 : (value > 65535 ? 65535 : static_cast<int>(value));
}

// Both transfer decodes consume a *bounded integer* code, not a continuous
// value, so each one is a finite table rather than a per-pixel transcendental:
//
//   HDR : clip_rgb16() yields exactly [0, 65535] -> pq_eotf_normalized(i/65535)
//   SDR : swscale_rgb_code() yields exactly [0, 255] -> bt1886_eotf(i/255)
//
// The tables are filled on the device by the unchanged double-precision
// reference functions above, evaluated at exactly the inputs the per-pixel
// path used to evaluate. Each result is therefore bit-identical to the
// previous implementation while costing one cached load instead of a
// double-precision pow() evaluation. Because the index domains are clamped
// at their source, neither lookup can leave its table.
//
// The BT.1886 table stays double because its value feeds the BT.709 -> BT.2020
// matrix in double. Rounding it to float first is measurably not
// behaviour-preserving: it perturbs the RGB the shot fitter consumes, which
// shifts the fitted gain/spatial fields and reaches 29 differing 10-bit output
// codes over the 20-frame guard interval.
constexpr int kPqEotfLutSize = 65536;
constexpr int kBt1886LutSize = 256;
constexpr int kBt1886Lut16Size = 65536;

// swscale 7.1.1 limited-range yuv2rgb fixed-point coefficients, reproduced
// exactly as ff_yuv2rgb_c_init_tables() derives them for a 16-bit RGB output:
//
//   cy  = (1 << 16) * 255 / 219,  oy = 16 << 16
//   coeff = roundToInt16(value * (1 << 13)) with roundToInt16(f) = (f + (1<<15)) >> 16
//
// The luma coefficient and offset are colourspace independent; only the chroma
// terms differ, taken from ff_yuv2rgb_coeffs[] for the source's signalled
// matrix. The HDR branch below keeps its already validated BT.2020-NCL literals
// (13752, -5328, -1535, 17545 from {110013, 140363, 12277, 42626}); these named
// constants are the BT.709 equivalents from {117489, 138438, 13975, 34925} and
// are used only by the 10-bit Open Matte branch.
constexpr int64_t kSwsLumaCoeff = 9539;
constexpr int64_t kSwsLumaOffset = 8192;
constexpr int64_t kSws709V2R = 14686;
constexpr int64_t kSws709V2G = -4366;
constexpr int64_t kSws709U2G = -1747;
constexpr int64_t kSws709U2B = 17305;

__global__ void build_pq_eotf_lut(float *lut) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= kPqEotfLutSize) {
        return;
    }
    lut[index] = static_cast<float>(
        pq_eotf_normalized(static_cast<double>(index) / 65535.0)
    );
}

__global__ void build_bt1886_lut(double *lut) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= kBt1886LutSize) {
        return;
    }
    lut[index] = bt1886_eotf(static_cast<double>(index) / 255.0);
}

// A 10-bit Open Matte source produces a genuine 16-bit RGB code, so its
// BT.1886 domain is i/65535 exactly as the CPU reference computes
// bt1886_eotf(rgb48_code / 65535.0).
__global__ void build_bt1886_lut16(double *lut) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= kBt1886Lut16Size) {
        return;
    }
    lut[index] = bt1886_eotf(static_cast<double>(index) / 65535.0);
}

__global__ void yuv_to_working_rgb(
    const uint8_t *y_plane,
    const uint8_t *uv_plane,
    int y_pitch,
    int uv_pitch,
    int width,
    int height,
    int software_format,
    int source_mode,
    const int16_t *hdr_chroma_filter,
    const int32_t *hdr_chroma_filter_pos,
    int hdr_chroma_filter_size,
    const float *__restrict__ pq_eotf_lut,
    const double *__restrict__ bt1886_lut,
    const double *__restrict__ bt1886_lut16,
    int sdr_is_p010,
    float *output
) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }

    if (source_mode == kHdrMode) {
        const int chroma_x = (x / 2) * 2;
        const int y_code = p010_sample(y_plane, y_pitch, x, y);
        const int chroma_y = hdr_chroma_filter_pos[y];
        const int16_t *filter =
            hdr_chroma_filter + static_cast<size_t>(y) * hdr_chroma_filter_size;
        int64_t u_acc = -(static_cast<int64_t>(128) << 23);
        int64_t v_acc = -(static_cast<int64_t>(128) << 23);
        for (int tap = 0; tap < hdr_chroma_filter_size; ++tap) {
            const uint16_t *uv_row = reinterpret_cast<const uint16_t *>(
                uv_plane + static_cast<size_t>(chroma_y + tap) * uv_pitch
            );
            const int u_code = static_cast<int>(uv_row[chroma_x] >> 6);
            const int v_code = static_cast<int>(uv_row[chroma_x + 1] >> 6);
            const int64_t coefficient = static_cast<int64_t>(filter[tap]);
            u_acc += (static_cast<int64_t>(u_code) << 9) * coefficient;
            v_acc += (static_cast<int64_t>(v_code) << 9) * coefficient;
        }

        const int64_t y_acc =
            -(static_cast<int64_t>(1) << 30)
            + (static_cast<int64_t>(y_code) << 9) * 4096;
        const int64_t y_scaled = ((y_acc >> 14) + 0x10000 - 8192) * 9539
            + (1 << 13) - (1 << 29);
        const int64_t u_scaled = u_acc >> 14;
        const int64_t v_scaled = v_acc >> 14;
        const int64_t red_fixed = (v_scaled * 13752 + y_scaled) >> 14;
        const int64_t green_fixed = (v_scaled * -5328 + u_scaled * -1535 + y_scaled) >> 14;
        const int64_t blue_fixed = (u_scaled * 17545 + y_scaled) >> 14;
        const float r = __ldg(pq_eotf_lut + clip_rgb16(red_fixed + (1 << 15)));
        const float g = __ldg(pq_eotf_lut + clip_rgb16(green_fixed + (1 << 15)));
        const float b = __ldg(pq_eotf_lut + clip_rgb16(blue_fixed + (1 << 15)));
        const size_t pixel = (static_cast<size_t>(y) * width + x) * 3;
        output[pixel + 0] = r;
        output[pixel + 1] = g;
        output[pixel + 2] = b;
        return;
    }

    // 10-bit Open Matte source. This mirrors the HDR branch's swscale
    // reproduction exactly -- same P010 sampling, same vertical chroma filter,
    // same fixed-point stages, same 16-bit clip -- and differs only in the
    // BT.709 chroma coefficients and the 16-bit BT.1886 domain. The 8-bit NV12
    // path below is left untouched.
    if (sdr_is_p010 != 0) {
        const int chroma_x_10 = (x / 2) * 2;
        const int y_code = p010_sample(y_plane, y_pitch, x, y);
        const int chroma_y_10 = hdr_chroma_filter_pos[y];
        const int16_t *filter =
            hdr_chroma_filter + static_cast<size_t>(y) * hdr_chroma_filter_size;
        int64_t u_acc = -(static_cast<int64_t>(128) << 23);
        int64_t v_acc = -(static_cast<int64_t>(128) << 23);
        for (int tap = 0; tap < hdr_chroma_filter_size; ++tap) {
            const uint16_t *uv_row = reinterpret_cast<const uint16_t *>(
                uv_plane + static_cast<size_t>(chroma_y_10 + tap) * uv_pitch
            );
            const int u_code = static_cast<int>(uv_row[chroma_x_10] >> 6);
            const int v_code = static_cast<int>(uv_row[chroma_x_10 + 1] >> 6);
            const int64_t coefficient = static_cast<int64_t>(filter[tap]);
            u_acc += (static_cast<int64_t>(u_code) << 9) * coefficient;
            v_acc += (static_cast<int64_t>(v_code) << 9) * coefficient;
        }

        const int64_t y_acc =
            -(static_cast<int64_t>(1) << 30)
            + (static_cast<int64_t>(y_code) << 9) * 4096;
        const int64_t y_scaled = ((y_acc >> 14) + 0x10000 - kSwsLumaOffset) * kSwsLumaCoeff
            + (1 << 13) - (1 << 29);
        const int64_t u_scaled = u_acc >> 14;
        const int64_t v_scaled = v_acc >> 14;
        const int64_t red_fixed = (v_scaled * kSws709V2R + y_scaled) >> 14;
        const int64_t green_fixed =
            (v_scaled * kSws709V2G + u_scaled * kSws709U2G + y_scaled) >> 14;
        const int64_t blue_fixed = (u_scaled * kSws709U2B + y_scaled) >> 14;
        const double linear_r = __ldg(bt1886_lut16 + clip_rgb16(red_fixed + (1 << 15)));
        const double linear_g = __ldg(bt1886_lut16 + clip_rgb16(green_fixed + (1 << 15)));
        const double linear_b = __ldg(bt1886_lut16 + clip_rgb16(blue_fixed + (1 << 15)));
        double r10 = 0.6274040 * linear_r + 0.3292820 * linear_g + 0.0433136 * linear_b;
        double g10 = 0.0690970 * linear_r + 0.9195400 * linear_g + 0.0113612 * linear_b;
        double b10 = 0.0163916 * linear_r + 0.0880132 * linear_g + 0.8955950 * linear_b;
        r10 = fmax(r10, 0.0);
        g10 = fmax(g10, 0.0);
        b10 = fmax(b10, 0.0);
        const size_t pixel10 = (static_cast<size_t>(y) * width + x) * 3;
        output[pixel10 + 0] = static_cast<float>(r10);
        output[pixel10 + 1] = static_cast<float>(g10);
        output[pixel10 + 2] = static_cast<float>(b10);
        return;
    }

    const int chroma_x = (x / 2) * 2;
    const int chroma_y = y / 2;
    const int y_byte = nv12_sample(y_plane, y_pitch, x, y);
    const int u_byte = nv12_sample(uv_plane, uv_pitch, chroma_x, chroma_y);
    const int v_byte = nv12_sample(uv_plane, uv_pitch, chroma_x + 1, chroma_y);

    // swscale_rgb_code() already returns the clamped 8-bit code that the
    // previous implementation divided by 255 and clamped to [0, 1], so the
    // table index reproduces that value exactly.
    const double linear_r = __ldg(bt1886_lut + swscale_rgb_code(y_byte, u_byte, v_byte, 0));
    const double linear_g = __ldg(bt1886_lut + swscale_rgb_code(y_byte, u_byte, v_byte, 1));
    const double linear_b = __ldg(bt1886_lut + swscale_rgb_code(y_byte, u_byte, v_byte, 2));
    double r = 0.6274040 * linear_r + 0.3292820 * linear_g + 0.0433136 * linear_b;
    double g = 0.0690970 * linear_r + 0.9195400 * linear_g + 0.0113612 * linear_b;
    double b = 0.0163916 * linear_r + 0.0880132 * linear_g + 0.8955950 * linear_b;
    r = fmax(r, 0.0);
    g = fmax(g, 0.0);
    b = fmax(b, 0.0);

    const size_t pixel = (static_cast<size_t>(y) * width + x) * 3;
    output[pixel + 0] = static_cast<float>(r);
    output[pixel + 1] = static_cast<float>(g);
    output[pixel + 2] = static_cast<float>(b);
}

// Exactly the HDR fixed-point YUV -> RGB48 stage used by yuv_to_working_rgb,
// isolated so the resampling kernel can regenerate the source codes it needs
// without an intermediate buffer. The integer arithmetic is copied verbatim so
// the unscaled HDR path above stays untouched.
__device__ inline void hdr_rgb48_codes(
    const uint8_t *y_plane,
    const uint8_t *uv_plane,
    int y_pitch,
    int uv_pitch,
    int x,
    int y,
    const int16_t *hdr_chroma_filter,
    const int32_t *hdr_chroma_filter_pos,
    int hdr_chroma_filter_size,
    int *codes
) {
    const int chroma_x = (x / 2) * 2;
    const int y_code = p010_sample(y_plane, y_pitch, x, y);
    const int chroma_y = hdr_chroma_filter_pos[y];
    const int16_t *filter =
        hdr_chroma_filter + static_cast<size_t>(y) * hdr_chroma_filter_size;
    int64_t u_acc = -(static_cast<int64_t>(128) << 23);
    int64_t v_acc = -(static_cast<int64_t>(128) << 23);
    for (int tap = 0; tap < hdr_chroma_filter_size; ++tap) {
        const uint16_t *uv_row = reinterpret_cast<const uint16_t *>(
            uv_plane + static_cast<size_t>(chroma_y + tap) * uv_pitch
        );
        const int u_code = static_cast<int>(uv_row[chroma_x] >> 6);
        const int v_code = static_cast<int>(uv_row[chroma_x + 1] >> 6);
        const int64_t coefficient = static_cast<int64_t>(filter[tap]);
        u_acc += (static_cast<int64_t>(u_code) << 9) * coefficient;
        v_acc += (static_cast<int64_t>(v_code) << 9) * coefficient;
    }
    const int64_t y_acc =
        -(static_cast<int64_t>(1) << 30)
        + (static_cast<int64_t>(y_code) << 9) * 4096;
    const int64_t y_scaled = ((y_acc >> 14) + 0x10000 - 8192) * 9539
        + (1 << 13) - (1 << 29);
    const int64_t u_scaled = u_acc >> 14;
    const int64_t v_scaled = v_acc >> 14;
    const int64_t red_fixed = (v_scaled * 13752 + y_scaled) >> 14;
    const int64_t green_fixed = (v_scaled * -5328 + u_scaled * -1535 + y_scaled) >> 14;
    const int64_t blue_fixed = (u_scaled * 17545 + y_scaled) >> 14;
    codes[0] = clip_rgb16(red_fixed + (1 << 15));
    codes[1] = clip_rgb16(green_fixed + (1 << 15));
    codes[2] = clip_rgb16(blue_fixed + (1 << 15));
}

// Native replacement for openmatte_hdr.decode_hdr:
//
//   code = cv2.resize(raw / 65535.0, (dst_w, dst_h), cv2.INTER_LINEAR)
//   out  = pq_eotf(code) / PEAK_NITS
//
// The resampling therefore happens in the normalized pre-PQ code domain, and
// the transfer function is evaluated afterwards on the interpolated value. That
// ordering is what makes a code-indexed PQ table impossible here, so this kernel
// keeps the reference's double-precision evaluation.
//
// OpenCV's INTER_LINEAR source mapping is reproduced exactly:
//   fx = (dx + 0.5) * src_w / dst_w - 0.5, sx = floor(fx), weight = fx - sx,
// with the source index clamped to the last interpolable pair. The horizontal
// pair is combined first and the vertical pair second, matching OpenCV's
// HResizeLinear followed by VResizeLinear so the rounding order is identical.
__global__ void hdr_resample_pq(
    const uint8_t *y_plane,
    const uint8_t *uv_plane,
    int y_pitch,
    int uv_pitch,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    const int16_t *hdr_chroma_filter,
    const int32_t *hdr_chroma_filter_pos,
    int hdr_chroma_filter_size,
    float *output
) {
    const int dx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int dy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (dx >= dst_width || dy >= dst_height) {
        return;
    }

    const double scale_x = static_cast<double>(src_width) / static_cast<double>(dst_width);
    const double scale_y = static_cast<double>(src_height) / static_cast<double>(dst_height);

    double source_x = (static_cast<double>(dx) + 0.5) * scale_x - 0.5;
    int sx = static_cast<int>(floor(source_x));
    double weight_x = source_x - static_cast<double>(sx);
    if (sx < 0) {
        sx = 0;
        weight_x = 0.0;
    }
    if (sx >= src_width - 1) {
        sx = src_width - 2;
        weight_x = 1.0;
    }

    double source_y = (static_cast<double>(dy) + 0.5) * scale_y - 0.5;
    int sy = static_cast<int>(floor(source_y));
    double weight_y = source_y - static_cast<double>(sy);
    if (sy < 0) {
        sy = 0;
        weight_y = 0.0;
    }
    if (sy >= src_height - 1) {
        sy = src_height - 2;
        weight_y = 1.0;
    }

    const double alpha0 = 1.0 - weight_x;
    const double alpha1 = weight_x;
    const double beta0 = 1.0 - weight_y;
    const double beta1 = weight_y;

    int top_left[3];
    int top_right[3];
    int bottom_left[3];
    int bottom_right[3];
    hdr_rgb48_codes(
        y_plane, uv_plane, y_pitch, uv_pitch, sx, sy,
        hdr_chroma_filter, hdr_chroma_filter_pos, hdr_chroma_filter_size, top_left
    );
    hdr_rgb48_codes(
        y_plane, uv_plane, y_pitch, uv_pitch, sx + 1, sy,
        hdr_chroma_filter, hdr_chroma_filter_pos, hdr_chroma_filter_size, top_right
    );
    hdr_rgb48_codes(
        y_plane, uv_plane, y_pitch, uv_pitch, sx, sy + 1,
        hdr_chroma_filter, hdr_chroma_filter_pos, hdr_chroma_filter_size, bottom_left
    );
    hdr_rgb48_codes(
        y_plane, uv_plane, y_pitch, uv_pitch, sx + 1, sy + 1,
        hdr_chroma_filter, hdr_chroma_filter_pos, hdr_chroma_filter_size, bottom_right
    );

    const size_t pixel = (static_cast<size_t>(dy) * dst_width + dx) * 3;
    #pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
        const double top =
            alpha0 * (static_cast<double>(top_left[channel]) / 65535.0)
            + alpha1 * (static_cast<double>(top_right[channel]) / 65535.0);
        const double bottom =
            alpha0 * (static_cast<double>(bottom_left[channel]) / 65535.0)
            + alpha1 * (static_cast<double>(bottom_right[channel]) / 65535.0);
        const double resampled = beta0 * top + beta1 * bottom;
        output[pixel + channel] = static_cast<float>(pq_eotf_normalized(resampled));
    }
}

bool build_transfer_luts(V5Decoder *decoder) {
    if (decoder->source_mode == kHdrMode) {
        if (!cuda_ok(
                decoder,
                cudaMalloc(
                    reinterpret_cast<void **>(&decoder->pq_eotf_lut),
                    static_cast<size_t>(kPqEotfLutSize) * sizeof(float)
                ),
                "cudaMalloc PQ EOTF table"
            )) {
            return false;
        }
        build_pq_eotf_lut<<<(kPqEotfLutSize + 255) / 256, 256>>>(decoder->pq_eotf_lut);
        if (!cuda_ok(decoder, cudaGetLastError(), "build_pq_eotf_lut launch")) {
            return false;
        }
    } else {
        if (!cuda_ok(
                decoder,
                cudaMalloc(
                    reinterpret_cast<void **>(&decoder->bt1886_lut),
                    static_cast<size_t>(kBt1886LutSize) * sizeof(double)
                ),
                "cudaMalloc BT.1886 table"
            )) {
            return false;
        }
        build_bt1886_lut<<<1, kBt1886LutSize>>>(decoder->bt1886_lut);
        if (!cuda_ok(decoder, cudaGetLastError(), "build_bt1886_lut launch")) {
            return false;
        }
        // The NVDEC surface format is only known once the first frame arrives,
        // so both Open Matte transfer domains are tabulated up front. The 8-bit
        // path keeps using the 256-entry table unchanged.
        if (!cuda_ok(
                decoder,
                cudaMalloc(
                    reinterpret_cast<void **>(&decoder->bt1886_lut16),
                    static_cast<size_t>(kBt1886Lut16Size) * sizeof(double)
                ),
                "cudaMalloc 10-bit BT.1886 table"
            )) {
            return false;
        }
        build_bt1886_lut16<<<(kBt1886Lut16Size + 255) / 256, 256>>>(decoder->bt1886_lut16);
        if (!cuda_ok(decoder, cudaGetLastError(), "build_bt1886_lut16 launch")) {
            return false;
        }
    }
    // The table is produced once on the default stream but read by every
    // conversion on the decoder's nonblocking stream, so it must be fully
    // published before any conversion can observe it.
    return cuda_ok(decoder, cudaDeviceSynchronize(), "transfer table initialization");
}

bool convert_frame(V5Decoder *decoder, Slot &slot, cudaStream_t stream) {
    if (!frame_is_cuda(decoder, slot.frame)) {
        return false;
    }
    const AVHWFramesContext *frames = reinterpret_cast<const AVHWFramesContext *>(
        slot.frame->hw_frames_ctx->data
    );
    const int format = frames->sw_format;
    if (decoder->source_mode == kHdrMode && format != AV_PIX_FMT_P010LE) {
        set_error(decoder, "HDR NVDEC surface is not P010");
        return false;
    }
    // 8-bit Open Matte sources arrive as NV12, 10-bit ones as P010. Both are
    // converted natively on the GPU; there is no P010-to-NV12 downconversion
    // and no host round trip.
    const int sdr_is_p010 = (decoder->source_mode == kSdrMode && format == AV_PIX_FMT_P010LE) ? 1 : 0;
    if (decoder->source_mode == kSdrMode && format != AV_PIX_FMT_NV12 &&
        format != AV_PIX_FMT_P010LE) {
        set_error(decoder, "Open Matte NVDEC surface is neither NV12 nor P010");
        return false;
    }
    if (slot.frame->data[0] == nullptr || slot.frame->data[1] == nullptr) {
        set_error(decoder, "CUDA NVDEC frame has a missing Y or UV plane");
        return false;
    }
    if ((decoder->source_mode == kHdrMode || sdr_is_p010 != 0) &&
        (decoder->hdr_chroma_filter == nullptr || decoder->hdr_chroma_filter_pos == nullptr ||
         decoder->hdr_chroma_filter_size <= 0)) {
        set_error(decoder, "10-bit vertical chroma filter is not initialized");
        return false;
    }
    if (decoder->source_mode == kHdrMode && decoder->pq_eotf_lut == nullptr) {
        set_error(decoder, "HDR PQ EOTF table is not initialized");
        return false;
    }
    if (decoder->source_mode == kSdrMode && decoder->bt1886_lut == nullptr) {
        set_error(decoder, "Open Matte BT.1886 table is not initialized");
        return false;
    }
    if (sdr_is_p010 != 0 && decoder->bt1886_lut16 == nullptr) {
        set_error(decoder, "Open Matte 10-bit BT.1886 table is not initialized");
        return false;
    }
    dim3 block(16, 16, 1);
    if (decoder->resample_active) {
        dim3 resample_grid(
            static_cast<unsigned int>((decoder->output_width + block.x - 1) / block.x),
            static_cast<unsigned int>((decoder->output_height + block.y - 1) / block.y),
            1
        );
        hdr_resample_pq<<<resample_grid, block, 0, stream>>>(
            slot.frame->data[0],
            slot.frame->data[1],
            slot.frame->linesize[0],
            slot.frame->linesize[1],
            decoder->width,
            decoder->height,
            decoder->output_width,
            decoder->output_height,
            decoder->hdr_chroma_filter,
            decoder->hdr_chroma_filter_pos,
            decoder->hdr_chroma_filter_size,
            slot.rgb
        );
        return cuda_ok(decoder, cudaGetLastError(), "hdr_resample_pq launch");
    }
    dim3 grid(
        static_cast<unsigned int>((decoder->width + block.x - 1) / block.x),
        static_cast<unsigned int>((decoder->height + block.y - 1) / block.y),
        1
    );
    yuv_to_working_rgb<<<grid, block, 0, stream>>>(
        slot.frame->data[0],
        slot.frame->data[1],
        slot.frame->linesize[0],
        slot.frame->linesize[1],
        decoder->width,
        decoder->height,
        format,
        decoder->source_mode,
        decoder->hdr_chroma_filter,
        decoder->hdr_chroma_filter_pos,
        decoder->hdr_chroma_filter_size,
        decoder->pq_eotf_lut,
        decoder->bt1886_lut,
        decoder->bt1886_lut16,
        sdr_is_p010,
        slot.rgb
    );
    return cuda_ok(decoder, cudaGetLastError(), "yuv_to_working_rgb launch");
}

bool open_decoder(
    V5Decoder *decoder,
    const char *path,
    const char *decoder_name,
    int64_t start_frame,
    int fps_num,
    int fps_den,
    int device_index,
    int ring_size
) {
    decoder->path = path != nullptr ? path : "";
    decoder->decoder_name = decoder_name != nullptr ? decoder_name : "";
    if (decoder->path.empty() || decoder->decoder_name.empty()) {
        set_error(decoder, "path and decoder_name are required");
        return false;
    }
    if (ring_size < 2 || fps_num <= 0 || fps_den <= 0 || start_frame < 0 || device_index < 0) {
        set_error(decoder, "invalid decoder ring/fps/start/device configuration");
        return false;
    }
    if (!cuda_ok(decoder, cudaSetDevice(device_index), "cudaSetDevice")) {
        return false;
    }
    decoder->device_index = device_index;

    const int open_log_level = av_log_get_level();
    av_log_set_level(AV_LOG_FATAL);
    int status = avformat_open_input(&decoder->format, decoder->path.c_str(), nullptr, nullptr);
    av_log_set_level(open_log_level);
    if (status < 0) {
        set_error(decoder, "avformat_open_input: " + ffmpeg_error(status));
        return false;
    }

    // Match `-map 0:v:0`: select the first video stream before probing. Marking
    // every other stream discarded prevents packet analysis of audio, PGS
    // subtitles, and additional streams. FFmpeg's final stream-info pass still
    // checks discarded streams, so temporarily present them as DATA below to
    // keep that pass video-only without changing their public stream metadata.
    decoder->stream_index = -1;
    std::vector<int> discarded_stream_types(decoder->format->nb_streams, -1);
    for (unsigned int index = 0; index < decoder->format->nb_streams; ++index) {
        AVStream *candidate = decoder->format->streams[index];
        const bool is_first_video =
            decoder->stream_index < 0 &&
            candidate != nullptr &&
            candidate->codecpar != nullptr &&
            candidate->codecpar->codec_type == AVMEDIA_TYPE_VIDEO;
        if (is_first_video) {
            decoder->stream_index = static_cast<int>(index);
            candidate->discard = AVDISCARD_DEFAULT;
        } else if (candidate != nullptr) {
            candidate->discard = AVDISCARD_ALL;
            if (candidate->codecpar != nullptr) {
                discarded_stream_types[index] = candidate->codecpar->codec_type;
                candidate->codecpar->codec_type = AVMEDIA_TYPE_DATA;
            }
        }
    }
    if (decoder->stream_index < 0) {
        set_error(decoder, "no video stream found for video-only input");
        return false;
    }

    const int stream_info_log_level = av_log_get_level();
    av_log_set_level(AV_LOG_FATAL);
    status = avformat_find_stream_info(decoder->format, nullptr);
    av_log_set_level(stream_info_log_level);
    for (unsigned int index = 0; index < decoder->format->nb_streams; ++index) {
        if (discarded_stream_types[index] >= 0) {
            decoder->format->streams[index]->codecpar->codec_type =
                static_cast<AVMediaType>(discarded_stream_types[index]);
        }
    }
    if (status < 0) {
        set_error(decoder, "avformat_find_stream_info: " + ffmpeg_error(status));
        return false;
    }
    AVStream *stream = decoder->format->streams[decoder->stream_index];
    const AVCodecParameters *parameters = stream->codecpar;
    const AVCodec *codec = avcodec_find_decoder_by_name(decoder->decoder_name.c_str());
    if (codec == nullptr) {
        set_error(decoder, "decoder not found: " + decoder->decoder_name);
        return false;
    }
    decoder->codec = avcodec_alloc_context3(codec);
    if (decoder->codec == nullptr) {
        set_error(decoder, "avcodec_alloc_context3 failed");
        return false;
    }
    status = avcodec_parameters_to_context(decoder->codec, parameters);
    if (status < 0) {
        set_error(decoder, "avcodec_parameters_to_context: " + ffmpeg_error(status));
        return false;
    }
    decoder->codec->pkt_timebase = stream->time_base;
    decoder->codec->get_format = choose_cuda_format;
    char device_name[32] = {};
    std::snprintf(device_name, sizeof(device_name), "%d", device_index);
    AVDictionary *device_options = nullptr;
    status = av_dict_set(&device_options, "current_ctx", "1", 0);
    if (status < 0) {
        set_error(decoder, "av_dict_set(current_ctx): " + ffmpeg_error(status));
        return false;
    }
    status = av_hwdevice_ctx_create(
        &decoder->hw_device,
        AV_HWDEVICE_TYPE_CUDA,
        device_name,
        device_options,
        0
    );
    av_dict_free(&device_options);
    if (status < 0) {
        set_error(decoder, "av_hwdevice_ctx_create: " + ffmpeg_error(status));
        return false;
    }
    decoder->codec->hw_device_ctx = av_buffer_ref(decoder->hw_device);
    if (decoder->codec->hw_device_ctx == nullptr) {
        set_error(decoder, "av_buffer_ref(hw_device_ctx) failed");
        return false;
    }
    status = avcodec_open2(decoder->codec, codec, nullptr);
    if (status < 0) {
        set_error(decoder, "avcodec_open2: " + ffmpeg_error(status));
        return false;
    }
    decoder->width = decoder->codec->width;
    decoder->height = decoder->codec->height;
    decoder->output_width = decoder->width;
    decoder->output_height = decoder->height;
    // The filter depends only on frame height, and the NVDEC surface format is
    // not known until the first frame, so it is built for both source modes.
    // An 8-bit NV12 Open Matte source simply never reads it.
    if (!build_hdr_chroma_filter(decoder)) {
        return false;
    }
    if (!build_transfer_luts(decoder)) {
        return false;
    }
    decoder->packet = av_packet_alloc();
    decoder->decoded = av_frame_alloc();
    if (decoder->packet == nullptr || decoder->decoded == nullptr) {
        set_error(decoder, "FFmpeg packet/frame allocation failed");
        return false;
    }

    const AVRational frame_duration{fps_den, fps_num};
    const int64_t relative_timestamp = av_rescale_q(start_frame, frame_duration, stream->time_base);
    const int64_t stream_start = stream->start_time == AV_NOPTS_VALUE ? 0 : stream->start_time;
    decoder->target_timestamp = stream_start + relative_timestamp;
    if (start_frame > 0) {
        status = av_seek_frame(
            decoder->format,
            decoder->stream_index,
            decoder->target_timestamp,
            AVSEEK_FLAG_BACKWARD
        );
        if (status < 0) {
            set_error(decoder, "av_seek_frame: " + ffmpeg_error(status));
            return false;
        }
        avcodec_flush_buffers(decoder->codec);
    }

    const size_t bytes =
        static_cast<size_t>(decoder->output_width) * decoder->output_height * 3 * sizeof(float);
    decoder->slots.resize(static_cast<size_t>(ring_size));
    for (size_t index = 0; index < decoder->slots.size(); ++index) {
        Slot &slot = decoder->slots[index];
        slot.surface_id = static_cast<int>(index);
        slot.frame = av_frame_alloc();
        if (slot.frame == nullptr) {
            set_error(decoder, "slot AVFrame allocation failed");
            return false;
        }
        if (!cuda_ok(
                decoder,
                cudaMalloc(reinterpret_cast<void **>(&slot.rgb), bytes),
                "cudaMalloc working RGB"
            )) {
            return false;
        }
        if (!cuda_ok(
                decoder,
                cudaEventCreateWithFlags(&slot.source_ready, cudaEventDisableTiming),
                "cudaEventCreateWithFlags source_ready"
            )) {
            return false;
        }
        if (!cuda_ok(
                decoder,
                cudaEventCreateWithFlags(&slot.consumer_done, cudaEventDisableTiming),
                "cudaEventCreateWithFlags consumer_done"
            )) {
            return false;
        }
    }
    bridge_marker(
        decoder->source_mode == kHdrMode ? "HDR_DECODER_OK" : "OM_DECODER_OK",
        "decoder=" + decoder->decoder_name +
        " path=" + decoder->path +
        " device=" + std::to_string(decoder->device_index) +
        " geometry=" + std::to_string(decoder->width) + "x" + std::to_string(decoder->height) +
        " stream_index=" + std::to_string(decoder->stream_index)
    );
    return true;
}

bool receive_frame(V5Decoder *decoder, int *result) {
    while (true) {
        av_frame_unref(decoder->decoded);
        int status = avcodec_receive_frame(decoder->codec, decoder->decoded);
        if (status == 0) {
            const AVFrame *frame = decoder->decoded;
            const int64_t timestamp = frame->best_effort_timestamp;
            if (!decoder->target_reached && decoder->target_timestamp != AV_NOPTS_VALUE &&
                timestamp != AV_NOPTS_VALUE && timestamp < decoder->target_timestamp) {
                av_frame_unref(decoder->decoded);
                continue;
            }
            decoder->target_reached = true;
            *result = 1;
            return true;
        }
        if (status == AVERROR(EAGAIN)) {
            if (decoder->demux_eof) {
                if (!decoder->decoder_draining) {
                    status = avcodec_send_packet(decoder->codec, nullptr);
                    decoder->decoder_draining = true;
                    if (status < 0 && status != AVERROR_EOF) {
                        set_error(decoder, "avcodec_send_packet(EOF): " + ffmpeg_error(status));
                        return false;
                    }
                    continue;
                }
                *result = 0;
                return true;
            }
            status = av_read_frame(decoder->format, decoder->packet);
            if (status < 0) {
                decoder->demux_eof = true;
                continue;
            }
            if (decoder->packet->stream_index == decoder->stream_index) {
                status = avcodec_send_packet(decoder->codec, decoder->packet);
                av_packet_unref(decoder->packet);
                if (status < 0 && status != AVERROR(EAGAIN)) {
                    set_error(decoder, "avcodec_send_packet: " + ffmpeg_error(status));
                    return false;
                }
            } else {
                av_packet_unref(decoder->packet);
            }
            continue;
        }
        if (status == AVERROR_EOF) {
            *result = 0;
            return true;
        }
        set_error(decoder, "avcodec_receive_frame: " + ffmpeg_error(status));
        return false;
    }
}

void encoder_set_error(V5Encoder *encoder, const char *message) {
    if (encoder != nullptr) {
        encoder->last_error = message != nullptr ? message : "unknown V5 encoder error";
        bridge_error("encoder", encoder->last_error);
    } else {
        bridge_error("encoder", message != nullptr ? message : "unknown V5 encoder error");
    }
}

void encoder_set_error(V5Encoder *encoder, const std::string &message) {
    if (encoder != nullptr) {
        encoder->last_error = message;
        bridge_error("encoder", encoder->last_error);
    } else {
        bridge_error("encoder", message);
    }
}

void encoder_copy_error(const V5Encoder *encoder, char *buffer, int capacity) {
    if (buffer == nullptr || capacity <= 0) {
        return;
    }
    const char *message = (encoder != nullptr && !encoder->last_error.empty())
        ? encoder->last_error.c_str()
        : "unknown V5 encoder error";
    std::snprintf(buffer, static_cast<size_t>(capacity), "%s", message);
}

bool encoder_cuda_ok(V5Encoder *encoder, cudaError_t status, const char *operation) {
    if (status == cudaSuccess) {
        return true;
    }
    encoder_set_error(encoder, std::string(operation) + ": " + cudaGetErrorString(status));
    return false;
}

void encoder_cleanup(V5Encoder *encoder, bool write_trailer) {
    if (encoder == nullptr) {
        return;
    }
    if (write_trailer && encoder->format != nullptr && encoder->header_written && !encoder->trailer_written) {
        av_write_trailer(encoder->format);
        encoder->trailer_written = true;
    }
    if (encoder->format != nullptr && !(encoder->format->oformat->flags & AVFMT_NOFILE) && encoder->format->pb != nullptr) {
        avio_closep(&encoder->format->pb);
    }
    if (encoder->packet != nullptr) {
        av_packet_free(&encoder->packet);
    }
    if (encoder->codec != nullptr) {
        avcodec_free_context(&encoder->codec);
    }
    if (encoder->hw_frames != nullptr) {
        av_buffer_unref(&encoder->hw_frames);
    }
    if (encoder->hw_device != nullptr) {
        av_buffer_unref(&encoder->hw_device);
    }
    if (encoder->format != nullptr) {
        avformat_free_context(encoder->format);
        encoder->format = nullptr;
    }
    encoder->stream = nullptr;
    encoder->header_written = false;
}

bool encoder_receive_packets(V5Encoder *encoder) {
    while (true) {
        const int status = avcodec_receive_packet(encoder->codec, encoder->packet);
        if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) {
            return true;
        }
        if (status < 0) {
            encoder_set_error(encoder, "avcodec_receive_packet: " + ffmpeg_error(status));
            return false;
        }
        av_packet_rescale_ts(encoder->packet, encoder->codec->time_base, encoder->stream->time_base);
        encoder->packet->stream_index = encoder->stream->index;
        const int64_t packet_pts = encoder->packet->pts;
        const int write_status = av_interleaved_write_frame(encoder->format, encoder->packet);
        av_packet_unref(encoder->packet);
        if (write_status < 0) {
            encoder_set_error(encoder, "av_interleaved_write_frame: " + ffmpeg_error(write_status));
            return false;
        }
        if (!encoder->first_packet_reported) {
            encoder->first_packet_reported = true;
            bridge_marker(
                "FIRST_PACKET_OK",
                "path=" + encoder->path +
                " stream_index=" + std::to_string(encoder->stream->index) +
                " pts=" + std::to_string(packet_pts)
            );
        }
    }
}

bool open_encoder(
    V5Encoder *encoder,
    const char *path,
    int width,
    int height,
    int fps_num,
    int fps_den,
    int device_index,
    int qp
) {
    if (encoder == nullptr || path == nullptr || *path == '\0' || width <= 0 || height <= 0 ||
        fps_num <= 0 || fps_den <= 0 || device_index < 0 || qp < 0 || qp > 51) {
        encoder_set_error(encoder, "invalid GPU P010 encoder configuration");
        return false;
    }
    encoder->path = path;
    encoder->width = width;
    encoder->height = height;
    encoder->device_index = device_index;
    if (!encoder_cuda_ok(encoder, cudaSetDevice(device_index), "cudaSetDevice(encoder)")) {
        return false;
    }

    int status = avformat_alloc_output_context2(
        &encoder->format,
        nullptr,
        "matroska",
        encoder->path.c_str()
    );
    if (status < 0 || encoder->format == nullptr) {
        encoder_set_error(encoder, "avformat_alloc_output_context2: " + ffmpeg_error(status));
        return false;
    }
    const AVCodec *codec = avcodec_find_encoder_by_name("hevc_nvenc");
    if (codec == nullptr) {
        encoder_set_error(encoder, "hevc_nvenc encoder is unavailable");
        return false;
    }
    encoder->stream = avformat_new_stream(encoder->format, nullptr);
    if (encoder->stream == nullptr) {
        encoder_set_error(encoder, "avformat_new_stream failed");
        return false;
    }
    encoder->codec = avcodec_alloc_context3(codec);
    if (encoder->codec == nullptr) {
        encoder_set_error(encoder, "avcodec_alloc_context3(hevc_nvenc) failed");
        return false;
    }
    encoder->codec->width = width;
    encoder->codec->height = height;
    encoder->codec->pix_fmt = AV_PIX_FMT_CUDA;
    encoder->codec->time_base = AVRational{fps_den, fps_num};
    encoder->codec->framerate = AVRational{fps_num, fps_den};
    encoder->codec->sample_aspect_ratio = AVRational{1, 1};
    encoder->codec->gop_size = 60;
    encoder->codec->max_b_frames = 2;
    encoder->codec->color_range = AVCOL_RANGE_MPEG;
    encoder->codec->colorspace = AVCOL_SPC_BT2020_NCL;
    encoder->codec->color_trc = AVCOL_TRC_SMPTE2084;
    encoder->codec->color_primaries = AVCOL_PRI_BT2020;
    // Matroska requires codec extradata in the stream header. Set this before
    // avcodec_open2() so NVENC produces the global HEVC parameter sets.
    encoder->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    av_opt_set(encoder->codec->priv_data, "preset", "p5", 0);
    av_opt_set(encoder->codec->priv_data, "profile", "main10", 0);
    av_opt_set(encoder->codec->priv_data, "rc", "constqp", 0);
    av_opt_set_int(encoder->codec->priv_data, "qp", qp, 0);

    char device_name[32] = {};
    std::snprintf(device_name, sizeof(device_name), "%d", device_index);
    AVDictionary *device_options = nullptr;
    status = av_dict_set(&device_options, "current_ctx", "1", 0);
    if (status < 0) {
        encoder_set_error(encoder, "av_dict_set(current_ctx): " + ffmpeg_error(status));
        av_dict_free(&device_options);
        return false;
    }
    status = av_hwdevice_ctx_create(
        &encoder->hw_device,
        AV_HWDEVICE_TYPE_CUDA,
        device_name,
        device_options,
        0
    );
    av_dict_free(&device_options);
    if (status < 0) {
        encoder_set_error(encoder, "av_hwdevice_ctx_create(encoder): " + ffmpeg_error(status));
        return false;
    }
    encoder->hw_frames = av_hwframe_ctx_alloc(encoder->hw_device);
    if (encoder->hw_frames == nullptr) {
        encoder_set_error(encoder, "av_hwframe_ctx_alloc(encoder) failed");
        return false;
    }
    AVHWFramesContext *frames = reinterpret_cast<AVHWFramesContext *>(encoder->hw_frames->data);
    frames->format = AV_PIX_FMT_CUDA;
    frames->sw_format = AV_PIX_FMT_P010LE;
    frames->width = width;
    frames->height = height;
    frames->initial_pool_size = 4;
    status = av_hwframe_ctx_init(encoder->hw_frames);
    if (status < 0) {
        encoder_set_error(encoder, "av_hwframe_ctx_init(encoder): " + ffmpeg_error(status));
        return false;
    }
    encoder->codec->hw_device_ctx = av_buffer_ref(encoder->hw_device);
    encoder->codec->hw_frames_ctx = av_buffer_ref(encoder->hw_frames);
    if (encoder->codec->hw_device_ctx == nullptr || encoder->codec->hw_frames_ctx == nullptr) {
        encoder_set_error(encoder, "failed to attach CUDA encoder hardware contexts");
        return false;
    }
    status = avcodec_open2(encoder->codec, codec, nullptr);
    if (status < 0) {
        encoder_set_error(encoder, "avcodec_open2(hevc_nvenc): " + ffmpeg_error(status));
        return false;
    }
    bridge_marker(
        "NVENC_CODEC_OPEN_OK",
        "codec=" + std::string(codec->name != nullptr ? codec->name : "unknown") +
        " codec_extradata_size=" + std::to_string(encoder->codec->extradata_size) +
        " global_header=" + std::to_string((encoder->codec->flags & AV_CODEC_FLAG_GLOBAL_HEADER) != 0)
    );
    if (encoder->codec->extradata == nullptr || encoder->codec->extradata_size <= 0) {
        encoder_set_error(encoder, "avcodec_open2(hevc_nvenc) produced no codec extradata");
        return false;
    }
    if (encoder->stream == nullptr || encoder->stream->codecpar == nullptr) {
        encoder_set_error(encoder, "encoder stream has no AVCodecParameters");
        return false;
    }
    encoder->stream->time_base = encoder->codec->time_base;
    encoder->stream->avg_frame_rate = encoder->codec->framerate;
    status = avcodec_parameters_from_context(encoder->stream->codecpar, encoder->codec);
    if (status < 0) {
        encoder_set_error(encoder, "avcodec_parameters_from_context: " + ffmpeg_error(status));
        return false;
    }
    if (encoder->stream->codecpar->extradata == nullptr || encoder->stream->codecpar->extradata_size <= 0) {
        encoder_set_error(encoder, "AVCodecParameters has no codec extradata after avcodec_open2");
        return false;
    }
    bridge_marker(
        "NVENC_PRE_HEADER",
        "path=" + encoder->path +
        " codec=" + std::string(codec->name != nullptr ? codec->name : "unknown") +
        " codec_id=" + std::to_string(encoder->stream->codecpar->codec_id) +
        " codec_format=" + std::to_string(encoder->stream->codecpar->format) +
        " codec_pix_fmt=" + std::to_string(encoder->codec->pix_fmt) +
        " width=" + std::to_string(encoder->stream->codecpar->width) +
        " height=" + std::to_string(encoder->stream->codecpar->height) +
        " extradata_size=" + std::to_string(encoder->stream->codecpar->extradata_size) +
        " codec_extradata_size=" + std::to_string(encoder->codec->extradata_size) +
        " global_header=" + std::to_string((encoder->codec->flags & AV_CODEC_FLAG_GLOBAL_HEADER) != 0) +
        " time_base=" + std::to_string(encoder->stream->time_base.num) + "/" + std::to_string(encoder->stream->time_base.den)
    );
    if (!(encoder->format->oformat->flags & AVFMT_NOFILE)) {
        status = avio_open(&encoder->format->pb, encoder->path.c_str(), AVIO_FLAG_WRITE);
        if (status < 0) {
            encoder_set_error(encoder, "avio_open: " + ffmpeg_error(status));
            return false;
        }
    }
    status = avformat_write_header(encoder->format, nullptr);
    if (status < 0) {
        encoder_set_error(encoder, "avformat_write_header: " + ffmpeg_error(status));
        return false;
    }
    bridge_marker(
        "NVENC_HEADER_OK",
        "path=" + encoder->path +
        " format=" + std::string(encoder->format->oformat->name != nullptr ? encoder->format->oformat->name : "unknown")
    );
    encoder->header_written = true;
    encoder->packet = av_packet_alloc();
    if (encoder->packet == nullptr) {
        encoder_set_error(encoder, "av_packet_alloc(encoder) failed");
        return false;
    }
    bridge_marker(
        "NVENC_OK",
        "path=" + encoder->path +
        " geometry=" + std::to_string(encoder->width) + "x" + std::to_string(encoder->height) +
        " pix_fmt=CUDA sw_format=P010LE device=" + std::to_string(encoder->device_index)
    );
    return true;
}

}  // namespace

extern "C" {

V5_GPU_API V5Decoder *v5_decoder_open(
    const char *path,
    const char *decoder_name,
    int source_mode,
    int64_t start_frame,
    int fps_num,
    int fps_den,
    int device_index,
    int ring_size,
    char *error_buffer,
    int error_capacity
) {
    V5Decoder *decoder = new V5Decoder();
    decoder->source_mode = source_mode;
    if (source_mode != kHdrMode && source_mode != kSdrMode) {
        set_error(decoder, "source_mode must be 0 (HDR) or 1 (SDR)");
    } else if (!open_decoder(
                   decoder,
                   path,
                   decoder_name,
                   start_frame,
                   fps_num,
                   fps_den,
                   device_index,
                   ring_size
               )) {
        copy_error(decoder, error_buffer, error_capacity);
        v5_decoder_close(decoder);
        return nullptr;
    }
    copy_error(decoder, error_buffer, error_capacity);
    return decoder;
}

V5_GPU_API int v5_decoder_next(
    V5Decoder *decoder,
    uint64_t cuda_stream,
    V5GpuFrame *frame,
    char *error_buffer,
    int error_capacity
) {
    if (decoder == nullptr || frame == nullptr) {
        if (error_buffer != nullptr && error_capacity > 0) {
            std::snprintf(error_buffer, static_cast<size_t>(error_capacity), "null decoder/frame");
        }
        return -1;
    }
    int slot_index = -1;
    if (!acquire_slot(decoder, &slot_index)) {
        copy_error(decoder, error_buffer, error_capacity);
        return -1;
    }
    Slot &slot = decoder->slots[static_cast<size_t>(slot_index)];
    bool frame_received = false;
    auto release_reserved_slot = [&]() {
        // Error paths may have queued FFmpeg's default-stream copy or bridge
        // conversion work before returning. Do not drop the AVFrame/surface
        // ownership until all such work is complete.
        if (frame_received) {
            cuda_ok(decoder, cudaDeviceSynchronize(), "cudaDeviceSynchronize reserved slot cleanup");
        }
        std::lock_guard<std::mutex> lock(decoder->lifecycle_mutex);
        transition_slot_to_free_locked(decoder, slot);
    };
    int received = 0;
    if (!receive_frame(decoder, &received)) {
        release_reserved_slot();
        copy_error(decoder, error_buffer, error_capacity);
        return -1;
    }
    if (received == 0) {
        release_reserved_slot();
        copy_error(decoder, error_buffer, error_capacity);
        return 0;
    }
    frame_received = true;
    av_frame_unref(slot.frame);
    if (av_frame_ref(slot.frame, decoder->decoded) < 0) {
        set_error(decoder, "av_frame_ref for CUDA surface failed");
        release_reserved_slot();
        copy_error(decoder, error_buffer, error_capacity);
        return -1;
    }
    if (!retain_surface_reference(decoder, slot, decoder->decoded)) {
        release_reserved_slot();
        copy_error(decoder, error_buffer, error_capacity);
        return -1;
    }
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
    // FFmpeg's CUVID path enqueues the AVFrame surface copy on the CUDA
    // device-context default stream (NULL). Establish an explicit dependency
    // before the bridge reads that AVFrame on the caller's nonblocking stream.
    if (!cuda_ok(
            decoder,
            cudaEventRecord(slot.source_ready, nullptr),
            "cudaEventRecord source_ready"
        ) || !cuda_ok(
            decoder,
            cudaStreamWaitEvent(stream, slot.source_ready, 0),
            "cudaStreamWaitEvent source_ready"
        )) {
        release_reserved_slot();
        copy_error(decoder, error_buffer, error_capacity);
        return -1;
    }
    if (!convert_frame(decoder, slot, stream)) {
        release_reserved_slot();
        copy_error(decoder, error_buffer, error_capacity);
        return -1;
    }
    uint64_t sequence = 0;
    {
        std::lock_guard<std::mutex> lock(decoder->lifecycle_mutex);
        slot.state = V5_SLOT_CONSUMER_OWNED;
        slot.sequence = decoder->next_sequence++;
        sequence = slot.sequence;
    }
    frame->device_ptr = reinterpret_cast<uint64_t>(slot.rgb);
    frame->bytes = static_cast<uint64_t>(
        static_cast<size_t>(decoder->output_width) * decoder->output_height * 3 * sizeof(float)
    );
    frame->width = decoder->output_width;
    frame->height = decoder->output_height;
    frame->pitch_bytes = decoder->output_width * 3 * static_cast<int>(sizeof(float));
    frame->slot = slot.surface_id;
    frame->software_format = decoder->software_format;
    frame->sequence = sequence;
    if (sequence == 0 && !decoder->first_frame_reported) {
        decoder->first_frame_reported = true;
        bridge_marker(
            "FIRST_FRAME_OK",
            "decoder=" + decoder->decoder_name +
            " mode=" + (decoder->source_mode == kHdrMode ? std::string("HDR") : std::string("OM")) +
            " sequence=0 software_format=" + std::to_string(decoder->software_format) +
            " device_ptr=" + std::to_string(reinterpret_cast<uintptr_t>(slot.rgb))
        );
    }
    copy_error(decoder, error_buffer, error_capacity);
    return 1;
}

V5_GPU_API int v5_decoder_release(
    V5Decoder *decoder,
    int slot_index,
    uint64_t sequence,
    uint64_t cuda_stream,
    char *error_buffer,
    int error_capacity
) {
    if (decoder == nullptr || slot_index < 0 || static_cast<size_t>(slot_index) >= decoder->slots.size()) {
        if (error_buffer != nullptr && error_capacity > 0) {
            std::snprintf(error_buffer, static_cast<size_t>(error_capacity), "invalid decoder slot");
        }
        return -1;
    }
    Slot &slot = decoder->slots[static_cast<size_t>(slot_index)];
    uint64_t released_sequence = 0;
    {
        std::lock_guard<std::mutex> lock(decoder->lifecycle_mutex);
        if (slot.surface_id != slot_index) {
            set_error(decoder, "decoder surface handle does not match ring slot");
            copy_error(decoder, error_buffer, error_capacity);
            return -1;
        }
        if (slot.sequence != sequence) {
            set_error(
                decoder,
                "decoder frame ownership sequence does not match ring slot"
            );
            copy_error(decoder, error_buffer, error_capacity);
            return -1;
        }
        if (slot.state != V5_SLOT_CONSUMER_OWNED) {
            set_error(decoder, "decoder slot is not owned by the consumer");
            copy_error(decoder, error_buffer, error_capacity);
            return -1;
        }
        const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
        if (!cuda_ok(decoder, cudaEventRecord(slot.consumer_done, stream), "cudaEventRecord consumer_done")) {
            copy_error(decoder, error_buffer, error_capacity);
            return -1;
        }
        released_sequence = slot.sequence;
        slot.state = V5_SLOT_GPU_PENDING;
    }
    bridge_marker(
        "NVDEC_SURFACE_RELEASED",
        "decoder=" + decoder->decoder_name +
        " surface_id=" + std::to_string(slot_index) +
        " slot=" + std::to_string(slot_index) +
        " sequence=" + std::to_string(released_sequence) +
        " state=GPU_PENDING"
    );
    emit_surface_counters(decoder, "release");
    copy_error(decoder, error_buffer, error_capacity);
    return 0;
}

V5_GPU_API int v5_decoder_width(const V5Decoder *decoder) {
    return decoder != nullptr ? decoder->output_width : 0;
}

V5_GPU_API int v5_decoder_height(const V5Decoder *decoder) {
    return decoder != nullptr ? decoder->output_height : 0;
}

V5_GPU_API int v5_decoder_source_width(const V5Decoder *decoder) {
    return decoder != nullptr ? decoder->width : 0;
}

V5_GPU_API int v5_decoder_source_height(const V5Decoder *decoder) {
    return decoder != nullptr ? decoder->height : 0;
}

V5_GPU_API int v5_decoder_resample_active(const V5Decoder *decoder) {
    return (decoder != nullptr && decoder->resample_active) ? 1 : 0;
}

/* Request the emitted working-RGB geometry for an HDR source. Requesting the
 * decoded size is an explicit no-op: no resampling kernel is selected and no
 * buffer is reallocated, so scale-1.0 material keeps its exact previous path.
 * Must be called before the first v5_decoder_next. */
V5_GPU_API int v5_decoder_set_target_size(
    V5Decoder *decoder,
    int width,
    int height,
    char *error_buffer,
    int error_capacity
) {
    if (decoder == nullptr) {
        return -1;
    }
    if (decoder->next_sequence != 0) {
        set_error(decoder, "v5_decoder_set_target_size must be called before the first frame");
        copy_error(decoder, error_buffer, error_capacity);
        return -1;
    }
    if (width <= 0 || height <= 0) {
        set_error(decoder, "target size must be positive");
        copy_error(decoder, error_buffer, error_capacity);
        return -1;
    }
    if (width == decoder->width && height == decoder->height) {
        decoder->resample_active = false;
        decoder->output_width = decoder->width;
        decoder->output_height = decoder->height;
        bridge_marker(
            "RESAMPLE_NOOP",
            "decoder=" + decoder->decoder_name +
            " geometry=" + std::to_string(width) + "x" + std::to_string(height) +
            " resample_active=0"
        );
        return 0;
    }
    if (decoder->source_mode != kHdrMode) {
        set_error(decoder, "only the HDR source supports pre-PQ resampling");
        copy_error(decoder, error_buffer, error_capacity);
        return -1;
    }
    if (decoder->width < 2 || decoder->height < 2) {
        set_error(decoder, "source is too small to interpolate");
        copy_error(decoder, error_buffer, error_capacity);
        return -1;
    }
    if (decoder->hdr_chroma_filter == nullptr || decoder->hdr_chroma_filter_pos == nullptr ||
        decoder->hdr_chroma_filter_size <= 0) {
        set_error(decoder, "10-bit vertical chroma filter is not initialized");
        copy_error(decoder, error_buffer, error_capacity);
        return -1;
    }
    const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3 * sizeof(float);
    for (size_t index = 0; index < decoder->slots.size(); ++index) {
        Slot &slot = decoder->slots[index];
        if (slot.state != V5_SLOT_FREE) {
            set_error(decoder, "cannot resize working RGB while a slot is in use");
            copy_error(decoder, error_buffer, error_capacity);
            return -1;
        }
        if (slot.rgb != nullptr) {
            cudaFree(slot.rgb);
            slot.rgb = nullptr;
        }
        if (!cuda_ok(
                decoder,
                cudaMalloc(reinterpret_cast<void **>(&slot.rgb), bytes),
                "cudaMalloc resampled working RGB"
            )) {
            copy_error(decoder, error_buffer, error_capacity);
            return -1;
        }
    }
    decoder->output_width = width;
    decoder->output_height = height;
    decoder->resample_active = true;
    bridge_marker(
        "RESAMPLE_ACTIVE",
        "decoder=" + decoder->decoder_name +
        " source=" + std::to_string(decoder->width) + "x" + std::to_string(decoder->height) +
        " target=" + std::to_string(width) + "x" + std::to_string(height) +
        " interpolation=INTER_LINEAR domain=pre_pq_rgb48_normalized"
    );
    return 0;
}

V5_GPU_API int v5_decoder_software_format(const V5Decoder *decoder) {
    return decoder != nullptr ? decoder->software_format : AV_PIX_FMT_NONE;
}

V5_GPU_API uint64_t v5_decoder_frame_bytes(const V5Decoder *decoder) {
    if (decoder == nullptr) {
        return 0;
    }
    return static_cast<uint64_t>(
        static_cast<size_t>(decoder->output_width) * decoder->output_height * 3 * sizeof(float)
    );
}

V5_GPU_API const char *v5_decoder_last_error(const V5Decoder *decoder) {
    return decoder != nullptr ? decoder->last_error.c_str() : "null decoder";
}

V5_GPU_API void v5_decoder_close(V5Decoder *decoder) {
    if (decoder == nullptr) {
        return;
    }
    bool forced_device_sync = false;
    for (size_t index = 0; index < decoder->slots.size(); ++index) {
        Slot &slot = decoder->slots[index];
        int state = V5_SLOT_FREE;
        cudaEvent_t consumer_done = nullptr;
        {
            std::lock_guard<std::mutex> lock(decoder->lifecycle_mutex);
            state = slot.state;
            consumer_done = slot.consumer_done;
        }
        if (state == V5_SLOT_GPU_PENDING && consumer_done != nullptr) {
            cudaEventSynchronize(consumer_done);
            std::lock_guard<std::mutex> lock(decoder->lifecycle_mutex);
            if (slot.state == V5_SLOT_GPU_PENDING && slot.consumer_done == consumer_done) {
                transition_slot_to_free_locked(decoder, slot);
            }
        } else if (state == V5_SLOT_CONSUMER_OWNED) {
            // Defensive close-only fallback for a caller that failed before
            // v5_decoder_release(). Normal Phase 2 release uses consumer_done.
            if (!forced_device_sync) {
                cudaDeviceSynchronize();
                forced_device_sync = true;
            }
            std::lock_guard<std::mutex> lock(decoder->lifecycle_mutex);
            if (slot.state == V5_SLOT_CONSUMER_OWNED) {
                transition_slot_to_free_locked(decoder, slot);
            }
        } else if (state == V5_SLOT_DECODER_OWNED) {
            std::lock_guard<std::mutex> lock(decoder->lifecycle_mutex);
            if (slot.state == V5_SLOT_DECODER_OWNED) {
                transition_slot_to_free_locked(decoder, slot);
            }
        }
    }
    emit_surface_counters(decoder, "close");
    for (Slot &slot : decoder->slots) {
        // Normally transition_slot_to_free_locked already released these. Keep
        // cleanup idempotent for partial-open/error paths as well.
        av_buffer_unref(&slot.surface_ref);
        if (slot.frame != nullptr) {
            av_frame_free(&slot.frame);
        }
        if (slot.source_ready != nullptr) {
            cudaEventDestroy(slot.source_ready);
        }
        if (slot.consumer_done != nullptr) {
            cudaEventDestroy(slot.consumer_done);
        }
        if (slot.rgb != nullptr) {
            cudaFree(slot.rgb);
        }
    }
    decoder->slots.clear();
    if (decoder->hdr_chroma_filter != nullptr) {
        cudaFree(decoder->hdr_chroma_filter);
    }
    if (decoder->hdr_chroma_filter_pos != nullptr) {
        cudaFree(decoder->hdr_chroma_filter_pos);
    }
    if (decoder->pq_eotf_lut != nullptr) {
        cudaFree(decoder->pq_eotf_lut);
        decoder->pq_eotf_lut = nullptr;
    }
    if (decoder->bt1886_lut16 != nullptr) {
        cudaFree(decoder->bt1886_lut16);
        decoder->bt1886_lut16 = nullptr;
    }
    if (decoder->bt1886_lut != nullptr) {
        cudaFree(decoder->bt1886_lut);
        decoder->bt1886_lut = nullptr;
    }
    if (decoder->decoded != nullptr) {
        av_frame_free(&decoder->decoded);
    }
    if (decoder->packet != nullptr) {
        av_packet_free(&decoder->packet);
    }
    if (decoder->codec != nullptr) {
        avcodec_free_context(&decoder->codec);
    }
    if (decoder->hw_device != nullptr) {
        av_buffer_unref(&decoder->hw_device);
    }
    if (decoder->format != nullptr) {
        avformat_close_input(&decoder->format);
    }
    delete decoder;
}

V5_GPU_API V5Encoder *v5_encoder_open(
    const char *path,
    int width,
    int height,
    int fps_num,
    int fps_den,
    int device_index,
    int qp,
    char *error_buffer,
    int error_capacity
) {
    V5Encoder *encoder = new V5Encoder();
    if (!open_encoder(encoder, path, width, height, fps_num, fps_den, device_index, qp)) {
        encoder_copy_error(encoder, error_buffer, error_capacity);
        encoder_cleanup(encoder, false);
        delete encoder;
        return nullptr;
    }
    encoder_copy_error(encoder, error_buffer, error_capacity);
    return encoder;
}

V5_GPU_API int v5_encoder_write_p010(
    V5Encoder *encoder,
    uint64_t y_device_ptr,
    int y_pitch_bytes,
    uint64_t uv_device_ptr,
    int uv_pitch_bytes,
    int64_t pts,
    uint64_t cuda_stream,
    char *error_buffer,
    int error_capacity
) {
    if (encoder == nullptr || encoder->codec == nullptr || encoder->packet == nullptr ||
        y_device_ptr == 0 || uv_device_ptr == 0 || y_pitch_bytes < encoder->width * 2 ||
        uv_pitch_bytes < encoder->width * 2) {
        if (encoder != nullptr) {
            encoder_set_error(encoder, "invalid GPU P010 frame or closed encoder");
            encoder_copy_error(encoder, error_buffer, error_capacity);
        }
        return -1;
    }
    AVFrame *frame = av_frame_alloc();
    if (frame == nullptr) {
        encoder_set_error(encoder, "av_frame_alloc(P010) failed");
        encoder_copy_error(encoder, error_buffer, error_capacity);
        return -1;
    }
    frame->format = AV_PIX_FMT_CUDA;
    frame->width = encoder->width;
    frame->height = encoder->height;
    int status = av_hwframe_get_buffer(encoder->hw_frames, frame, 0);
    if (status < 0) {
        encoder_set_error(encoder, "av_hwframe_get_buffer(P010): " + ffmpeg_error(status));
        av_frame_free(&frame);
        encoder_copy_error(encoder, error_buffer, error_capacity);
        return -1;
    }
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
    status = static_cast<int>(cudaMemcpy2DAsync(
        frame->data[0],
        static_cast<size_t>(frame->linesize[0]),
        reinterpret_cast<const void *>(static_cast<uintptr_t>(y_device_ptr)),
        static_cast<size_t>(y_pitch_bytes),
        static_cast<size_t>(encoder->width) * 2,
        static_cast<size_t>(encoder->height),
        cudaMemcpyDeviceToDevice,
        stream
    ));
    if (status != static_cast<int>(cudaSuccess)) {
        encoder_set_error(encoder, "cudaMemcpy2DAsync(Y): " + std::string(cudaGetErrorString(static_cast<cudaError_t>(status))));
        av_frame_free(&frame);
        encoder_copy_error(encoder, error_buffer, error_capacity);
        return -1;
    }
    status = static_cast<int>(cudaMemcpy2DAsync(
        frame->data[1],
        static_cast<size_t>(frame->linesize[1]),
        reinterpret_cast<const void *>(static_cast<uintptr_t>(uv_device_ptr)),
        static_cast<size_t>(uv_pitch_bytes),
        static_cast<size_t>(encoder->width) * 2,
        static_cast<size_t>(encoder->height / 2),
        cudaMemcpyDeviceToDevice,
        stream
    ));
    if (status != static_cast<int>(cudaSuccess)) {
        encoder_set_error(encoder, "cudaMemcpy2DAsync(UV): " + std::string(cudaGetErrorString(static_cast<cudaError_t>(status))));
        av_frame_free(&frame);
        encoder_copy_error(encoder, error_buffer, error_capacity);
        return -1;
    }
    if (!encoder_cuda_ok(encoder, cudaStreamSynchronize(stream), "cudaStreamSynchronize(P010 copy)")) {
        av_frame_free(&frame);
        encoder_copy_error(encoder, error_buffer, error_capacity);
        return -1;
    }
    if (!encoder->output_surface_reported) {
        encoder->output_surface_reported = true;
        bridge_marker(
            "OUTPUT_SURFACE_OK",
            "path=" + encoder->path +
            " geometry=" + std::to_string(encoder->width) + "x" + std::to_string(encoder->height) +
            " format=P010LE hw_format=CUDA"
        );
    }
    frame->pts = pts;
    status = avcodec_send_frame(encoder->codec, frame);
    if (status == AVERROR(EAGAIN)) {
        if (!encoder_receive_packets(encoder)) {
            av_frame_free(&frame);
            encoder_copy_error(encoder, error_buffer, error_capacity);
            return -1;
        }
        status = avcodec_send_frame(encoder->codec, frame);
    }
    av_frame_free(&frame);
    if (status < 0) {
        encoder_set_error(encoder, "avcodec_send_frame(P010): " + ffmpeg_error(status));
        encoder_copy_error(encoder, error_buffer, error_capacity);
        return -1;
    }
    if (!encoder_receive_packets(encoder)) {
        encoder_copy_error(encoder, error_buffer, error_capacity);
        return -1;
    }
    encoder_copy_error(encoder, error_buffer, error_capacity);
    return 0;
}

V5_GPU_API int v5_encoder_close(
    V5Encoder *encoder,
    char *error_buffer,
    int error_capacity
) {
    if (encoder == nullptr) {
        if (error_buffer != nullptr && error_capacity > 0) {
            std::snprintf(error_buffer, static_cast<size_t>(error_capacity), "null encoder");
        }
        return -1;
    }
    bool success = true;
    if (encoder->codec != nullptr) {
        int status = avcodec_send_frame(encoder->codec, nullptr);
        if (status < 0 && status != AVERROR_EOF) {
            encoder_set_error(encoder, "avcodec_send_frame(EOF): " + ffmpeg_error(status));
            success = false;
        } else if (success && !encoder_receive_packets(encoder)) {
            success = false;
        }
    }
    if (encoder->format != nullptr && encoder->header_written && !encoder->trailer_written) {
        const int status = av_write_trailer(encoder->format);
        if (status < 0) {
            encoder_set_error(encoder, "av_write_trailer: " + ffmpeg_error(status));
            success = false;
        }
        encoder->trailer_written = true;
    }
    encoder_copy_error(encoder, error_buffer, error_capacity);
    encoder_cleanup(encoder, false);
    delete encoder;
    return success ? 0 : -1;
}

V5_GPU_API void v5_encoder_abort(V5Encoder *encoder) {
    if (encoder == nullptr) {
        return;
    }
    encoder_cleanup(encoder, false);
    delete encoder;
}

}  // extern "C"
