#pragma once

#include <stdint.h>

#ifdef _WIN32
#define V5_GPU_API __declspec(dllexport)
#else
#define V5_GPU_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct V5Decoder V5Decoder;
typedef struct V5Encoder V5Encoder;

typedef struct V5GpuFrame {
    uint64_t device_ptr;
    uint64_t bytes;
    int width;
    int height;
    int pitch_bytes;
    int slot;
    int software_format;
    uint64_t sequence;
} V5GpuFrame;

/* source_mode: 0 = HDR PQ/BT.2020 P010LE, 1 = SDR BT.1886/BT.709 P010LE or NV12 */
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
);

/* Returns 1 for a frame, 0 for clean EOF, -1 for an error. */
V5_GPU_API int v5_decoder_next(
    V5Decoder *decoder,
    uint64_t cuda_stream,
    V5GpuFrame *frame,
    char *error_buffer,
    int error_capacity
);

/* The caller records all consumer work on cuda_stream before releasing the
 * exact frame ownership represented by (slot, sequence). */
V5_GPU_API int v5_decoder_release(
    V5Decoder *decoder,
    int slot,
    uint64_t sequence,
    uint64_t cuda_stream,
    char *error_buffer,
    int error_capacity
);

/* Emitted working-RGB geometry. It differs from the decoded geometry only
 * when a native target-size resampler is requested. The production HDR
 * resampler operates before PQ on the GPU; a request equal to the decoded
 * geometry remains an explicit no-op. */
V5_GPU_API int v5_decoder_width(const V5Decoder *decoder);
V5_GPU_API int v5_decoder_height(const V5Decoder *decoder);
V5_GPU_API int v5_decoder_source_width(const V5Decoder *decoder);
V5_GPU_API int v5_decoder_source_height(const V5Decoder *decoder);
V5_GPU_API int v5_decoder_resample_active(const V5Decoder *decoder);

/* Optional target-size request. Requesting the decoded size is an explicit
 * no-op. For the production HDR path, a different size resamples the
 * normalized pre-PQ RGB48 code plane with INTER_LINEAR on the GPU before the
 * PQ transfer function is applied, which is the native equivalent of
 * openmatte_hdr.decode_hdr. Must be called before the first frame. */
V5_GPU_API int v5_decoder_set_target_size(
    V5Decoder *decoder,
    int width,
    int height,
    char *error_buffer,
    int error_capacity
);
V5_GPU_API int v5_decoder_software_format(const V5Decoder *decoder);
V5_GPU_API uint64_t v5_decoder_frame_bytes(const V5Decoder *decoder);
V5_GPU_API const char *v5_decoder_last_error(const V5Decoder *decoder);
V5_GPU_API void v5_decoder_close(V5Decoder *decoder);

/* GPU P010 -> NVENC/Matroska encoder. The input planes are CUDA device
 * pointers; encoded bitstream retrieval is internal to the bridge. */
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
);

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
);

V5_GPU_API int v5_encoder_close(
    V5Encoder *encoder,
    char *error_buffer,
    int error_capacity
);

V5_GPU_API void v5_encoder_abort(V5Encoder *encoder);

#ifdef __cplusplus
}
#endif
