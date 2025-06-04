#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SIFT_MAX_DIMENSION                  (16384)
#define SIFT_MAX_ATTEMPTS_UNTIL_CONVERGENCE (5)

/* === Core Data Structures === */

/**
 * @brief Grayscale image container for SIFT processing
 * @details Row-major layout: data[y * width + x]
 */
typedef struct {
  uint32_t width;  ///< Image width in pixels (>0)
  uint32_t height; ///< Image height in pixels (>0)
  float *data;     ///< Contiguous normalized pixel data [0,1]
} sift_image_t;

/**
 * @brief SIFT detector configuration parameters
 * @details Values based on Lowe's 2004 paper defaults
 */
typedef struct {
  float sigma;              ///< Base Gaussian blur (σ) for input image at octave 0
  float contrast_threshold; ///< Filters weak features in low-contrast regions (effective threshold = value / octave_layers)
  float edge_threshold;     ///< Filters edge-like features using Hessian eigenvalue ratio
  uint32_t octave_layers;   ///< Scales per octave (k-value in Lowe's paper)
  uint32_t border_width;    ///< Width of border in which to ignore keypoints
} sift_detector_t;

/**
 * @brief SIFT keypoint descriptor container
 * @details Contains both geometric attributes and feature descriptor
 */
typedef struct {
  float x;                 ///< X-coordinate in original image space (subpixel precision)
  float y;                 ///< Y-coordinate in original image space (subpixel precision)
  float size;              ///< Keypoint diameter in original image space
  float angle;             ///< Computed orientation of the keypoint
  float response;          ///< Keypoint strength (absolute Hessian value)
  uint32_t octave;         ///< Octave index where keypoint was detected
  uint8_t descriptor[128]; ///< Normalized 128-element SIFT descriptor (L2-normalized)
} sift_keypoint_t;

/* === Image Operations === */

/**
 * @brief Creates empty image container with allocated memory
 * @param width Image width (>0)
 * @param height Image height (>0)
 * @return Initialized image or NULL on allocation failure
 * @post Caller must call sift_image_destroy()
 */
sift_image_t *sift_image_create(uint32_t width, uint32_t height);

/**
 * @brief Releases image resources
 * @param image Double pointer to image (set to NULL after destruction)
 */
void sift_image_destroy(sift_image_t **image);

/**
 * @brief Safe pixel access with bounds checking
 * @param image Valid image pointer
 * @param x X coordinate [0,width)
 * @param y Y coordinate [0,height)
 * @return Pixel value or 0.0f for invalid coordinates
 */
float sift_image_get_pixel(const sift_image_t *image, uint32_t x, uint32_t y);

/**
 * @brief Safe pixel modification
 * @param image Valid image pointer
 * @param x X coordinate [0,width)
 * @param y Y coordinate [0,height)
 * @param value New pixel value
 * @note Silently ignores out-of-bounds writes
 */
void sift_image_set_pixel(sift_image_t *image, uint32_t x, uint32_t y, float value);

/**
 * @brief Creates SIFT-compatible float image from 8-bit input data
 * @param data Pointer to raw pixel buffer (must be non-NULL)
 * @param width Image width in pixels (>0)
 * @param height Image height in pixels (>0)
 * @param channels Number of color channels:
 *                 - 1: Grayscale (direct copy with 1/255 normalization)
 *                 - 3: RGB (auto-converts to weighted average)
 *                 - 4: RGBA (the same as 3, but ignores alpha channel)
 * @return New sift_image_t instance or NULL on:
 *         - Invalid parameters
 *         - Memory allocation failure
 *         - Unsupported channel count
 * @note For RGB inputs, uses standard weighted average formula:
 *       Y = 0.299*R + 0.587*G + 0.114*B
 * @warning The input buffer must outlive this function call (no internal copy)
 */
sift_image_t *sift_image_create_from_u8(const uint8_t *data, uint32_t width, uint32_t height, uint8_t channels);

/* === SIFT Detector === */

/**
 * @brief Creates a SIFT feature detector with specified parameters
 * @return Initialized SIFT detector or NULL on failure
 * @note Caller takes ownership of the returned pointer and must destroy it with sift_detector_destroy()
 * @details Default parameters (Lowe, 2004):
 *          - sigma: 1.6
 *          - contrast threshold: 0.03
 *          - edge threshold: 10.0
 *          - octave layers: 3
 *          - border width: 5
 * @example
 * // Create detector with standard parameters
 * sift_detector_t *det = sift_detector_create();
 * if (!det) {
 *   // handle error
 * }
 */
sift_detector_t *sift_detector_create();

/**
 * @brief Safely destroys SIFT detector
 * @param detector Double pointer to detector object (set to NULL after destruction)
 * @example
 * sift_detector_t *det = sift_detector_create(...);
 * // ... usage ...
 * sift_detector_destroy(&det);
 */
void sift_detector_destroy(sift_detector_t **detector);

uint32_t sift_detect_and_compute(const sift_detector_t *detector, const sift_image_t *image, sift_keypoint_t **keypoints);
void sift_keypoints_destroy(sift_keypoint_t **keypoints);

#ifdef __cplusplus
}
#endif

