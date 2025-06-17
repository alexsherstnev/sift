#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SIFT_MAX_DIMENSION                  (16384)
#define SIFT_MAX_ATTEMPTS_UNTIL_CONVERGENCE (5)
#define SIFT_HISTOGRAM_BINS                 (36)
#define SIFT_ORIENTATION_SIGMA              (1.5f)
#define SIFT_ORIENTATION_RADIUS             (3 * SIFT_ORIENTATION_SIGMA)
#define SIFT_DESCRIPTOR_HISTOGRAMS          (4)
#define SIFT_DESCRIPTOR_BINS                (8)
#define SIFT_DESCRIPTOR_SCALE_FACTOR        (3.0f)
#define SIFT_DESCRIPTOR_WINDOW_SIZE         (16.0f)
#define SIFT_DESCRIPTOR_MAG_THRESH          (0.2f)
#define SIFT_DESCRIPTOR_NORM_FACTOR         (512.0f)

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
  uint32_t layer;          ///< Layer index where keypoint was detected
  uint8_t descriptor[128]; ///< Normalized 128-element SIFT descriptor (L2-normalized)
} sift_keypoint_t;

/**
 * @brief SIFT feature match pair
 * @details Contains indices of matched keypoints from two sets and their distance measure
 */
typedef struct {
  uint32_t from_idx; ///< Index of the keypoint in from set (first image)
  uint32_t to_idx;   ///< Index of the keypoint in to set (second image)
  float distance;    ///< L2 distance between matched descriptors
} sift_match_t;

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

/**
 * @brief Performs full SIFT feature detection and descriptor computation
 * @param detector Initialized SIFT detector (must be non-NULL)
 * @param image Input image to process (must be non-NULL)
 * @param[out] keypoints Output array of detected keypoints (will be allocated)
 * @return Number of detected keypoints or 0 on:
 *         - Invalid parameters
 *         - Detection failure
 *         - Memory allocation error
 * @note The caller takes ownership of the returned keypoints and must destroy them
 *       with sift_keypoints_destroy()
 * @warning This function may take significant time for large images
 * @example
 * sift_keypoint_t *keys = NULL;
 * uint32_t count = sift_detect_and_compute(det, img, &keys);
 * if (count > 0) {
 *   // Process keypoints...
 *   sift_keypoints_destroy(&keys);
 * }
 */
uint32_t sift_detect_and_compute(const sift_detector_t *detector, const sift_image_t *image, sift_keypoint_t **keypoints);

/**
 * @brief Safely destroys array of SIFT keypoints
 * @param keypoints Double pointer to keypoints array (set to NULL after destruction)
 * @note Handles NULL input gracefully
 * @example
 * sift_keypoint_t *keys = ...;
 * sift_keypoints_destroy(&keys);
 * assert(keys == NULL);
 */
void sift_keypoints_destroy(sift_keypoint_t **keypoints);

/**
 * @brief Finds matches between two sets of SIFT keypoints using descriptor similarity
 * @param keys1 First set of keypoints (query features, must be non-NULL)
 * @param count1 Number of keypoints in first set (must be >0)
 * @param keys2 Second set of keypoints (train features, must be non-NULL)
 * @param count2 Number of keypoints in second set (must be >0)
 * @param[out] matches Output array of matched pairs (will be allocated)
 * @param match_threshold Maximum distance ratio between best/second-best matches (0.6-0.8 recommended)
 * @param mutual_check If true, requires matches to be mutual (both-way best matches)
 * @param sort_by_distance If true, sorts matches by ascending descriptor distance
 * @return Number of valid matches found or 0 on:
 *         - Invalid parameters
 *         - No matches passing threshold
 *         - Memory allocation error
 * @note Matching algorithm details:
 *       1. For each keypoint in keys1, finds two nearest neighbors in keys2
 *       2. Applies Lowe's ratio test (best_dist/second_best_dist < threshold)
 *       3. Optionally verifies mutual best match (if mutual_check=true)
 *       4. Optionally sorts results by match quality (if sort_by_distance=true)
 * @warning Caller must free matches array with sift_matches_destroy()
 * @see Lowe, D.G. "Distinctive Image Features from Scale-Invariant Keypoints", IJCV 2004
 * @example
 * sift_keypoint_t *keys1, *keys2;
 * uint32_t count1, count2;
 * sift_match_t *matches = NULL;
 * 
 * uint32_t num_matches = sift_find_matches(
 *     keys1, count1,
 *     keys2, count2,
 *     &matches,
 *     0.7f,    // match_threshold
 *     true,    // mutual_check
 *     true     // sort_by_distance
 * );
 * 
 * if (num_matches > 0) {
 *     for (uint32_t i = 0; i < num_matches; i++) {
 *         printf("Match %d: query_idx=%u, train_idx=%u, distance=%.2f\n",
 *                i, matches[i].query_idx, matches[i].train_idx, matches[i].distance);
 *     }
 *     free(matches);
 * }
 */
uint32_t sift_find_matches(const sift_keypoint_t *keys1, uint32_t count1,
                           const sift_keypoint_t *keys2, uint32_t count2,
                           sift_match_t **matches, float match_threshold,
                           bool mutual_check, bool sort_by_distance);

/**
 * @brief Safely deallocates memory for SIFT feature matches
 * @param[in,out] matches Double pointer to matches array (will be set to NULL after destruction)
 * @example
 * // Safe usage pattern:
 * sift_match_t *matches = NULL;
 * uint32_t count = sift_find_matches(..., &matches);
 * 
 * // Process matches...
 * 
 * sift_matches_destroy(&matches); // matches set to NULL
 * assert(matches == NULL);
 * 
 * // Safe to call again:
 * sift_matches_destroy(&matches); // No effect
 */
void sift_matches_destroy(sift_match_t **matches);

#ifdef __cplusplus
}
#endif

