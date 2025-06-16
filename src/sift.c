#include "sift/sift.h"
#include <float.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* === Private Helpers === */

#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define MAX(x,y) ((x) > (y) ? (x) : (y))

typedef struct {
  float sigma;
  uint32_t size;
  float *weights;
} gaussian_kernel_t;

typedef struct {
  gaussian_kernel_t **data;
  uint32_t count;
} gaussian_kernel_array_t;

typedef struct {
  sift_image_t **data;
  uint32_t count;
} image_array_t;

typedef struct {
  float sigma;
  float contrast_threshold;
  float edge_threshold;
  uint32_t octave_layers;
  uint32_t border_width;
  uint32_t octaves;
  sift_image_t *base_image;
  gaussian_kernel_array_t kernels;
  image_array_t gaussian_pyramid;
  image_array_t dog_pyramid;
} detector_context_t;

static inline int32_t reflect_border(int32_t idx, int32_t boundary) {
  if (idx < 0)         { return -idx;                     }
  if (idx >= boundary) { return 2 * (boundary - 1) - idx; }
  return idx;
}

static void gaussian_kernel_horizontal_convolve(const gaussian_kernel_t *kernel, const sift_image_t *src, sift_image_t *dst) {
  const int32_t radius = kernel->size / 2;
  for (uint32_t y = 0; y < src->height; ++y) {
    for (uint32_t x = 0; x < src->width; ++x) {
      float sum = 0.0f;
      for (int32_t d = -radius; d <= radius; ++d) {
        const int32_t k = reflect_border(x + d, src->width);
        sum += sift_image_get_pixel(src, k, y) * kernel->weights[d + radius];
      }
      sift_image_set_pixel(dst, x, y, sum);
    }
  }
}

static void gaussian_kernel_vertical_convolve(const gaussian_kernel_t *kernel, const sift_image_t *src, sift_image_t *dst) {
  const int32_t radius = kernel->size / 2;
  for (uint32_t y = 0; y < src->height; ++y) {
    for (uint32_t x = 0; x < src->width; ++x) {
      float sum = 0.0f;
      for (int32_t d = -radius; d <= radius; ++d) {
        const int32_t k = reflect_border(y + d, src->height);
        sum += sift_image_get_pixel(src, x, k) * kernel->weights[d + radius];
      }
      sift_image_set_pixel(dst, x, y, sum);
    }
  }
}

static void gaussian_kernel_destroy(gaussian_kernel_t **kernel) {
  if (!kernel || !(*kernel)) { return; }
  if ((*kernel)->weights) {
    free((*kernel)->weights);
    (*kernel)->weights = NULL;
  }
  free(*kernel);
  *kernel = NULL;
}

static gaussian_kernel_t *gaussian_kernel_create(float sigma) {
  if (sigma <= 0.0f) { return NULL; }
  
  // Create kernel container
  gaussian_kernel_t *kernel = calloc(1, sizeof(gaussian_kernel_t));
  if (!kernel) { return NULL; }

  // Calculate kernel size
  kernel->size = ((uint32_t)(6.0f * sigma + 1)) | 1;

  // Allocate memory for kernel weights
  kernel->weights = calloc(kernel->size, sizeof(float));
  if (!kernel->weights) {
    gaussian_kernel_destroy(&kernel);
    return NULL;
  }

  // Fill kernel weights
  const int32_t radius = kernel->size / 2;
  const float inv_2_sqr_sigma = 1.0f / (2.0f * sigma * sigma);
  float sum = 0.0f;
  for (int32_t i = -radius; i <= radius; ++i) {
    const float x = (float)i;
    const float weight = expf(-(x * x) * inv_2_sqr_sigma);
    kernel->weights[i + radius] = weight;
    sum += weight;
  }

  // Normalization
  const float inv_sum = 1.0f / sum;
  for (uint32_t i = 0; i < kernel->size; ++i) { kernel->weights[i] *= inv_sum; }

  // Store kernel meta
  kernel->sigma = sigma;

  return kernel;
}

static void gaussian_kernel_apply(const gaussian_kernel_t *kernel, const sift_image_t *src, sift_image_t *dst) {
  if (!kernel || !src || !dst || src->width != dst->width || src->height != dst->height) { return; }

  sift_image_t *tmp = sift_image_create(src->width, src->height);
  if (!tmp) { return; }

  gaussian_kernel_horizontal_convolve(kernel, src, tmp);
  gaussian_kernel_vertical_convolve(kernel, tmp, dst);

  sift_image_destroy(&tmp);
}

static sift_image_t *image_copy(const sift_image_t *src) {
  // 1. Create image container
  sift_image_t *new_image = sift_image_create(src->width, src->height);
  if (!new_image) { return NULL; }

  // 2. Raw copy data
  memcpy(new_image->data, src->data, src->width * src->height * sizeof(float));

  return new_image;
}

static sift_image_t *image_resize(const sift_image_t *src, uint32_t new_width, uint32_t new_height) {
  // 1. Create image container
  sift_image_t *new_image = sift_image_create(new_width, new_height);
  if (!new_image) { return NULL; }


  // 2. Resize
  const bool is_downscale = (new_width < src->width) || (new_height < src->height);
  const float x_ratio = src->width / (float)new_width;
  const float y_ratio = src->height / (float)new_height;
  for (uint32_t y = 0; y < new_height; ++y) {
    for (uint32_t x = 0; x < new_width; ++x) {
      float value = 0.0f;
      if (is_downscale) {
        // Nearest interpolation
        const uint32_t src_x = MIN((uint32_t)floorf(x * x_ratio), src->width - 1);
        const uint32_t src_y = MIN((uint32_t)floorf(y * y_ratio), src->height - 1);
        value = sift_image_get_pixel(src, src_x, src_y);
      } else {
        // Linear interpolation
        const float src_x = (x + 0.5f) * x_ratio - 0.5f;
        const float src_y = (y + 0.5f) * y_ratio - 0.5f;

        const int32_t x0 = (int32_t)floorf(MAX(0.0f, src_x));
        const int32_t y0 = (int32_t)floorf(MAX(0.0f, src_y));
        const int32_t x1 = MIN((int32_t)src->width - 1, x0 + 1);
        const int32_t y1 = MIN((int32_t)src->height - 1, y0 + 1);

        const float dx = MAX(0.0f, MIN(1.0f, src_x - x0));
        const float dy = MAX(0.0f, MIN(1.0f, src_y - y0));

        const float v00 = sift_image_get_pixel(src, x0, y0);
        const float v10 = sift_image_get_pixel(src, x1, y0);
        const float v01 = sift_image_get_pixel(src, x0, y1);
        const float v11 = sift_image_get_pixel(src, x1, y1);
        value = v00 * (1.0f - dx) * (1.0f - dy) +
                v10 * dx          * (1.0f - dy) +
                v01 * (1.0f - dx) * dy +
                v11 * dx          * dy;
      }
      sift_image_set_pixel(new_image, x, y, value);
    }
  }

  return new_image;
}

static sift_image_t *image_subtract(const sift_image_t *a, const sift_image_t *b) {
  // 1. Create image result container
  sift_image_t *subtract = sift_image_create(a->width, a->height);
  if (!subtract) { return NULL; }

  // 2. Subtract
  for (uint32_t y = 0; y < a->height; ++y) {
    for (uint32_t x = 0; x < a->width; ++x) {
      const float value = sift_image_get_pixel(a, x, y) - sift_image_get_pixel(b, x, y);
      sift_image_set_pixel(subtract, x, y, value);
    }
  }

  return subtract;
}

static void detector_context_destroy(detector_context_t *context) {
  if (context->dog_pyramid.count > 0 && context->dog_pyramid.data && *context->dog_pyramid.data) {
    for (uint32_t i = 0; i < context->dog_pyramid.count; ++i) { sift_image_destroy(&context->dog_pyramid.data[i]); }
    free(*context->dog_pyramid.data);
    *context->dog_pyramid.data = NULL;
    context->dog_pyramid.count = 0;
  }

  if (context->gaussian_pyramid.count > 0 && context->gaussian_pyramid.data && *context->gaussian_pyramid.data) {
    for (uint32_t i = 0; i < context->gaussian_pyramid.count; ++i) { sift_image_destroy(&context->gaussian_pyramid.data[i]); }
    free(*context->gaussian_pyramid.data);
    *context->gaussian_pyramid.data = NULL;
    context->gaussian_pyramid.count = 0;
  }

  if (context->kernels.count > 0 && context->kernels.data && *context->kernels.data) {
    for (uint32_t i = 0; i < context->kernels.count; ++i) { gaussian_kernel_destroy(&context->kernels.data[i]); }
    free(*context->kernels.data);
    *context->kernels.data = NULL;
    context->kernels.count = 0;
  }

  if (context->base_image) { sift_image_destroy(&context->base_image); }
}

static bool create_base_image(const sift_image_t *image, float inital_sigma, detector_context_t *context) {
  // 1. Double size original image
  sift_image_t *doubled_image = image_resize(image, image->width * 2, image->height * 2);
  if (!doubled_image) { return false; }

  // 2. Create initial gaussian kernel
  const float sigma_diff = sqrtf(fmaxf(context->sigma * context->sigma - 4.0f * inital_sigma * inital_sigma, 0.01f));
  gaussian_kernel_t *kernel = gaussian_kernel_create(sigma_diff);
  if (!kernel) {
    sift_image_destroy(&doubled_image);
    return false;
  }

  // 3. Apply kernel and store results
  bool result = false;
  sift_image_t *base_image = sift_image_create(doubled_image->width, doubled_image->height);
  if (base_image) {
    gaussian_kernel_apply(kernel, doubled_image, base_image);
    context->base_image = base_image;
    result = true;
  }

  // 4. Free all resources
  gaussian_kernel_destroy(&kernel);
  sift_image_destroy(&doubled_image);

  return result;
}

static bool create_gaussian_kernels(detector_context_t *context) {
  // 1. Allocate needed memory
  context->kernels.count = context->octave_layers + 3;
  context->kernels.data = calloc(context->kernels.count, sizeof(gaussian_kernel_t *));
  if (!context->kernels.data) { return false; }

  // 2. Create kernels
  const float k = powf(2.0f, 1.0f / context->octave_layers);
  context->kernels.data[0] = gaussian_kernel_create(context->sigma);
  if (!context->kernels.data[0]) { return false; }
  for (uint32_t i = 1; i < context->kernels.count; ++i) {
    const float prev_sigma = powf(k, i - 1) * context->sigma;
    const float total_sigma = k * prev_sigma;
    context->kernels.data[i] = gaussian_kernel_create(sqrtf(total_sigma * total_sigma - prev_sigma * prev_sigma));
    if (!context->kernels.data[i]) { return false; }
  }

  return true;
}

static void calculate_num_octaves(detector_context_t *context) {
  const float min_dimension = context->base_image->width > context->base_image->height ? context->base_image->height
                                                                                       : context->base_image->width;
  const float max_kernel_size = context->kernels.data[context->kernels.count - 1]->size;
  context->octaves = fmaxf(1, log2f(min_dimension / max_kernel_size) + 1);
}

static bool build_gaussian_pyramid(detector_context_t *context) {
  // 1. Allocate needed memory
  context->gaussian_pyramid.count = context->octaves * (context->octave_layers + 3);
  context->gaussian_pyramid.data = calloc(context->gaussian_pyramid.count, sizeof(sift_image_t *));
  if (!context->gaussian_pyramid.data) { return false; }

  // 2. Build pyramid
  for (uint32_t octave = 0; octave < context->octaves; ++octave) {
    for (uint32_t layer = 0; layer < context->octave_layers + 3; ++layer) {
      const uint32_t idx = octave * (context->octave_layers + 3)+ layer;
      if (octave == 0 && layer == 0) {
        // Use base image as first image in pyramid
        context->gaussian_pyramid.data[idx] = image_copy(context->base_image);
      } else if (layer == 0) {
        // First image in octave?
        sift_image_t *prev_image = context->gaussian_pyramid.data[(octave - 1) * (context->octave_layers + 3) + context->octave_layers];
        context->gaussian_pyramid.data[idx] = image_resize(prev_image, prev_image->width / 2, prev_image->height / 2);
      } else {
        // Regular image in octave
        sift_image_t *prev_image = context->gaussian_pyramid.data[octave * (context->octave_layers + 3) + (layer - 1)];
        sift_image_t *new_image = sift_image_create(prev_image->width, prev_image->height);
        if (!new_image) { return false; }
        gaussian_kernel_apply(context->kernels.data[layer], prev_image, new_image);
        context->gaussian_pyramid.data[idx] = new_image;
      }
    }
  }

  return true;
}

static bool build_dog_pyramid(detector_context_t *context) {
  // 1. Allocate needed memory
  context->dog_pyramid.count = context->octaves * (context->octave_layers + 2);
  context->dog_pyramid.data = calloc(context->dog_pyramid.count, sizeof(sift_image_t *));
  if (!context->dog_pyramid.data) { return false; }

  // 2. DoG
  for (uint32_t octave = 0; octave < context->octaves; ++octave) {
    for (uint32_t layer = 0; layer < context->octave_layers + 2; ++layer) {
      const sift_image_t *a = context->gaussian_pyramid.data[octave * (context->octave_layers + 3) + layer + 1];
      const sift_image_t *b = context->gaussian_pyramid.data[octave * (context->octave_layers + 3) + layer];
      context->dog_pyramid.data[octave * (context->octave_layers + 2) + layer] = image_subtract(a, b);
    }
  }

  return true;
}

static bool is_local_scale_space_extremum(const image_array_t *dog_pyramid, uint32_t index, uint32_t x, uint32_t y) {
  const float center = sift_image_get_pixel(dog_pyramid->data[index], x, y);
  for (int32_t dz = -1; dz <= 1; ++dz) {
    const sift_image_t *dog = dog_pyramid->data[index + dz];
    for (int32_t dy = -1; dy <= 1; ++dy) {
      for (int32_t dx = -1; dx <= 1; ++dx) {
        if (dz == 0 && dy == 0 && dx == 0) { continue; }
        const float value = sift_image_get_pixel(dog, x + dx, y + dy);
        if ((center > 0.0f && center <= value) || (center < 0.0f && center >= value)) { return false; }
      }
    }
  }

  return true;
}

static bool adjust_or_discard_keypoint(const detector_context_t *context, uint32_t x, uint32_t y, sift_keypoint_t *keypoint) {
  uint32_t attempt = 0;
  float dx = 0.0f, dy = 0.0f, ds = 0.0f;
  while (attempt++ < SIFT_MAX_ATTEMPTS_UNTIL_CONVERGENCE) {
    const uint32_t idx = keypoint->octave * (context->octave_layers + 2) + keypoint->layer;
    const sift_image_t *curr = context->dog_pyramid.data[idx];
    const sift_image_t *next = context->dog_pyramid.data[idx + 1];
    const sift_image_t *prev = context->dog_pyramid.data[idx - 1];

    // Gradients
    const float gx = 0.5f * (sift_image_get_pixel(curr, x + 1, y) - sift_image_get_pixel(curr, x - 1, y));
    const float gy = 0.5f * (sift_image_get_pixel(curr, x, y + 1) - sift_image_get_pixel(curr, x, y - 1));
    const float gs = 0.5f * (sift_image_get_pixel(next, x, y)     - sift_image_get_pixel(prev, x, y));

    // Hessian (due symmetry calculate only 6 values)
    const float c = sift_image_get_pixel(curr, x, y);
    const float c2 = 2.0f * c;
    const float dxx = sift_image_get_pixel(curr, x + 1, y) + sift_image_get_pixel(curr, x - 1, y) - c2;
    const float dyy = sift_image_get_pixel(curr, x, y + 1) + sift_image_get_pixel(curr, x, y - 1) - c2;
    const float dss = sift_image_get_pixel(next, x, y)     + sift_image_get_pixel(prev, x, y)     - c2;
    const float dxy = 0.25f * (sift_image_get_pixel(curr, x + 1, y + 1) - sift_image_get_pixel(curr, x - 1, y + 1) -
                               sift_image_get_pixel(curr, x + 1, y - 1) + sift_image_get_pixel(curr, x - 1, y - 1));
    const float dxs = 0.25f * (sift_image_get_pixel(next, x + 1, y)     - sift_image_get_pixel(next, x - 1, y)     -
                               sift_image_get_pixel(prev, x + 1, y)     + sift_image_get_pixel(prev, x - 1, y));
    const float dys = 0.25f * (sift_image_get_pixel(next, x, y + 1)     - sift_image_get_pixel(next, x, y - 1)     -
                               sift_image_get_pixel(prev, x, y + 1)     + sift_image_get_pixel(prev, x, y - 1));

    // Inverse Hessian matrix: H * offset = -G
    const float det = dxx * dyy * dss - dxx * dys * dys - dxy * dxy * dss + 2.0f * dxs * dxy * dys - dxs * dyy * dxs;
    if (fabsf(det) < 1e-10f) { return false; }

    const float inv_det = 1.0f / det;
    const float inv_dxx = (dyy * dss - dys * dys) * inv_det;
    const float inv_dyy = (dxx * dss - dxs * dxs) * inv_det;
    const float inv_dss = (dxx * dyy - dxy * dxy) * inv_det;
    const float inv_dxy = (dxy * dss - dys * dxs) * inv_det;
    const float inv_dxs = (dxy * dys - dyy * dxs) * inv_det;
    const float inv_dys = (dxx * dys - dxy * dxs) * inv_det;

    // Calculate offsets
    dx = -(inv_dxx * gx + inv_dxy * gy + inv_dxs * gs);
    dy = -(inv_dxy * gx + inv_dyy * gy + inv_dys * gs);
    ds = -(inv_dxs * gx + inv_dys * gy + inv_dss * gs);

    if(fabsf(dx) < 0.5f && fabsf(dy) < 0.5f && fabsf(ds) < 0.5f ) { break; }

    // Adjust grid x, y, l
    x += roundf(dx);
    y += roundf(dy);
    keypoint->layer += roundf(ds);

    if (keypoint->layer < 1 || keypoint->layer > context->octave_layers)          { return false; }
    if (x < context->border_width || x >= (curr->width - context->border_width))  { return false; }
    if (y < context->border_width || y >= (curr->height - context->border_width)) { return false; }
  }

  if (attempt >= SIFT_MAX_ATTEMPTS_UNTIL_CONVERGENCE) { return false; }

  const uint32_t idx = keypoint->octave * (context->octave_layers + 2) + keypoint->layer;
  const sift_image_t *curr = context->dog_pyramid.data[idx];
  const sift_image_t *next = context->dog_pyramid.data[idx + 1];
  const sift_image_t *prev = context->dog_pyramid.data[idx - 1];

  // Gradients
  const float gx = 0.5f * (sift_image_get_pixel(curr, x + 1, y) - sift_image_get_pixel(curr, x - 1, y));
  const float gy = 0.5f * (sift_image_get_pixel(curr, x, y + 1) - sift_image_get_pixel(curr, x, y - 1));
  const float gs = 0.5f * (sift_image_get_pixel(next, x, y)     - sift_image_get_pixel(prev, x, y));

  // Response
  const float t = gx * dx + gy * dy + gs * ds;
  const float c = sift_image_get_pixel(curr, x, y);
  const float response = c + 0.5f * t;
  if (fabsf(response) < context->contrast_threshold) { return false; }

  const float c2 = 2.0f * c;
  const float dxx = sift_image_get_pixel(curr, x + 1, y) + sift_image_get_pixel(curr, x - 1, y) - c2;
  const float dyy = sift_image_get_pixel(curr, x, y + 1) + sift_image_get_pixel(curr, x, y - 1) - c2;
  const float dxy = 0.25f * (sift_image_get_pixel(curr, x + 1, y + 1) - sift_image_get_pixel(curr, x - 1, y + 1) -
                             sift_image_get_pixel(curr, x + 1, y - 1) + sift_image_get_pixel(curr, x - 1, y - 1));
  const float tr = dxx + dyy;
  const float det = dxx * dyy - dxy * dxy;
  if (det <= 0.0f || (tr * tr * context->edge_threshold) >= ((context->edge_threshold + 1) * (context->edge_threshold + 1) * det)) { return false; }

  const float octave_scale = 1 << keypoint->octave;
  keypoint->x = (x + dx) * octave_scale;
  keypoint->y = (y + dy) * octave_scale;
  keypoint->size = context->sigma * powf(2.0f, (keypoint->layer + ds) / (float)context->octave_layers) * octave_scale * 2.0f;
  keypoint->response = fabsf(response);

  return true;
}

static void smooth_orientation_histogram(float *hist) {
  const float kernel[] = {1.0f, 4.0f, 6.0f, 4.0f, 1.0f};
  const int32_t ksize = 5;
  const float norm = 16.0f;

  float temp[SIFT_HISTOGRAM_BINS];
  for (int32_t iter = 0; iter < 6; iter++) {
    for (int32_t i = 0; i < SIFT_HISTOGRAM_BINS; i++) {
      float sum = 0.0f;
      for (int32_t k = 0; k < ksize; k++) {
        int idx = (i + k - ksize/2 + SIFT_HISTOGRAM_BINS) % SIFT_HISTOGRAM_BINS;
        sum += hist[idx] * kernel[k];
      }
      temp[i] = sum / norm;
    }
    memcpy(hist, temp, sizeof(temp));
  }
}

static float calculate_keypoint_orientation_histogram(const detector_context_t *context, sift_keypoint_t *keypoint, float *histogram) {
  // Reset histogram
  memset(histogram, 0, SIFT_HISTOGRAM_BINS * sizeof(float));

  // Calculate scale-dependent parameters
  const float bin_size = SIFT_HISTOGRAM_BINS / 360.0f;
  const float octave_scale = 1.0f / (1 << keypoint->octave);
  const float scale = keypoint->size * octave_scale;
  const int32_t radius = (int32_t)roundf(SIFT_ORIENTATION_RADIUS * scale * 0.5f);
  const float sigma = SIFT_ORIENTATION_SIGMA * scale;
  const float inv_sigma_sq = -0.5f / (sigma * sigma);

  // Get the correct image from pyramid
  const sift_image_t *image = context->gaussian_pyramid.data[keypoint->octave * (context->octave_layers + 3) + keypoint->layer];

  // Convert keypoint coordinates to this octave
  const int32_t x_center = (int32_t)(keypoint->x * octave_scale + 0.5f);
  const int32_t y_center = (int32_t)(keypoint->y * octave_scale + 0.5f);

  // Process neighborhood
  for (int32_t y = -radius; y <= radius; ++y) {
    const int32_t yy = y_center + y;
    if (yy <= 0 || yy >= (int32_t)image->height - 1) { continue; }

    for (int32_t x = -radius; x <= radius; ++x) {
      const int32_t xx = x_center + x;
      if (xx <= 0 || xx >= (int32_t)image->width - 1) { continue; }

      // Calculate gradients
      const float dx = sift_image_get_pixel(image, xx + 1, yy) - sift_image_get_pixel(image, xx - 1, yy);
      const float dy = sift_image_get_pixel(image, xx, yy - 1) - sift_image_get_pixel(image, xx, yy + 1);
      const float mag = sqrtf(dx * dx + dy * dy);

      // Calculate orientation
      const float angle = atan2f(dy, dx) * (180.0f / M_PI);

      // Calculate Gaussian weight
      const float weight = expf((x*x + y*y) * inv_sigma_sq);

      // Add to histogram with linear interpolation
      const float bin = angle * bin_size;
      int32_t bin0 = (int32_t)roundf(bin);
      if (bin0 < 0)                    { bin0 += SIFT_HISTOGRAM_BINS; }
      if (bin0 >= SIFT_HISTOGRAM_BINS) { bin0 -= SIFT_HISTOGRAM_BINS; }
      histogram[bin0] += weight * mag;
    }
  }

  // Smooth histogram (6 iterations as in original SIFT)
  smooth_orientation_histogram(histogram);

  // Find maximum value
  float max_val = 0.0f;
  for (int32_t j = 0; j < SIFT_HISTOGRAM_BINS; ++j) {
    if (histogram[j] > max_val) {
      max_val = histogram[j];
    }
  }

  return max_val;
}

static uint32_t find_scale_space_extrema(const detector_context_t *context, sift_keypoint_t **keypoints) {
  // 1. Initial memory allocation (try to allocate max possible keypoints count)
  uint32_t max_possible_keypoints = 0;
  for (uint32_t octave = 0; octave < context->octaves; ++octave) {
    sift_image_t *image = context->dog_pyramid.data[octave * (context->octave_layers + 2)];
    max_possible_keypoints += (image->width - context->border_width) * (image->height - context->border_width);
  }
  *keypoints = calloc(max_possible_keypoints, sizeof(sift_keypoint_t));
  if (!(*keypoints)) { return 0; }

  // 2. Find keypoints
  uint32_t num_keypoints = 0;
  for (uint32_t octave = 0; octave < context->octaves; ++octave) {
    for (uint32_t layer = 1; layer <= context->octave_layers; ++layer) {
      const uint32_t idx = octave * (context->octave_layers + 2) + layer;

      // Loop over center layer
      for (uint32_t y = context->border_width; y < context->dog_pyramid.data[idx]->height - context->border_width; ++y) {
        for (uint32_t x = context->border_width; x < context->dog_pyramid.data[idx]->width - context->border_width; ++x) {
          if (!is_local_scale_space_extremum(&context->dog_pyramid, idx, x, y)) { continue; }

          sift_keypoint_t potential_keypoint = {
            .x = x,
            .y = y,
            .size = 0.0f,
            .angle = 0.0f,
            .response = 0.0f,
            .octave = octave,
            .layer = layer
          };
          if (adjust_or_discard_keypoint(context, x, y, &potential_keypoint)) {
            float histogram[SIFT_HISTOGRAM_BINS];
            const float orientation_max = calculate_keypoint_orientation_histogram(context, &potential_keypoint, histogram);
            const float orientation_threshold = 0.8f * orientation_max;

            for (int32_t j = 0; j < SIFT_HISTOGRAM_BINS; j++) {
              const int32_t prev_idx = (j - 1 + SIFT_HISTOGRAM_BINS) % SIFT_HISTOGRAM_BINS;
              const int32_t next_idx = (j + 1) % SIFT_HISTOGRAM_BINS;

              // Check if current bin is a local maximum and above threshold
              if (histogram[j] > histogram[prev_idx] && 
                  histogram[j] > histogram[next_idx] &&
                  histogram[j] >= orientation_threshold) {

                // Parabolic interpolation to find sub-bin peak position
                const float numerator = histogram[prev_idx] - histogram[next_idx];
                const float denominator = 2.0f * (histogram[prev_idx] - 2.0f * histogram[j] + histogram[next_idx]);
                const float offset = (fabsf(denominator) > 1e-6f) ? (0.5f * numerator / denominator) : 0.0f;

                // Handle wrap-around for circular histogram
                float peak_bin = j + offset;
                if (peak_bin < 0)                         { peak_bin += SIFT_HISTOGRAM_BINS; }
                else if (peak_bin >= SIFT_HISTOGRAM_BINS) { peak_bin -= SIFT_HISTOGRAM_BINS; }

                // Calcualte angle
                potential_keypoint.angle = 360.0f - (float)(360.0f / SIFT_HISTOGRAM_BINS * peak_bin);
                if (fabsf(potential_keypoint.angle - 360.0f) < FLT_EPSILON) { potential_keypoint.angle = 0.0f; }

                // Store the keypoint
                (*keypoints)[num_keypoints++] = potential_keypoint;
              }
            }
          }
        }
      }
    }
  }
  
  // 3. Re-allocate memory for optimization
  if (num_keypoints > 0) {
    *keypoints = realloc(*keypoints, num_keypoints * sizeof(sift_keypoint_t));
  } else {
    free(*keypoints);
    *keypoints = NULL;
  }

  return num_keypoints;
}

static int keypoint_comparator(const void *a, const void *b) {
  const sift_keypoint_t *kp1 = (const sift_keypoint_t *)a;
  const sift_keypoint_t *kp2 = (const sift_keypoint_t *)b;

  if (kp1->x != kp2->x)               { return kp1->x < kp2->x ? -1 : 1;               }
  if (kp1->y != kp2->y)               { return kp1->y < kp2->y ? -1 : 1;               }
  if (kp1->size != kp2->size)         { return kp1->size > kp2->size ? -1 : 1;         }
  if (kp1->angle != kp2->angle)       { return kp1->angle < kp2->angle ? -1 : 1;       }
  if (kp1->response != kp2->response) { return kp1->response > kp2->response ? -1 : 1; }
  if (kp1->octave != kp2->octave)     { return kp1->octave > kp2->octave ? -1 : 1;     }

  return 0;
}

static uint32_t remove_duplicate_keypoints(sift_keypoint_t **keypoints, uint32_t num_keypoints) {
  if (num_keypoints < 2) { return num_keypoints; }

  // 1. Sort all keypoints
  qsort(*keypoints, num_keypoints, sizeof(sift_keypoint_t), keypoint_comparator);

  // 2. Remove duplicates
  uint32_t i, j;
  for(i = 0, j = 1; j < num_keypoints; ++j) {
    sift_keypoint_t kp1 = (*keypoints)[i];
    sift_keypoint_t kp2 = (*keypoints)[j];
    if(kp1.x != kp2.x || kp1.y != kp2.y ||
       kp1.size != kp2.size || kp1.angle != kp2.angle) {
      (*keypoints)[++i] = (*keypoints)[j];
    }
  }

  // 3. Resize keyponts
  const uint32_t new_num_keypoints = i + 1;
  if (new_num_keypoints != num_keypoints) {
    *keypoints = realloc(*keypoints, new_num_keypoints * sizeof(sift_keypoint_t));
  }

  return new_num_keypoints;
}

static void calculate_keypoint_descriptors(const detector_context_t *context, uint32_t num_keypoints, sift_keypoint_t **keypoints) {
  const float bin_size = SIFT_DESCRIPTOR_BINS / 360.0f;

  for (uint32_t k = 0; k < num_keypoints; ++k) {
    sift_keypoint_t *kp = &(*keypoints)[k];
    const sift_image_t *img = context->gaussian_pyramid.data[kp->octave * (context->octave_layers + 3) + kp->layer];

    const float octave_scale = 1.0f / (1 << kp->octave);
    const float base_x = kp->x * octave_scale;
    const float base_y = kp->y * octave_scale;
    const float histogram_width = SIFT_DESCRIPTOR_SCALE_FACTOR * kp->size * octave_scale * 0.5f;
    const float sigma = 0.5f * SIFT_DESCRIPTOR_HISTOGRAMS;
    const float weight_multiplier = -0.5f / (sigma * sigma);

    const float adjusted_angle = 360.0f - kp->angle;
    const float cos_angle = cosf(M_PI / 180.0f * adjusted_angle);
    const float sin_angle = sinf(M_PI / 180.0f * adjusted_angle);
    const int32_t radius = (int32_t)roundf(histogram_width * sqrtf(2.0f) * (SIFT_DESCRIPTOR_HISTOGRAMS + 1) * 0.5f);

    float histogram[SIFT_DESCRIPTOR_HISTOGRAMS + 2][SIFT_DESCRIPTOR_HISTOGRAMS + 2][SIFT_DESCRIPTOR_BINS] = {{{0}}};
    for (int32_t y = -radius; y <= radius; ++y) {
      for (int32_t x = -radius; x <= radius; ++x) {
        const float y_rot = x * sin_angle + y * cos_angle;
        const float x_rot = x * cos_angle - y * sin_angle;
        const float y_bin = (y_rot / histogram_width) + SIFT_DESCRIPTOR_HISTOGRAMS * 0.5f - 0.5f;
        const float x_bin = (x_rot / histogram_width) + SIFT_DESCRIPTOR_HISTOGRAMS * 0.5f - 0.5f;

        if (y_bin > -1.0f && y_bin < SIFT_DESCRIPTOR_HISTOGRAMS &&
            x_bin > -1.0f && x_bin < SIFT_DESCRIPTOR_HISTOGRAMS) {

          const int32_t img_x = (int32_t)((base_x + x) + 0.5f);
          const int32_t img_y = (int32_t)((base_y + y) + 0.5f);

          if (img_x > 0 && img_x < (int32_t)(img->width - 1) &&
              img_y > 0 && img_y < (int32_t)(img->height - 1)) {

            const float dx = sift_image_get_pixel(img, img_x + 1, img_y) - sift_image_get_pixel(img, img_x - 1, img_y);
            const float dy = sift_image_get_pixel(img, img_x, img_y - 1) - sift_image_get_pixel(img, img_x, img_y + 1);
            const float mag = sqrtf(dx * dx + dy * dy);

            float angle = atan2f(dy, dx) * (180.0f / M_PI);
            angle = fmodf(angle + 360.0f, 360.0f);

            const float y_rot_rel = y_rot / histogram_width;
            const float x_rot_rel = x_rot / histogram_width;
            const float weight = expf((x_rot_rel * x_rot_rel + y_rot_rel * y_rot_rel) * weight_multiplier) * mag;
            float o_bin = (angle - adjusted_angle) * bin_size;

            const int32_t y0 = (int32_t)floorf(y_bin);
            const int32_t x0 = (int32_t)floorf(x_bin);
            int32_t o0 = (int32_t)floorf(o_bin);

            const float dy_frac = y_bin - y0;
            const float dx_frac = x_bin - x0;
            const float do_frac = o_bin - o0;

            if (o0 < 0)                     { o0 += SIFT_DESCRIPTOR_BINS; }
            if (o0 >= SIFT_DESCRIPTOR_BINS) { o0 -= SIFT_DESCRIPTOR_BINS; }

            const float c1   = weight * dy_frac;
            const float c0   = weight * (1.0f - dy_frac);
            const float c11  = c1 * dx_frac;
            const float c10  = c1 * (1.0f - dx_frac);
            const float c01  = c0 * dx_frac;
            const float c00  = c0 * (1.0f - dx_frac);
            const float c111 = c11 * do_frac;
            const float c110 = c11 * (1.0f - do_frac);
            const float c101 = c10 * do_frac;
            const float c100 = c10 * (1.0f - do_frac);
            const float c011 = c01 * do_frac;
            const float c010 = c01 * (1.0f - do_frac);
            const float c001 = c00 * do_frac;
            const float c000 = c00 * (1.0f - do_frac);
            histogram[y0 + 1][x0 + 1][o0]       += c000;
            histogram[y0 + 1][x0 + 1][(o0 + 1)] += c001;
            histogram[y0 + 1][x0 + 2][o0]       += c010;
            histogram[y0 + 1][x0 + 2][(o0 + 1)] += c011;
            histogram[y0 + 2][x0 + 1][o0]       += c100;
            histogram[y0 + 2][x0 + 1][(o0 + 1)] += c101;
            histogram[y0 + 2][x0 + 2][o0]       += c110;
            histogram[y0 + 2][x0 + 2][(o0 + 1)] += c111;
          }
        }
      }
    }

    float norm = 0.0f;
    for (int32_t i = 1; i < SIFT_DESCRIPTOR_HISTOGRAMS; ++i) {
      for (int32_t j = 1; j < SIFT_DESCRIPTOR_HISTOGRAMS; ++j) {
        for (int32_t k = 0; k < SIFT_DESCRIPTOR_BINS; ++k) {
          norm += histogram[i][j][k] * histogram[i][j][k];
        }
      }
    }
    norm = sqrtf(norm);

    float new_norm = 0.0f;
    for (int32_t i = 1; i < SIFT_DESCRIPTOR_HISTOGRAMS; ++i) {
      for (int32_t j = 1; j < SIFT_DESCRIPTOR_HISTOGRAMS; ++j) {
        for (int32_t k = 0; k < SIFT_DESCRIPTOR_BINS; ++k) {
          float value = fminf(histogram[i][j][k] / norm, 0.2f);
          histogram[i][j][k] = value;
          new_norm += value * value;
        }
      }
    }
    new_norm = sqrtf(new_norm);

    int32_t idx = 0;
    for (int32_t i = 1; i < SIFT_DESCRIPTOR_HISTOGRAMS; ++i) {
      for (int32_t j = 1; j < SIFT_DESCRIPTOR_HISTOGRAMS; ++j) {
        for (int32_t k = 0; k < SIFT_DESCRIPTOR_BINS; ++k) {
          float value = roundf(histogram[i][j][k] / new_norm * 512.0f);
          value = fminf(fmaxf(value, 0.0f), 255.0f);
          kp->descriptor[idx++] = value;
        }
      }
    }
  }
}

static float descriptor_distance(const uint8_t desc1[128], const uint8_t desc2[128]) {
  float distance_sq = 0.0f;

  // Process 4 elements at a time for better pipelining
  for (int32_t i = 0; i < 128; i += 4) {
    const float diff0 = desc1[i]     - desc2[i];
    const float diff1 = desc1[i + 1] - desc2[i + 1];
    const float diff2 = desc1[i + 2] - desc2[i + 2];
    const float diff3 = desc1[i + 3] - desc2[i + 3];

    distance_sq += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;

    // Early bailout if distance already too large
    if (distance_sq > 32768.0f) {  // Empirical threshold
      return FLT_MAX;
    }
  }

  return sqrtf(distance_sq);
}

static int matches_comparator(const void *a, const void *b) {
    const sift_match_t *match_a = (const sift_match_t *)a;
    const sift_match_t *match_b = (const sift_match_t *)b;
    return (match_a->distance > match_b->distance) ? 1 : -1;
}

/* === Public Interface Implementation === */

sift_image_t *sift_image_create(uint32_t width, uint32_t height) {
  if (width == 0 || height == 0 ||
      width > SIFT_MAX_DIMENSION || height > SIFT_MAX_DIMENSION) { return NULL; }

  // Create an empty container
  sift_image_t *image = calloc(1, sizeof(sift_image_t));
  if (!image) { return NULL; }

  // Allocate memory for image data
  image->data = calloc(width * height, sizeof(float));
  if (!image->data) {
    sift_image_destroy(&image);
    return NULL;
  }

  // Store image meta
  image->width = width;
  image->height = height;

  return image;
}

void sift_image_destroy(sift_image_t **image) {
  if (!image || !(*image)) { return; }
  if ((*image)->data) {
    free((*image)->data);
    (*image)->data = NULL;
  }
  free(*image);
  *image = NULL;
}

float sift_image_get_pixel(const sift_image_t *image, uint32_t x, uint32_t y) {
  if (!image || !image->data || x >= image->width || y >= image->height) { return 0.0f; }
  return image->data[y * image->width + x];
}

void sift_image_set_pixel(sift_image_t *image, uint32_t x, uint32_t y, float value) {
  if (!image || !image->data || x >= image->width || y >= image->height) { return; }
  image->data[y * image->width + x] = value;
}

sift_image_t *sift_image_create_from_u8(const uint8_t *data, uint32_t width, uint32_t height, uint8_t channels) {
  if (!data ||
      width == 0 || height == 0 ||
      width > SIFT_MAX_DIMENSION || height > SIFT_MAX_DIMENSION ||
      channels < 1) { return NULL; }

  // Create empty image container
  sift_image_t *image = sift_image_create(width, height);
  if (!image) { return NULL; }

  // Copy data with conversion
  const float inv_norm = 1.0f / 255.0f;
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      const uint8_t *pixel = &data[(y * width + x) * channels];
      float value = 0.0f;
      if (channels == 1) {
        value = pixel[0] * inv_norm;
      } else if (channels >= 3) {
        value = (0.299f * pixel[0] + 0.587f * pixel[1] + 0.114f * pixel[2]) * inv_norm;
      }
      sift_image_set_pixel(image, x, y, value);
    }
  }

  return image;
}

sift_detector_t *sift_detector_create() {
  sift_detector_t *detector = calloc(1, sizeof(sift_detector_t));
  if (!detector) { return NULL; }

  // Default values from Lowe, 2004
  detector->sigma = 1.6f;
  detector->contrast_threshold = 0.04f;
  detector->edge_threshold = 10.0f;
  detector->octave_layers = 3;
  detector->border_width = 0;

  return detector;
}

void sift_detector_destroy(sift_detector_t **detector) {
  if (!detector || !(*detector)) { return; }
  free(*detector);
  *detector = NULL;
}

uint32_t sift_detect_and_compute(const sift_detector_t *detector, const sift_image_t *image, sift_keypoint_t **keypoints) {
  if (!detector || !image || !keypoints) { return 0; }

  // Create detector context
  detector_context_t context = {
    .sigma = detector->sigma,
    .contrast_threshold = detector->contrast_threshold / (float)detector->octave_layers,
    .edge_threshold = detector->edge_threshold,
    .octave_layers = detector->octave_layers,
    .border_width = detector->border_width,
    .octaves = 0,
    .base_image = NULL,
    .kernels.count = 0,
    .kernels.data = NULL,
    .gaussian_pyramid.count = 0,
    .gaussian_pyramid.data = NULL,
    .dog_pyramid.count = 0,
    .dog_pyramid.data = NULL
  };

  // 1. Create base image
  if (!create_base_image(image, 0.5f, &context)) {
    detector_context_destroy(&context);
    return 0;
  }

  // 2. Create gaussian kernels
  if (!create_gaussian_kernels(&context)) {
    detector_context_destroy(&context);
    return 0;
  }

  // 3. Calculate how many octaves?
  calculate_num_octaves(&context);

  // 4. Build gaussian pyramid
  if (!build_gaussian_pyramid(&context)) {
    detector_context_destroy(&context);
    return 0;
  }

  // 5. Build DoG pyramid
  if (!build_dog_pyramid(&context)) {
    detector_context_destroy(&context);
    return 0;
  }

  // 6. Find scale space extrema
  uint32_t num_keypoints = find_scale_space_extrema(&context, keypoints);

  // 7. Remove duplicates
  num_keypoints = remove_duplicate_keypoints(keypoints, num_keypoints);

  // 8. Calculate descriptors
  calculate_keypoint_descriptors(&context, num_keypoints, keypoints);

  // 9. Convert found keypoints into original image size
  for (uint32_t i = 0; i < num_keypoints; ++i) {
    (*keypoints)[i].x    *= 0.5f;
    (*keypoints)[i].y    *= 0.5f;
    (*keypoints)[i].size *= 0.5f;
  }

  // 10. Free all resources
  detector_context_destroy(&context);

  return num_keypoints;
}

void sift_keypoints_destroy(sift_keypoint_t **keypoints) {
  if (!keypoints || !(*keypoints)) { return; }
  free(*keypoints);
  *keypoints = NULL;
}

uint32_t sift_find_matches(const sift_keypoint_t *keys1, uint32_t count1,
                           const sift_keypoint_t *keys2, uint32_t count2,
                           sift_match_t **matches_out, float match_threshold,
                           bool mutual_check, bool sort_by_distance) {
  if (!keys1 || !keys2 || count1 == 0 || count2 == 0 || !matches_out) { return 0; }

  sift_match_t *matches = calloc(count1, sizeof(sift_match_t));
  if (!matches) { return 0; }

  uint32_t good_matches = 0;

  for (uint32_t i = 0; i < count1; ++i) {
    uint32_t best_idx = 0;
    float best_dist = FLT_MAX, second_best_dist = FLT_MAX;

    for (uint32_t j = 0; j < count2; ++j) {
      float dist = descriptor_distance(keys1[i].descriptor, keys2[j].descriptor);
      if (dist < best_dist) {
        second_best_dist = best_dist;
        best_dist = dist;
        best_idx = j;
      } else if (dist < second_best_dist) {
        second_best_dist = dist;
      }
    }

    if (best_dist < match_threshold * second_best_dist) {
      if (mutual_check) {
        uint32_t mutual_best_idx = 0;
        float mutual_best_dist = FLT_MAX;
        for (uint32_t k = 0; k < count1; ++k) {
          float dist = descriptor_distance(keys2[best_idx].descriptor, keys1[k].descriptor);
          if (dist < mutual_best_dist) {
            mutual_best_dist = dist;
            mutual_best_idx = k;
          }
        }

        if (mutual_best_idx != i) { continue; }
      }

      matches[good_matches].from_idx = i;
      matches[good_matches].to_idx = best_idx;
      matches[good_matches].distance = best_dist;
      good_matches++;
    }
  }

  if (sort_by_distance && good_matches > 0) {
    qsort(matches, good_matches, sizeof(sift_match_t), matches_comparator);
  }

  if (good_matches > 0) {
    *matches_out = realloc(matches, good_matches * sizeof(sift_match_t));
  } else {
    free(matches);
    *matches_out = NULL;
  }

  return good_matches;
}

