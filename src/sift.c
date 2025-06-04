#include "sift/sift.h"
#include <stdlib.h>

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

