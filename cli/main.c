#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include "sift/sift.h"

static void save_sift_image_to_jpg(const char *path, const sift_image_t *image) {
  unsigned char *data = calloc(image->width * image->height * 4, sizeof(unsigned char));
  for (uint32_t y = 0; y < image->height; ++y) {
    for (uint32_t x = 0; x < image->width; ++x) {
      const uint32_t index = (y * image->width + x) * 4;

      float value = sift_image_get_pixel(image, x, y) * 255.0f;
      if (value < 0.0f)   { value = 0.0f;   }
      if (value > 255.0f) { value = 255.0f; }

      data[index + 0] = value;
      data[index + 1] = value;
      data[index + 2] = value;
      data[index + 3] = 255;
    }
  }

  stbi_write_jpg(path, image->width, image ->height, 4, data, 90);
  free(data);
}

int main(int argc, char *argv[]) {
  if (argc < 2) { return 0; }

  // Load image
  int width, height, channels;
  unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
  if (!image) {
    // TODO: Print error
    return 1;
  }

  // Create SIFT image
  sift_image_t *sift_image = sift_image_create_from_u8(image, width, height, channels);
  stbi_image_free(image);
  if (!sift_image) {
    // TODO: Print error
    return 1;
  }

  // Tests
  save_sift_image_to_jpg("image.jpg", sift_image);

  // Free resources
  sift_image_destroy(&sift_image);

  return 0;
}

