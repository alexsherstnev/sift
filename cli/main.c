#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include "sift/sift.h"

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

  // SIFT detector
  sift_detector_t *detector = sift_detector_create();
  if (!detector) {
    // TODO: Print error
    sift_image_destroy(&sift_image);
    return 1;
  }

  // Tests
  detector->contrast_threshold = 0.04f;
  sift_keypoint_t *keypoints = NULL;
  const uint32_t num_keypoints = sift_detect_and_compute(detector, sift_image, &keypoints);
  printf("Found keypoints: %d\n", num_keypoints);
  if (num_keypoints > 0) {
    for (uint32_t i = 0; i < num_keypoints; ++i) {
      sift_keypoint_t keypoint = keypoints[i];
      printf("%f, %f, %f, %f\n", keypoint.x, keypoint.y, keypoint.size, keypoint.angle);
    }
  }

  // Free resources
  sift_keypoints_destroy(&keypoints);
  sift_detector_destroy(&detector);
  sift_image_destroy(&sift_image);

  return 0;
}

