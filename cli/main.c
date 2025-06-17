#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"
#include "sift/sift.h"

#define CLI_VERSION "1.0.0"

static void print_help() {
  printf("SIFT Feature Tool v%s\n", CLI_VERSION);
  printf("Usage:\n");
  printf("  sift [options] <image1> [image2]\n\n");
  printf("Options:\n");
  printf("  --contrast <value>  Set contrast threshold (default: 0.03)\n");
  printf("  --edge <value>      Set edge threshold (default: 10.0)\n");
  printf("  --layers <n>        Layers per octave (default: 3)\n");
  printf("  --output <file>     Save keypoints to file\n");
  printf("  --help              Show this help\n\n");
  printf("Examples:\n");
  printf("  Detect keypoints: sift_tool image.jpg\n");
  printf("  Find matches: sift_tool image1.jpg image2.jpg\n");
}

static int detect_and_compute(const char* filename, const sift_detector_t* detector, const char* output_file) {
  // Load image
  int width, height, channels;
  unsigned char* image = stbi_load(filename, &width, &height, &channels, 0);
  if (!image) {
    fprintf(stderr, "Error loading image: %s\n", filename);
    return 1;
  }

  // Create SIFT image
  sift_image_t* sift_image = sift_image_create_from_u8(image, width, height, channels);
  stbi_image_free(image);
  if (!sift_image) {
    fprintf(stderr, "Error creating SIFT image\n");
    return 1;
  }

  // Detect keypoints
  sift_keypoint_t* keypoints = NULL;
  uint32_t num_keypoints = sift_detect_and_compute(detector, sift_image, &keypoints);
  printf("Found %d keypoints in %s\n", num_keypoints, filename);

  // Print or save keypoints
  if (output_file) {
    FILE* f = fopen(output_file, "w");
    if (f) {
      for (uint32_t i = 0; i < num_keypoints; i++) {
        fprintf(f, "%f,%f,%f,%f\n", 
                keypoints[i].x, keypoints[i].y, 
                keypoints[i].size, keypoints[i].angle);
      }
      fclose(f);
      printf("Saved keypoints to %s\n", output_file);
    } else {
      fprintf(stderr, "Error saving to file: %s\n", output_file);
    }
  } else {
    for (uint32_t i = 0; i < num_keypoints; i++) {
      printf("Keypoint %d: x=%.1f y=%.1f size=%.1f angle=%.1f\n",
             i, keypoints[i].x, keypoints[i].y, 
             keypoints[i].size, keypoints[i].angle);
      for (uint32_t j = 0; j < 128; ++j) {
        if (j % 18 == 0) { printf("\n"); }
        printf("%d ", keypoints[i].descriptor[j]);
      }
    }
  }

  // Cleanup
  sift_keypoints_destroy(&keypoints);
  sift_image_destroy(&sift_image);

  return 0;
}

static int find_matches(const char* file1, const char* file2, const sift_detector_t* detector, const char* output_file) {
  // Load first image
  int w1, h1, c1;
  unsigned char* img1 = stbi_load(file1, &w1, &h1, &c1, 0);
  if (!img1) {
    fprintf(stderr, "Error loading image: %s\n", file1);
    return 1;
  }
  sift_image_t* sift_img1 = sift_image_create_from_u8(img1, w1, h1, c1);
  stbi_image_free(img1);
  if (!sift_img1) return 1;

  // Load second image
  int w2, h2, c2;
  unsigned char* img2 = stbi_load(file2, &w2, &h2, &c2, 0);
  if (!img2) {
    fprintf(stderr, "Error loading image: %s\n", file2);
    sift_image_destroy(&sift_img1);
    return 1;
  }
  sift_image_t* sift_img2 = sift_image_create_from_u8(img2, w2, h2, c2);
  stbi_image_free(img2);
  if (!sift_img2) {
    sift_image_destroy(&sift_img1);
    return 1;
  }

  // Detect keypoints in both images
  sift_keypoint_t *keys1 = NULL, *keys2 = NULL;
  uint32_t count1 = sift_detect_and_compute(detector, sift_img1, &keys1);
  uint32_t count2 = sift_detect_and_compute(detector, sift_img2, &keys2);

  printf("Found %d keypoints in %s\n", count1, file1);
  printf("Found %d keypoints in %s\n", count2, file2);

  // Find matches
  sift_match_t *matches = NULL;
  uint32_t match_count = sift_find_matches(keys1, count1, keys2, count2, &matches, 0.7f, true, true);
  printf("Found %d good matches\n", match_count);

  // Print or save keypoints
  if (output_file) {
    FILE* f = fopen(output_file, "w");
    if (f) {
      for (uint32_t i = 0; i < match_count; i++) {
        uint32_t idx1 = matches[i].from_idx;
        uint32_t idx2 = matches[i].to_idx;
        fprintf(f, "%f,%f,%f,%f : %f,%f,%f,%f\n", keys1[idx1].x, keys1[idx1].y, keys1[idx1].size, keys1[idx1].angle,
                                                  keys2[idx2].x, keys2[idx2].y, keys2[idx2].size, keys2[idx2].angle);
      }
      fclose(f);
      printf("Saved keypoints to %s\n", output_file);
    } else {
      fprintf(stderr, "Error saving to file: %s\n", output_file);
    }
  } else {
    for (uint32_t i = 0; i < match_count; i++) {
      uint32_t idx1 = matches[i].from_idx;
      uint32_t idx2 = matches[i].to_idx;
      printf("Match %d: %s (%.1f,%.1f) <-> %s (%.1f,%.1f)\n",
             i, file1, keys1[idx1].x, keys1[idx1].y,
             file2, keys2[idx2].x, keys2[idx2].y);
    }
  }

  // Cleanup
  sift_matches_destroy(&matches);
  sift_keypoints_destroy(&keys1);
  sift_keypoints_destroy(&keys2);
  sift_image_destroy(&sift_img1);
  sift_image_destroy(&sift_img2);

  return 0;
}

int main(int argc, char *argv[]) {
  const char* output_file = NULL;
  const char* image1 = NULL;
  const char* image2 = NULL;

  // 1. Default detector parameters
  sift_detector_t* detector = sift_detector_create();
  if (!detector) {
    fprintf(stderr, "Error creating SIFT detector\n");
    return 1;
  }

  // 2. Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--help") == 0) {
      print_help();
      sift_detector_destroy(&detector);
      return 0;
    } else if (strcmp(argv[i], "--contrast") == 0 && (i + 1) < argc) {
      detector->contrast_threshold = atof(argv[++i]);
    } else if (strcmp(argv[i], "--edge") == 0 && (i + 1) < argc) {
      detector->edge_threshold = atof(argv[++i]);
    } else if (strcmp(argv[i], "--layers") == 0 && (i + 1) < argc) {
      detector->octave_layers = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--output") == 0 && (i + 1) < argc) {
      output_file = argv[++i];
    } else if (!image1) {
      image1 = argv[i];
    } else if (!image2) {
      image2 = argv[i];
    }
  }

  if (!image1) {
    print_help();
    sift_detector_destroy(&detector);
    return 1;
  }

  // 3. Processing
  int result;
  if (image2) {
    result = find_matches(image1, image2, detector, output_file);
  } else {
    result = detect_and_compute(image1, detector, output_file);
  }

  // Cleanup
  sift_detector_destroy(&detector);

  return result;
}

