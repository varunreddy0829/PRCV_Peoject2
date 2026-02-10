/*
Name: Varun Reddy Patlolla
*/


#include <opencv2/opencv.hpp>
#include <vector>
#include "features.h"

//Function prtotypes
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

/**
 * Baseline: Extracts a 7x7 square in the middle.
 * This is a spatial feature vector.
 */
int extractBaselineFeatures(cv::Mat &src, std::vector<float> &featureVector) {
    if (src.cols < 7 || src.rows < 7) return -1;

    int startX = (src.cols / 2) - 3;
    int startY = (src.rows / 2) - 3;

    for (int y = startY; y < startY + 7; y++) {
        for (int x = startX; x < startX + 7; x++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
            // Feature vector is built by flattening the 2D patch
            featureVector.push_back((float)pixel[0]); // B
            featureVector.push_back((float)pixel[1]); // G
            featureVector.push_back((float)pixel[2]); // R
        }
    }
    return 0;
}



/**
 * Task 2: extractHistogramFeatures
 * Purpose: Creates a 3D RGB histogram of the entire image.
 * Logic: 
 * - Divides 0-255 range into 8 buckets (size 32 each).
 * - Iterates through every pixel and increments the corresponding bucket.
 * - Normalizes by total pixel count so images of different sizes can be compared.
 */
int extractHistogramFeatures(cv::Mat &src, std::vector<float> &featureVector) {
    const int BINS = 8; 
    int dims[3] = {BINS, BINS, BINS};
    
    // Create a 3D matrix to act as our histogram (initialized to 0)
    cv::Mat hist = cv::Mat::zeros(3, dims, CV_32F);

    // Step 1: Count pixels into 3D bins
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            
            // Map 0-255 to 0-7 bin index using integer division
            int b_bin = pixel[0] / 32;
            int g_bin = pixel[1] / 32;
            int r_bin = pixel[2] / 32;

            hist.at<float>(r_bin, g_bin, b_bin)++;
        }
    }

    // Step 2: Normalize and Flatten
    // We flatten the 3D cube into a 1D vector so it can be saved in the CSV row.
    float totalPixels = (float)src.rows * src.cols;
    featureVector.clear();

    for (int r = 0; r < BINS; r++) {
        for (int g = 0; g < BINS; g++) {
            for (int b = 0; b < BINS; b++) {
                // Normalize by dividing by total pixel count
                float val = hist.at<float>(r, g, b) / totalPixels;
                featureVector.push_back(val);
            }
        }
    }

    return 0;
}

/**
 * compareHistogramIntersection
 * Purpose: Computes the "overlap" between two histograms.
 * Logic: 
 * - Sum of the minimum values of corresponding bins.
 * - Intersection returns 1.0 for perfect match, 0.0 for no common colors.
 * - We return (1.0 - intersection) because our Matcher treats SMALL numbers as better matches.
 */
float compareHistogramIntersection(const std::vector<float> &v1, const std::vector<float> &v2) {
    float intersection = 0.0;
    
    for (size_t i = 0; i < v1.size(); i++) {
        // Add the smaller of the two values to the intersection sum
        intersection += std::min(v1[i], v2[i]);
    }

    // Convert similarity (1 is best) to distance (0 is best)
    return 1.0f - intersection;
}

/**
 * getRegionHist (Helper)
 * Purpose: Extracts a normalized 8x8x8 RGB histogram from a specific sub-region.
 * Input: cv::Mat region
 * Output: std::vector<float> (512 bins)
 */
std::vector<float> getRegionHist(cv::Mat &region) {
    const int BINS = 8;
    int dims[3] = {BINS, BINS, BINS};
    cv::Mat hist = cv::Mat::zeros(3, dims, CV_32F);

    for (int i = 0; i < region.rows; i++) {
        for (int j = 0; j < region.cols; j++) {
            cv::Vec3b pixel = region.at<cv::Vec3b>(i, j);
            // Map BGR to 8 bins
            hist.at<float>(pixel[2]/32, pixel[1]/32, pixel[0]/32)++;
        }
    }

    float total = (float)region.rows * region.cols;
    std::vector<float> out;
    for (int r = 0; r < BINS; r++) {
        for (int g = 0; g < BINS; g++) {
            for (int b = 0; b < BINS; b++) {
                out.push_back(hist.at<float>(r, g, b) / total);
            }
        }
    }
    return out;
}

/**
 * Task 3: extractMultiHistogramFeatures
 * Purpose: Creates 3 separate histograms for spatial regions and joins them.
 * Logic: 
 * - Top Half (Rows 0 to Height/2)
 * - Bottom Half (Rows Height/2 to Height)
 * - Center 100x100 patch
 */
int extractMultiHistogramFeatures(cv::Mat &src, std::vector<float> &featureVector) {
    featureVector.clear();

    // 1. Top Half ROI
    cv::Rect topRect(0, 0, src.cols, src.rows / 2);
    cv::Mat topRegion = src(topRect);
    std::vector<float> topHist = getRegionHist(topRegion);

    // 2. Bottom Half ROI
    cv::Rect botRect(0, src.rows / 2, src.cols, src.rows / 2);
    cv::Mat botRegion = src(botRect);
    std::vector<float> botHist = getRegionHist(botRegion);

    // 3. Middle 100x100 ROI (ensure we don't go out of bounds)
    int startX = std::max(0, (src.cols / 2) - 50);
    int startY = std::max(0, (src.rows / 2) - 50);
    cv::Rect midRect(startX, startY, 100, 100);
    cv::Mat midRegion = src(midRect);
    std::vector<float> midHist = getRegionHist(midRegion);

    // Concatenate into a single feature vector (1536 elements)
    featureVector.insert(featureVector.end(), topHist.begin(), topHist.end());
    featureVector.insert(featureVector.end(), botHist.begin(), botHist.end());
    featureVector.insert(featureVector.end(), midHist.begin(), midHist.end());

    return 0;
}

/**
 * extractColorTextureFeatures
 * Logic: 
 * 1. Computes a 512-bin Color Histogram (same as Task 2).
 * 2. Computes Sobel Magnitude using your Project 1 code.
 * 3. Creates a 32-bin Histogram of those magnitudes.
 */
int extractColorTextureFeatures(cv::Mat &src, std::vector<float> &featureVector) {
    featureVector.clear();

    // --- PART 1: COLOR (512 Bins) ---
    std::vector<float> colorHist;
    extractHistogramFeatures(src, colorHist); // Reusing your Task 2 code
    featureVector.insert(featureVector.end(), colorHist.begin(), colorHist.end());

    // --- PART 2: TEXTURE (32 Bins) ---
    cv::Mat sx, sy, mag;
    sobelX3x3(src, sx);
    sobelY3x3(src, sy);
    magnitude(sx, sy, mag);

    // Convert 3-channel magnitude to 1-channel for texture analysis
    cv::Mat grayMag;
    cv::cvtColor(mag, grayMag, cv::COLOR_BGR2GRAY);

    const int TEXTURE_BINS = 32;
    std::vector<float> texHist(TEXTURE_BINS, 0.0f);

    for (int i = 0; i < grayMag.rows; i++) {
        for (int j = 0; j < grayMag.cols; j++) {
            uchar val = grayMag.at<uchar>(i, j);
            // Map 0-255 to 0-31
            int bin = val / (256 / TEXTURE_BINS);
            if (bin >= TEXTURE_BINS) bin = TEXTURE_BINS - 1;
            texHist[bin]++;
        }
    }

    // Normalize texture histogram
    float totalPixels = (float)grayMag.rows * grayMag.cols;
    for (int i = 0; i < TEXTURE_BINS; i++) {
        featureVector.push_back(texHist[i] / totalPixels);
    }

    return 0;
}

/**
 * Sobel X filter: detects vertical edges (positive right)
 * Horizontal: [-1 0 1]
 * Vertical: [1 2 1]
 * Combined: [-1 0 1; -2 0 2; -1 0 1]
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
  if (src.empty()) {
    printf("Error: Source image is empty\n");
    return -1;
  }

  // Create destination as 16-bit signed short
  dst.create(src.size(), CV_16SC3);
  dst = cv::Scalar(0, 0, 0);

  // Separable filters
  int horizontal[3] = {-1, 0, 1};
  int vertical[3] = {1, 2, 1};

  // Temporary image for horizontal pass
  cv::Mat temp(src.size(), CV_16SC3, cv::Scalar(0, 0, 0));

  // Horizontal pass: [-1 0 1]
  for (int i = 0; i < src.rows; i++) {
    cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
    cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(i);

    for (int j = 1; j < src.cols - 1; j++) {
      for (int c = 0; c < 3; c++) {
        int sum = 0;
        for (int k = -1; k <= 1; k++) {
          sum += srcRow[j + k][c] * horizontal[k + 1];
        }
        tempRow[j][c] = sum;
      }
    }
  }

  // Vertical pass: [1 2 1]
  for (int i = 1; i < temp.rows - 1; i++) {
    cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(i);

    for (int j = 0; j < temp.cols; j++) {
      for (int c = 0; c < 3; c++) {
        int sum = 0;
        for (int k = -1; k <= 1; k++) {
          sum += temp.ptr<cv::Vec3s>(i + k)[j][c] * vertical[k + 1];
        }
        dstRow[j][c] = sum / 4; // Normalize by sum of vertical filter
      }
    }
  }

  return 0;
}

/**
 * Sobel Y filter: detects horizontal edges (positive up)
 * Horizontal: [1 2 1]
 * Vertical: [1 0 -1]
 * Combined: [1 2 1; 0 0 0; -1 -2 -1]
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
  if (src.empty()) {
    printf("Error: Source image is empty\n");
    return -1;
  }

  // Create destination as 16-bit signed short
  dst.create(src.size(), CV_16SC3);
  dst = cv::Scalar(0, 0, 0);

  // Separable filters
  int horizontal[3] = {1, 2, 1};
  int vertical[3] = {1, 0, -1};

  // Temporary image for horizontal pass
  cv::Mat temp(src.size(), CV_16SC3, cv::Scalar(0, 0, 0));

  // Horizontal pass: [1 2 1]
  for (int i = 0; i < src.rows; i++) {
    cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
    cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(i);

    for (int j = 1; j < src.cols - 1; j++) {
      for (int c = 0; c < 3; c++) {
        int sum = 0;
        for (int k = -1; k <= 1; k++) {
          sum += srcRow[j + k][c] * horizontal[k + 1];
        }
        tempRow[j][c] = sum;
      }
    }
  }

  // Vertical pass: [1 0 -1]
  for (int i = 1; i < temp.rows - 1; i++) {
    cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(i);

    for (int j = 0; j < temp.cols; j++) {
      for (int c = 0; c < 3; c++) {
        int sum = 0;
        for (int k = -1; k <= 1; k++) {
          sum += temp.ptr<cv::Vec3s>(i + k)[j][c] * vertical[k + 1];
        }
        dstRow[j][c] = sum / 4; // Normalize by sum of horizontal filter
      }
    }
  }

  return 0;
}
//----------------Task 8------------------
/**
 * Calculate gradient magnitude from Sobel X and Y.
 * Magnitude = sqrt(sx*sx + sy*sy)
 */
/**
 * @brief Calculates the gradient magnitude from horizontal and vertical Sobel
 * images.
 *
 * Computes Euclidean distance: magnitude = sqrt(sx^2 + sy^2).
 *
 * @param sx [cv::Mat&] The Sobel X gradient image (16-bit signed).
 * @param sy [cv::Mat&] The Sobel Y gradient image (16-bit signed).
 * @param dst [cv::Mat&] The output magnitude image (8-bit unsigned).
 * @return int Returns 0 on success, -1 on empty input.
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
  if (sx.empty() || sy.empty()) {
    printf("Error: Input images are empty\n");
    return -1;
  }

  // Create destination as 8-bit unsigned char
  dst.create(sx.size(), CV_8UC3);

  for (int i = 0; i < sx.rows; i++) {
    cv::Vec3s *sxRow = sx.ptr<cv::Vec3s>(i);
    cv::Vec3s *syRow = sy.ptr<cv::Vec3s>(i);
    cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

    for (int j = 0; j < sx.cols; j++) {
      for (int c = 0; c < 3; c++) {
        float mag = sqrt(sxRow[j][c] * sxRow[j][c] + syRow[j][c] * syRow[j][c]);
        // Clamp to 0-255 range
        dstRow[j][c] = cv::saturate_cast<uchar>(mag);
      }
    }
  }

  return 0;
}