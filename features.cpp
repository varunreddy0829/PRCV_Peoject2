/*
Name: Varun Reddy Patlolla
Project: Computer Vision Image Retrieval (Tasks 1 - 5)
Date: 02/09/2026

DESCRIPTION:
This library contains all image processing and feature extraction logic, 
including custom Sobel filters, histogram binning, and distance metrics 
(SSD, Intersection, and Cosine).
*/

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "features.h"

// =============================================================================
// DISTANCE METRICS
// =============================================================================

/**
 * calculateSSD
 * Purpose: Sum of Squared Differences for Baseline matching.
 */
float calculateSSD(const std::vector<float> &v1, const std::vector<float> &v2) {
    float sum = 0.0;
    for (size_t i = 0; i < v1.size(); i++) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sum;
}

/**
 * compareHistogramIntersection
 * Purpose: Computes similarity based on overlap between two histograms.
 * Returns: (1.0 - intersection) as a distance (0.0 is perfect match).
 */
float compareHistogramIntersection(const std::vector<float> &v1, const std::vector<float> &v2) {
    float intersection = 0.0;
    for (size_t i = 0; i < v1.size(); i++) {
        intersection += std::min(v1[i], v2[i]);
    }
    return 1.0f - intersection;
}

/**
 * calculateCosineDistance
 * Purpose: Measures angular distance for Deep Embeddings (Task 5).
 */
float calculateCosineDistance(const std::vector<float> &v1, const std::vector<float> &v2) {
    float dotProduct = 0.0, normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < v1.size(); i++) {
        dotProduct += v1[i] * v2[i];
        normA += v1[i] * v1[i];
        normB += v2[i] * v2[i];
    }
    if (normA == 0 || normB == 0) return 1.0f;
    return 1.0f - (dotProduct / (std::sqrt(normA) * std::sqrt(normB)));
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * getRegionHist
 * Purpose: Extracts a normalized 8x8x8 RGB histogram from a specific sub-region.
 */
std::vector<float> getRegionHist(cv::Mat &region) {
    const int BINS = 8;
    int dims[3] = {BINS, BINS, BINS};
    cv::Mat hist = cv::Mat::zeros(3, dims, CV_32F);

    for (int i = 0; i < region.rows; i++) {
        for (int j = 0; j < region.cols; j++) {
            cv::Vec3b pixel = region.at<cv::Vec3b>(i, j);
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

// =============================================================================
// TASK EXTRACTION FUNCTIONS
// =============================================================================

/**
 * Task 1: extractBaselineFeatures
 * Extracts a 7x7 middle patch (147 values).
 */
int extractBaselineFeatures(cv::Mat &src, std::vector<float> &featureVector) {
    if (src.cols < 7 || src.rows < 7) return -1;
    int startX = (src.cols / 2) - 3;
    int startY = (src.rows / 2) - 3;

    for (int y = startY; y < startY + 7; y++) {
        for (int x = startX; x < startX + 7; x++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
            featureVector.push_back((float)pixel[0]); // B
            featureVector.push_back((float)pixel[1]); // G
            featureVector.push_back((float)pixel[2]); // R
        }
    }
    return 0;
}

/**
 * Task 2: extractHistogramFeatures
 * Global 3D RGB histogram (512 bins).
 */
int extractHistogramFeatures(cv::Mat &src, std::vector<float> &featureVector) {
    featureVector = getRegionHist(src);
    return 0;
}

/**
 * Task 3: extractMultiHistogramFeatures
 * Spatial histograms: Top, Bottom, and 100x100 Mid (1536 bins total).
 */
int extractMultiHistogramFeatures(cv::Mat &src, std::vector<float> &featureVector) {
    featureVector.clear();
    // Top
    cv::Rect topRect(0, 0, src.cols, src.rows / 2);
    cv::Mat topRegion = src(topRect);
    std::vector<float> topHist = getRegionHist(topRegion);
    // Bottom
    cv::Rect botRect(0, src.rows / 2, src.cols, src.rows / 2);
    cv::Mat botRegion = src(botRect);
    std::vector<float> botHist = getRegionHist(botRegion);
    // Mid 100x100
    int sX = std::max(0, (src.cols / 2) - 50);
    int sY = std::max(0, (src.rows / 2) - 50);
    cv::Rect midRect(sX, sY, std::min(100, src.cols), std::min(100, src.rows));
    cv::Mat midRegion = src(midRect);
    std::vector<float> midHist = getRegionHist(midRegion);

    featureVector.insert(featureVector.end(), topHist.begin(), topHist.end());
    featureVector.insert(featureVector.end(), botHist.begin(), botHist.end());
    featureVector.insert(featureVector.end(), midHist.begin(), midHist.end());
    return 0;
}

/**
 * Task 4: extractColorTextureFeatures
 * Global color (512) + Sobel Magnitude texture (32) = 544 bins.
 */
int extractColorTextureFeatures(cv::Mat &src, std::vector<float> &featureVector) {
    featureVector.clear();
    std::vector<float> colorHist = getRegionHist(src);
    featureVector.insert(featureVector.end(), colorHist.begin(), colorHist.end());

    cv::Mat sx, sy, mag;
    sobelX3x3(src, sx);
    sobelY3x3(src, sy);
    magnitude(sx, sy, mag);

    cv::Mat grayMag;
    cv::cvtColor(mag, grayMag, cv::COLOR_BGR2GRAY);

    const int TEX_BINS = 32;
    std::vector<float> texHist(TEX_BINS, 0.0f);
    for (int i = 0; i < grayMag.rows; i++) {
        for (int j = 0; j < grayMag.cols; j++) {
            uchar val = grayMag.at<uchar>(i, j);
            int bin = val / (256 / TEX_BINS);
            if (bin >= TEX_BINS) bin = TEX_BINS - 1;
            texHist[bin]++;
        }
    }
    float total = (float)grayMag.rows * grayMag.cols;
    for (int i = 0; i < TEX_BINS; i++) featureVector.push_back(texHist[i] / total);
    return 0;
}

// =============================================================================
// PROJECT 1 FILTERS (SOBEL & MAGNITUDE)
// =============================================================================

int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;
    dst.create(src.size(), CV_16SC3);
    cv::Mat temp(src.size(), CV_16SC3);
    int h[3] = {-1, 0, 1}, v[3] = {1, 2, 1};

    for (int i = 0; i < src.rows; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                int sum = 0;
                for (int k = -1; k <= 1; k++) sum += src.at<cv::Vec3b>(i, j+k)[c] * h[k+1];
                temp.at<cv::Vec3s>(i, j)[c] = sum;
            }
        }
    }
    for (int i = 1; i < temp.rows - 1; i++) {
        for (int j = 0; j < temp.cols; j++) {
            for (int c = 0; c < 3; c++) {
                int sum = 0;
                for (int k = -1; k <= 1; k++) sum += temp.at<cv::Vec3s>(i+k, j)[c] * v[k+1];
                dst.at<cv::Vec3s>(i, j)[c] = sum / 4;
            }
        }
    }
    return 0;
}

int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;
    dst.create(src.size(), CV_16SC3);
    cv::Mat temp(src.size(), CV_16SC3);
    int h[3] = {1, 2, 1}, v[3] = {1, 0, -1};

    for (int i = 0; i < src.rows; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                int sum = 0;
                for (int k = -1; k <= 1; k++) sum += src.at<cv::Vec3b>(i, j+k)[c] * h[k+1];
                temp.at<cv::Vec3s>(i, j)[c] = sum;
            }
        }
    }
    for (int i = 1; i < temp.rows - 1; i++) {
        for (int j = 0; j < temp.cols; j++) {
            for (int c = 0; c < 3; c++) {
                int sum = 0;
                for (int k = -1; k <= 1; k++) sum += temp.at<cv::Vec3s>(i+k, j)[c] * v[k+1];
                dst.at<cv::Vec3s>(i, j)[c] = sum / 4;
            }
        }
    }
    return 0;
}

int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    if (sx.empty() || sy.empty()) return -1;
    dst.create(sx.size(), CV_8UC3);
    for (int i = 0; i < sx.rows; i++) {
        for (int j = 0; j < sx.cols; j++) {
            for (int c = 0; c < 3; c++) {
                float m = std::sqrt(sx.at<cv::Vec3s>(i, j)[c] * sx.at<cv::Vec3s>(i, j)[c] + 
                                   sy.at<cv::Vec3s>(i, j)[c] * sy.at<cv::Vec3s>(i, j)[c]);
                dst.at<cv::Vec3b>(i, j)[c] = cv::saturate_cast<uchar>(m);
            }
        }
    }
    return 0;
}

// =============================================================================
// NEW: DISPLAY LOGIC
// =============================================================================

/**
 * ImageMatch Struct
 * Purpose: Pairs a filename with its distance for easy sorting and display.
 */
struct ImageMatch {
    std::string filename;
    float distance;
};

/**
 * displayTopMatches
 * Purpose: Iterates through results to print names and open OpenCV windows.
 */
void displayTopMatches(const std::vector<ImageMatch> &results, const std::string &imageDir) {
    std::cout << "\nDisplaying Top Results..." << std::endl;
    
    for (size_t i = 0; i < results.size(); i++) {
        // Build the full path if the directory is provided
        std::string fullPath = imageDir + "/" + results[i].filename;
        
        cv::Mat img = cv::imread(fullPath);
        if (img.empty()) {
            // Try reading without prepending directory in case filename is a full path
            img = cv::imread(results[i].filename);
        }

        if (!img.empty()) {
            std::string winName = "Match " + std::to_string(i) + ": " + results[i].filename;
            cv::imshow(winName, img);
            std::cout << "Opened window for: " << results[i].filename << std::endl;
        } else {
            std::cout << "Warning: Could not find image file " << results[i].filename << std::endl;
        }
    }

    std::cout << "Press any key on an image window to exit." << std::endl;
    cv::waitKey(0);
}
// =============================================================================
// HELPER: HSV HISTOGRAM
// =============================================================================
std::vector<float> getRegionHistHSV(cv::Mat &region) {
  const int H_BINS = 8;
  const int S_BINS = 8;
  const int V_BINS = 8;
  int dims[3] = {H_BINS, S_BINS, V_BINS};
  cv::Mat hist = cv::Mat::zeros(3, dims, CV_32F);

  cv::Mat hsv;
  cv::cvtColor(region, hsv, cv::COLOR_BGR2HSV);

  for (int i = 0; i < hsv.rows; i++) {
    for (int j = 0; j < hsv.cols; j++) {
      cv::Vec3b pixel = hsv.at<cv::Vec3b>(i, j);
      // Hue is 0-179 in OpenCV, Sat/Val are 0-255
      int h = pixel[0] * H_BINS / 180;
      if (h >= H_BINS)
        h = H_BINS - 1;

      int s = pixel[1] * S_BINS / 256;
      if (s >= S_BINS)
        s = S_BINS - 1;

      int v = pixel[2] * V_BINS / 256;
      if (v >= V_BINS)
        v = V_BINS - 1;

      hist.at<float>(h, s, v)++;
    }
  }

  float total = (float)hsv.rows * hsv.cols;
  std::vector<float> out;
  for (int h = 0; h < H_BINS; h++) {
    for (int s = 0; s < S_BINS; s++) {
      for (int v = 0; v < V_BINS; v++) {
        out.push_back(hist.at<float>(h, s, v) / total);
      }
    }
  }
  return out;
}

/**
 * Task 7: extractTask7Features
 * HSV Spatial histograms: Top, Bottom, and 100x100 Mid (1536 bins total).
 */
int extractTask7Features(cv::Mat &src, std::vector<float> &featureVector) {
  featureVector.clear();
  // Top
  cv::Rect topRect(0, 0, src.cols, src.rows / 2);
  cv::Mat topRegion = src(topRect);
  std::vector<float> topHist = getRegionHistHSV(topRegion);
  // Bottom
  cv::Rect botRect(0, src.rows / 2, src.cols, src.rows / 2);
  cv::Mat botRegion = src(botRect);
  std::vector<float> botHist = getRegionHistHSV(botRegion);
  // Mid 100x100
  int sX = std::max(0, (src.cols / 2) - 50);
  int sY = std::max(0, (src.rows / 2) - 50);
  cv::Rect midRect(sX, sY, std::min(100, src.cols), std::min(100, src.rows));
  cv::Mat midRegion = src(midRect);
  std::vector<float> midHist = getRegionHistHSV(midRegion);

  featureVector.insert(featureVector.end(), topHist.begin(), topHist.end());
  featureVector.insert(featureVector.end(), botHist.begin(), botHist.end());
  featureVector.insert(featureVector.end(), midHist.begin(), midHist.end());
  return 0;
}

// =============================================================================
// EXTENSION 1: GABOR TEXTURE FEATURES
// =============================================================================

/**
 * extractGaborFeatures
 * Generates a bank of Gabor filters (4 orientations * 2 scales).
 * Computes Mean and StdDev of the response for each filter.
 * Vector Size: 16 floats.
 */
int extractGaborFeatures(cv::Mat &src, std::vector<float> &featureVector) {
  featureVector.clear();

  cv::Mat gray;
  if (src.channels() == 3) {
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = src.clone(); // Already gray
  }

  // Gabor Parameters
  int ksize = 31;      // Kernel size
  double lambd = 10.0; // Wavelength
  double gamma = 0.5;  // Aspect ratio
  double psi = 0;      // Phase offset

  // 2 Scales (Sigmas)
  std::vector<double> sigmas = {2.0, 4.0};

  // 4 Orientations (Thetas in Radians)
  // 0, 45, 90, 135 degrees
  std::vector<double> thetas = {0, CV_PI / 4, CV_PI / 2, 3 * CV_PI / 4};

  for (double sigma : sigmas) {
    for (double theta : thetas) {
      cv::Mat kernel = cv::getGaborKernel(cv::Size(ksize, ksize), sigma, theta,
                                          lambd, gamma, psi, CV_32F);

      cv::Mat dest;
      cv::filter2D(gray, dest, CV_32F, kernel);

      // Compute Mean and StdDev of the response
      cv::Scalar mean, stddev;
      cv::meanStdDev(dest, mean, stddev);

      featureVector.push_back((float)mean[0]);
      featureVector.push_back((float)stddev[0]);
    }
  }

  return 0;
}
