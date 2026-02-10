/*
  Name:Varun Reddy Patlolla
  Date : 9th Feb 2026
  Library for feature extraction and distance metrics
*/

#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>
#include <vector>

// TASK 1: Baseline 7x7 Square
int extractBaselineFeatures(cv::Mat &src, std::vector<float> &featureVector);

// TASK 2: 3D RGB Histogram Extraction
// Input: cv::Mat src (The image to process)
// Output: std::vector<float> &featureVector (The flattened, normalized 512-bin histogram)
int extractHistogramFeatures(cv::Mat &src, std::vector<float> &featureVector);

/* TASK 3: Multi-Histogram Extraction 
   Input: Image
   Output: Vector of 1536 floats (Top, Bottom, and Mid histograms concatenated)
*/
int extractMultiHistogramFeatures(cv::Mat &src, std::vector<float> &featureVector);

/* TASK 4: Color + Texture Matching 
   Input: Image
   Output: Vector of 512 (Color) + 32 (Texture) = 544 floats
*/
int extractColorTextureFeatures(cv::Mat &src, std::vector<float> &featureVector);

int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);


// DISTANCE METRIC: Histogram Intersection
// Input: Two feature vectors (v1, v2)
// Output: A float representing the similarity/distance
float compareHistogramIntersection(const std::vector<float> &v1, const std::vector<float> &v2);

#endif