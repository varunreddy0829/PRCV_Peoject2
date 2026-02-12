/*
Name: Varun Reddy Patlolla
Project: Computer Vision Image Retrieval (Tasks 1 - 5)
Date: 02/09/2026

PROGRAM DESCRIPTION:
This program is the "Online" search component of the image retrieval system.
It takes a target image, computes its feature vector, and compares it against
all images stored in a pre-computed CSV database using a specified distance
metric. It then ranks and displays the most similar images.

INPUTS (via Command Line):
1. Target Image Path (argv[1]):
   - Path to the image you want to find matches for.
2. Feature Set (argv[2]):
   - "BaselineMatching", "Histogram", "MultiHistogram", "ColorTexture",
"DeepEmbeddings"
3. Distance Metric (argv[3]):
   - "SSD", "Intersection", "MultiIntersection", "Cosine"
4. Number of Outputs (argv[4]):
   - The integer N.
5. Database CSV (argv[5]):
   - Pre-computed features CSV.
*/

#include "csv_util.h"
#include "features.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct Match {
  std::string imagePath;
  float distance;
};

bool compareMatches(const Match &a, const Match &b) {
  return a.distance < b.distance;
}

int main(int argc, char *argv[]) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <TargetImage> <FeatureSet> <DistanceMetric> <NumOutputs> "
                 "<FeatureFile.csv>"
              << std::endl;
    return 1;
  }

  std::string targetPath = argv[1];
  std::string featureSet = argv[2];
  std::string distanceMetric = argv[3];
  int N = std::atoi(argv[4]);
  char *csvFile = argv[5];

  if (featureSet == "CustomTask7" && argc < 7) {
    std::cerr << "Usage for CustomTask7: " << argv[0]
              << " <Target> CustomTask7 <Metric> <N> <ResNetCSV> <MultiHistCSV>"
              << std::endl;
    return 1;
  }

  // 1. Read the database(s)
  std::vector<char *> dbFilenames;
  std::vector<std::vector<float>> dbData;

  std::cout << "Reading " << csvFile << std::endl;
  if (read_image_data_csv(csvFile, dbFilenames, dbData) != 0) {
    std::cerr << "Error reading CSV file " << csvFile << std::endl;
    return -1;
  }

  // For CustomTask7, read the SECOND database (MultiHistogram)
  std::vector<char *> dbFilenames2;
  std::vector<std::vector<float>> dbData2;
  float dnnWeight = 0.5f; // Hardcoded weight

  if (featureSet == "CustomTask7") {
    char *csvFile2 = argv[6];
    std::cout << "Reading " << csvFile2 << std::endl;
    if (read_image_data_csv(csvFile2, dbFilenames2, dbData2) != 0) {
      std::cerr << "Error reading second CSV file " << csvFile2 << std::endl;
      return -1;
    }
    std::cout << "Using DNN Weight: " << dnnWeight
              << " | Color Weight: " << (1.0f - dnnWeight) << std::endl;
  }

  // 2. Compute/Load target features
  std::vector<float> targetFeatures; // Will store DNN features
  std::vector<float>
      targetFeatures2; // Will store Histogram features (for CustomTask7)
  int status = 0;

  if (featureSet == "DeepEmbeddings" || featureSet == "CustomTask7") {
    // Find Target in DNN CSV
    bool found = false;
    std::string targetFilename = targetPath;
    size_t lastSlash = targetPath.find_last_of("/\\");
    if (lastSlash != std::string::npos)
      targetFilename = targetPath.substr(lastSlash + 1);

    for (size_t i = 0; i < dbFilenames.size(); i++) {
      std::string dbName(dbFilenames[i]);
      // Extract filename from db entry if it has path
      std::string dbBase = dbName;
      size_t slash = dbName.find_last_of("/\\");
      if (slash != std::string::npos)
        dbBase = dbName.substr(slash + 1);

      if (dbBase == targetFilename) {
        targetFeatures = dbData[i];
        found = true;
        break;
      }
    }
    if (!found) {
      std::cerr << "Error: Target " << targetPath << " not found in " << csvFile
                << std::endl;
      return -1;
    }

    if (featureSet == "CustomTask7") {
      // Find Target in MultiHistogram CSV
      bool found2 = false;
      for (size_t i = 0; i < dbFilenames2.size(); i++) {
        std::string dbName(dbFilenames2[i]);
        std::string dbBase = dbName;
        size_t slash = dbName.find_last_of("/\\");
        if (slash != std::string::npos)
          dbBase = dbName.substr(slash + 1);

        if (dbBase == targetFilename) {
          targetFeatures2 = dbData2[i];
          found2 = true;
          break;
        }
      }
      if (!found2) {
        std::cerr << "Error: Target " << targetPath
                  << " not found in second CSV (MultiHist)." << std::endl;
        return -1;
      }
    }

  } else {
    cv::Mat targetImg = cv::imread(targetPath);
    if (targetImg.empty()) {
      std::cerr << "Error: Could not load target image " << targetPath
                << std::endl;
      return -1;
    }
    if (featureSet == "BaselineMatching")
      status = extractBaselineFeatures(targetImg, targetFeatures);
    else if (featureSet == "Histogram")
      status = extractHistogramFeatures(targetImg, targetFeatures);
    else if (featureSet == "MultiHistogram")
      status = extractMultiHistogramFeatures(targetImg, targetFeatures);
    else if (featureSet == "ColorTexture")
      status = extractColorTextureFeatures(targetImg, targetFeatures);
    else if (featureSet == "GaborTexture")
      status = extractGaborFeatures(targetImg, targetFeatures);
  }

  // 3. Compute distances
  std::vector<Match> matches;
  for (size_t i = 0; i < dbData.size(); i++) {
    float dist = 0.0;

    if (featureSet == "CustomTask7") {
      // We need to find the matching row in dbData2 (MultiHist DB)
      // Ideally they are in the same order if generated from same dir.
      std::vector<float> currentHistFeatures;
      bool foundHist = false;

      // Optimization: Check if index i matches first
      if (i < dbFilenames2.size()) {
        std::string n1 = dbFilenames[i];
        std::string n2 = dbFilenames2[i];
        if (n1 == n2) {
          currentHistFeatures = dbData2[i];
          foundHist = true;
        }
      }

      if (!foundHist) {
        // Fallback to linear search
        std::string currentName = dbFilenames[i];
        std::string currentBase = currentName;
        size_t slash = currentName.find_last_of("/\\");
        if (slash != std::string::npos)
          currentBase = currentName.substr(slash + 1);

        for (size_t j = 0; j < dbFilenames2.size(); j++) {
          std::string otherName = dbFilenames2[j];
          std::string otherBase = otherName;
          size_t s = otherName.find_last_of("/\\");
          if (s != std::string::npos)
            otherBase = otherName.substr(s + 1);

          if (currentBase == otherBase) {
            currentHistFeatures = dbData2[j];
            foundHist = true;
            break;
          }
        }
      }

      if (!foundHist) {
        dist = 9999.0f; // Image not found in 2nd DB
      } else {
        // 1. "The Gatekeeper": Check Semantic Distance first
        float dnnDist = calculateCosineDistance(targetFeatures, dbData[i]);

        // Threshold: If it's not a car, don't even check color.
        // 0.45 is a reasonable strictness for Deep Embeddings
        if (dnnDist > 0.45f) {
          dist = 1.0f + dnnDist; // Apply penalty, ensure it's > 1.0
        } else {
          // 2. "The Ranker": Check Spatial Color Distance
          // MultiHist has 3 parts: Top, Bottom, Center (512 bins each)
          // We want "Red Car" -> Center and Bottom matters most.
          float rgbDist = 0.0f;
          if (targetFeatures2.size() == 1536 &&
              currentHistFeatures.size() == 1536) {
            int b = 512;
            // Top (0-511), Bottom (512-1023), Center (1024-1535) -> based on
            // extractMultiHistogramFeatures order Wait, let's check
            // features.cpp logic: Top, Bottom, Mid. So: 0-511: Top 512-1023:
            // Bottom 1024-1535: Mid/Center

            std::vector<float> tTop(targetFeatures2.begin(),
                                    targetFeatures2.begin() + b);
            std::vector<float> dbTop(currentHistFeatures.begin(),
                                     currentHistFeatures.begin() + b);

            std::vector<float> tBot(targetFeatures2.begin() + b,
                                    targetFeatures2.begin() + 2 * b);
            std::vector<float> dbBot(currentHistFeatures.begin() + b,
                                     currentHistFeatures.begin() + 2 * b);

            std::vector<float> tMid(targetFeatures2.begin() + 2 * b,
                                    targetFeatures2.end());
            std::vector<float> dbMid(currentHistFeatures.begin() + 2 * b,
                                     currentHistFeatures.end());

            // Weights: Center (0.6), Bottom (0.2), Top (0.2)
            // Cars are mostly in center/bottom. Sky/Background is top.
            float dTop = compareHistogramIntersection(tTop, dbTop);
            float dBot = compareHistogramIntersection(tBot, dbBot);
            float dMid = compareHistogramIntersection(tMid, dbMid);

            rgbDist = (0.2f * dTop) + (0.2f * dBot) + (0.6f * dMid);
          } else {
            // Fallback if not MultiHist
            rgbDist = compareHistogramIntersection(targetFeatures2,
                                                   currentHistFeatures);
          }

          // Final Combination
          dist = (dnnWeight * dnnDist) + ((1.0f - dnnWeight) * rgbDist);
        }
      }
    } else {
      if (distanceMetric == "SSD")
        dist = calculateSSD(targetFeatures, dbData[i]);
      else if (distanceMetric == "Intersection")
        dist = compareHistogramIntersection(targetFeatures, dbData[i]);
      else if (distanceMetric == "Cosine")
        dist = calculateCosineDistance(targetFeatures, dbData[i]);
      else if (distanceMetric == "MultiIntersection") {
        // Task 3 weighting logic
        int b = 512;
        std::vector<float> tT(targetFeatures.begin(),
                              targetFeatures.begin() + b),
            dbT(dbData[i].begin(), dbData[i].begin() + b);
        std::vector<float> tB(targetFeatures.begin() + b,
                              targetFeatures.begin() + 2 * b),
            dbB(dbData[i].begin() + b, dbData[i].begin() + 2 * b);
        std::vector<float> tM(targetFeatures.begin() + 2 * b,
                              targetFeatures.end()),
            dbM(dbData[i].begin() + 2 * b, dbData[i].end());
        dist = (0.3f * compareHistogramIntersection(tT, dbT)) +
               (0.3f * compareHistogramIntersection(tB, dbB)) +
               (0.4f * compareHistogramIntersection(tM, dbM));
      } else if (distanceMetric == "ColorTexture") {
        std::vector<float> tC(targetFeatures.begin(),
                              targetFeatures.begin() + 512),
            dbC(dbData[i].begin(), dbData[i].begin() + 512);
        std::vector<float> tTx(targetFeatures.begin() + 512,
                               targetFeatures.end()),
            dbTx(dbData[i].begin() + 512, dbData[i].end());
        dist = (0.5f * compareHistogramIntersection(tC, dbC)) +
               (0.5f * compareHistogramIntersection(tTx, dbTx));
      } else if (distanceMetric == "GaborTexture") {
        // Use SSD or Cosine. SSD is standard for texture stats.
        // Euclidean distance might be better if normalized, but SSD is fine.
        dist = calculateSSD(targetFeatures,
                            dbData[i]); // Simple SSD for 16D vector
      }
    }
    matches.push_back(Match{std::string(dbFilenames[i]), dist});
  }

  std::sort(matches.begin(), matches.end(), compareMatches);

  // 5. Display Results
  std::cout << "\nTop " << N << " matches for " << targetPath << ":"
            << std::endl;

  for (int i = 0; i < N && i < (int)matches.size(); i++) {
    std::cout << i << ": " << matches[i].imagePath
              << " (Distance: " << matches[i].distance << ")" << std::endl;

    std::string finalPath = matches[i].imagePath;

    // ONLY apply "olympus/" prefix if we are in DeepEmbeddings task
    // AND the path doesn't already contain a slash.
    if ((featureSet == "DeepEmbeddings" || featureSet == "CustomTask7") &&
        finalPath.find('/') == std::string::npos) {
      finalPath = "olympus/" + finalPath;
    }

    cv::Mat matchImg = cv::imread(finalPath);
    if (!matchImg.empty()) {
      std::string winName = "Rank " + std::to_string(i);
      cv::imshow(winName, matchImg);
    } else {
      std::cout << "   [Note: Could not open image at " << finalPath << "]"
                << std::endl;
    }
  }

  // 6. If CustomTask7, also show the WORST matches
  if (featureSet == "CustomTask7") {
    std::cout << "\nWorst " << N << " matches (highest distance):" << std::endl;
    int count = 0;
    for (int i = matches.size() - 1; i >= 0 && count < N; i--) {
      std::cout << count << ": " << matches[i].imagePath
                << " (Distance: " << matches[i].distance << ")" << std::endl;

      // Optionally display them? Maybe just text is enough to not clutter
      // screen. Uncomment below to show windows for worst matches

      std::string finalPath = matches[i].imagePath;
      if (finalPath.find('/') == std::string::npos)
        finalPath = "olympus/" + finalPath;
      cv::Mat badImg = cv::imread(finalPath);
      if (!badImg.empty()) {
        cv::imshow("Worst Rank " + std::to_string(count), badImg);
      }

      count++;
    }
  }

  std::cout << "\nPress any key to exit." << std::endl;
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}