/*
Name: Varun Reddy Patlolla
Project: Computer Vision Image Retrieval (Task 1, 2, & 3)
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
   - "BaselineMatching": Uses the 7x7 center patch.
   - "Histogram": Uses the 3D RGB color histogram.
   - "MultiHistogram": Uses spatial histograms (Top, Bottom, and 100x100 Mid).
3. Distance Metric (argv[3]):
   - "SSD": Sum of Squared Differences (used for Baseline).
   - "Intersection": Histogram Intersection (used for Histogram).
   - "MultiIntersection": Weighted Intersection (0.3 Top, 0.3 Bottom, 0.4 Mid).
4. Number of Outputs (argv[4]):
   - The integer N (number of top matches to display).
5. Database CSV (argv[5]):
   - The file containing pre-computed features (e.g., HistogramDB.csv).

OUTPUTS:
1. Terminal List: Prints the top N image paths and their calculated distances.
2. Image Windows: Opens OpenCV windows showing the matching images.

DATA FLOW:
Target Image -> Feature Extraction -> Distance Calculation (vs CSV) -> Sorting -> Display Results
*/

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "csv_util.h"
#include "features.h"

// Struct to store match data for sorting
struct Match {
    std::string imagePath;
    float distance;
};

// Comparison function for std::sort (ascending order: smaller distance is better)
bool compareMatches(const Match &a, const Match &b) {
    return a.distance < b.distance;
}

/**
 * calculateSSD
 * Input: Two feature vectors (std::vector<float>)
 * Output: Sum of Squared Differences float
 */
float calculateSSD(const std::vector<float> &v1, const std::vector<float> &v2) {
    float sum = 0.0;
    for (size_t i = 0; i < v1.size(); i++) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sum;
}

int main(int argc, char* argv[]) {
    // Check for correct number of command line arguments
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]  
                  << " <TargetImage> <FeatureSet> <DistanceMetric> <NumOutputs> <FeatureFile.csv>"
                  << std::endl;
        return 1;
    }

    // Parse command line arguments
    std::string targetPath = argv[1];
    std::string featureSet = argv[2];
    std::string distanceMetric = argv[3];
    int N = std::atoi(argv[4]);
    char* csvFile = argv[5];

    // 1. Load target image
    cv::Mat targetImg = cv::imread(targetPath);
    if (targetImg.empty()) {
        std::cerr << "Error: Could not load target image " << targetPath << std::endl;
        return -1;
    }

    // 2. Compute features for the target image based on user selection
    std::vector<float> targetFeatures;
    int status = -1;

    if (featureSet == "BaselineMatching") {
        status = extractBaselineFeatures(targetImg, targetFeatures);
    } 
    else if (featureSet == "Histogram") {
        status = extractHistogramFeatures(targetImg, targetFeatures);
    } 
    // 
    else if (featureSet == "MultiHistogram") {
        status = extractMultiHistogramFeatures(targetImg, targetFeatures);
    }
    else if (featureSet == "ColorTexture") { 
        status = extractColorTextureFeatures(targetImg, targetFeatures); 
    }
    else {
        std::cerr << "Error: Unknown feature set " << featureSet << std::endl;
        return -1;
    }

    if (status != 0) {
        std::cerr << "Error: Feature extraction failed for target image." << std::endl;
        return -1;
    }

    // 3. Read the pre-computed database (CSV)
    std::vector<char *> dbFilenames;
    std::vector<std::vector<float>> dbData;
    if (read_image_data_csv(csvFile, dbFilenames, dbData) != 0) {
        std::cerr << "Error reading CSV file: " << csvFile << std::endl;
        return -1;
    }

    // 4. Compute distances between Target and every image in the Database
    std::vector<Match> matches;
    for (size_t i = 0; i < dbData.size(); i++) {
        float dist = 0.0;
        
        // Dispatcher for Distance Metrics
        if (distanceMetric == "SSD") {
            dist = calculateSSD(targetFeatures, dbData[i]);
        } 
        else if (distanceMetric == "Intersection") {
            // Logic: 1.0 - sum(min(targetBins, dbBins))
            dist = compareHistogramIntersection(targetFeatures, dbData[i]);
        } 
        // 
        else if (distanceMetric == "MultiIntersection") {
            // Each of our 3 regions has 512 bins (8x8x8)
            int binSize = 512;

            // Extract Top histograms from concatenated vectors
            std::vector<float> targetTop(targetFeatures.begin(), targetFeatures.begin() + binSize);
            std::vector<float> dbTop(dbData[i].begin(), dbData[i].begin() + binSize);

            // Extract Bottom histograms
            std::vector<float> targetBot(targetFeatures.begin() + binSize, targetFeatures.begin() + 2 * binSize);
            std::vector<float> dbBot(dbData[i].begin() + binSize, dbData[i].begin() + 2 * binSize);

            // Extract Mid histograms
            std::vector<float> targetMid(targetFeatures.begin() + 2 * binSize, targetFeatures.end());
            std::vector<float> dbMid(dbData[i].begin() + 2 * binSize, dbData[i].end());

            // Compute weighted distance: 0.3(Top) + 0.3(Bottom) + 0.4(Mid)
            float dTop = compareHistogramIntersection(targetTop, dbTop);
            float dBot = compareHistogramIntersection(targetBot, dbBot);
            float dMid = compareHistogramIntersection(targetMid, dbMid);

            dist = (0.3f * dTop) + (0.3f * dBot) + (0.4f * dMid);
        }
        else if (distanceMetric == "ColorTexture") {
            // 1. Separate the features
            std::vector<float> tColor(targetFeatures.begin(), targetFeatures.begin() + 512);
            std::vector<float> tTexture(targetFeatures.begin() + 512, targetFeatures.end());

            std::vector<float> dbColor(dbData[i].begin(), dbData[i].begin() + 512);
            std::vector<float> dbTexture(dbData[i].begin() + 512, dbData[i].end());

            // 2. Calculate individual distances using Intersection
            float dColor = compareHistogramIntersection(tColor, dbColor);
            float dTexture = compareHistogramIntersection(tTexture, dbTexture);

            // 3. Weighted average (50/50)
            dist = (0.5f * dColor) + (0.5f * dTexture);
        }   
        else {
            std::cerr << "Error: Distance metric '" << distanceMetric << "' not implemented." << std::endl;
            return -1;
        }
        
        matches.push_back({std::string(dbFilenames[i]), dist});
    }

    // 5. Sort matches in ascending order (0.0 distance is a perfect match)
    std::sort(matches.begin(), matches.end(), compareMatches);

    // 6. Display results
    std::cout << "\nTop " << N << " matches for " << targetPath << " using " << featureSet << ":" << std::endl;
    
    for (int i = 0; i < N && i < (int)matches.size(); i++) {
        std::cout << i << ": " << matches[i].imagePath << " (Distance: " << matches[i].distance << ")" << std::endl;
        
        // Show matching images in windows
        std::string winName = "Match " + std::to_string(i);
        cv::Mat matchImg = cv::imread(matches[i].imagePath);
        if (!matchImg.empty()) {
            cv::imshow(winName, matchImg);
        }
    }

    std::cout << "\nPress any key in an image window to exit." << std::endl;
    cv::waitKey(0);

    return 0;
}