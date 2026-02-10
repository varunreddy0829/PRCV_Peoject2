/*
Name: Varun Reddy Patlolla
Project: Computer Vision Image Retrieval 
Date: 02/09/2026

PROGRAM DESCRIPTION:
This program performs "Feature Extraction" on a bulk set of images. It iterates 
through every image in a specified directory, computes a numerical "fingerprint" 
(feature vector) based on the chosen method, and saves these results into a 
CSV file. This creates a searchable database for the Matcher program.

INPUTS (via Command Line):
1. Directory Path (argv[1]): 
   - A string path to the folder containing the images.
2. Feature Set (argv[2]):
   - "BaselineMatching": Computes a 7x7 RGB patch from the center (147 values).
   - "Histogram": Computes a 3D RGB color histogram (512 values).
3. Output CSV Filename (argv[3]):
   - The name of the file to create/update .

OUTPUTS:
1. A CSV File: 
   - Each row represents one image.
   - Column 1: The string path/filename of the image.
   - Columns 2+: The floating-point values representing the features.
2. Terminal Progress:
   - Prints the total count of successfully processed images.

DATA FLOW:

*/

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "csv_util.h"
#include "features.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // Usage: ./FeatureFile <Dir> <FeatureType> <OutputCSV>
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <Directory> <FeatureType> <OutputCSV>" << std::endl;
        std::cerr << "Available Types: BaselineMatching, Histogram (TBD)" << std::endl;
        return -1;
    }

    std::string dirPath = argv[1];
    std::string featureType = argv[2];
    char* csvFilename = argv[3];

    bool resetFile = true;
    int count = 0;

    // Check if directory exists
    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        std::cerr << "Error: Directory " << dirPath << " does not exist." << std::endl;
        return -1;
    }

    for (const auto & entry : fs::directory_iterator(dirPath)) {
        std::string path = entry.path().string();
        cv::Mat img = cv::imread(path);
        
        if (img.empty()) continue;

        std::vector<float> features;
        int status = -1;

        // --- Dispatcher Logic ---
        // Changed to extractBaselineFeatures to match features.h
        if (featureType == "BaselineMatching") {
            status = extractBaselineFeatures(img, features);
        } 
        else if (featureType == "Histogram") {
            // This now calls your 3D 8-bin logic
            status = extractHistogramFeatures(img, features);
        }
        else if (featureType == "MultiHistogram") {
            // This now calls your 3D 8-bin logic
            status = extractMultiHistogramFeatures(img, features);
        }
        else if (featureType == "ColorTexture") { // Added this branch
            status = extractColorTextureFeatures(img, features);
        }
        else {
            std::cerr << "Unknown feature type: " << featureType << std::endl;
            return -1;
        }

        if (status == 0) {
            char* pathChar = new char[path.length() + 1];
            strcpy(pathChar, path.c_str());
            
            append_image_data_csv(csvFilename, pathChar, features, resetFile);
            
            resetFile = false;
            delete[] pathChar;
            count++;
        }
    }

    std::cout << "Done! Processed " << count << " images using " << featureType << std::endl;
    return 0;
}