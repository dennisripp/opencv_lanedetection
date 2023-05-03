#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <numeric>


#include "poly.hpp"
#include "lane.hpp"


// Struktur für Debug-Daten
struct DebugData {
    cv::Mat *imageBRG2Gray;
    cv::Mat *imageBirdsEye;
    cv::Mat *imageCanny;
    cv::Mat *imageEdges;
    cv::Mat *slidingWindow;
    std::tuple<std::vector<std::pair<cv::Mat, cv::Mat>>, std::vector<std::pair<cv::Mat, cv::Mat>>> *detectedLanePixels;
    std::pair<cv::Mat, cv::Mat>* polynomalFit;
    std::pair<cv::Mat, cv::Mat>* polynomalFit_rw;
    Direction *curveDirection;
    Direction *laneDirection;
    Curvature *laneCurvature;
    int *polynomialFit2Angle;
    void (*debug)(DebugData *debugData);
};


// volatile button_fsm_t global_buttons = {STATE_BUTTON_IDLE, 0, 0, 0, 0, 0, 0, update_button};

class ImageProcessor {
public:
    // Funktion zum Berechnen des Winkels
    int calculateAngle(const cv::Mat& inputImage, DebugData* debugData = nullptr);

private:
    // Funktion zum Vorverarbeiten des Bildes
    void processImageBRG2Gray(const cv::Mat& inputImage, cv::Mat& outputImage, DebugData* debugData);

    // Funktion zum Vorverarbeiten des Bildes
    void processImage2BirdsEye(const cv::Mat& inputImage, cv::Mat& outputImage, DebugData* debugData);

        // Funktion zum Vorverarbeiten des Bildes
    void processImage2Canny(const cv::Mat& inputImage, cv::Mat& outputImage, DebugData* debugData);

        // Funktion zum Vorverarbeiten des Bildes
    void processImage2Edges(const cv::Mat& inputImage, cv::Mat& outputImage, DebugData* debugData);


    void processImage2DetectedLanePixels(const cv::Mat& dilated_edges, const cv::Mat& warped, std::tuple<std::vector<std::pair<cv::Mat, cv::Mat>>, std::vector<std::pair<cv::Mat, cv::Mat>>>& output , DebugData* debugData);


    void processDetermineLineDirection(std::vector<std::pair<cv::Mat, cv::Mat>>& left_lane, std::vector<std::pair<cv::Mat, cv::Mat>>& right_lane, Direction& output , DebugData *debugData);


    void processDetermineCurveDirection(cv::Mat& left_fit_real_world, cv::Mat& right_fit_real_world, Direction& output, DebugData* debugData);


    void processImage2Polynomial(std::vector<std::pair<cv::Mat, cv::Mat>>& left_lane, std::vector<std::pair<cv::Mat, cv::Mat>>& right_lane, std::pair<cv::Mat, cv::Mat>& output, DebugData* debugData);


    void processImage2PolynomialRealWorld(std::vector<std::pair<cv::Mat, cv::Mat>>& left_lane, std::vector<std::pair<cv::Mat, cv::Mat>>& right_lane, std::pair<cv::Mat, cv::Mat>& output, DebugData* debugData);


    void processFitting2straight_or_curved(const cv::Mat& right_fit, const cv::Mat& left_fit, Curvature& output, DebugData* debugData);


    void processFitting2Angle(const cv::Mat& left_fit, const cv::Mat& right_fit, const cv::Mat& warped, Direction& direction , int& angle, DebugData* debugData);


    static void debug(DebugData *debugData);
};


#endif // IMAGE_PROCESSOR_H
