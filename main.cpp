/* Sample Main*/

#include "header/ImageProcessor.hpp"

int main() {
    ImageProcessor processor;

    // Beispielbild laden
    cv::Mat inputImage = cv::imread("frame2.jpg");

    // Debug-Daten Struktur erstellen
    DebugData* debugData  = new DebugData();
    cv::Mat imageAfterGRB2Gray;
    cv::Mat imageCanny;
    cv::Mat imageEdges;
    cv::Mat slidingWindow;
    Direction direction_lane;
    Direction direction_curvature;
    Curvature curv;
    std::pair<cv::Mat, cv::Mat> poly_coeff;
    std::pair<cv::Mat, cv::Mat> poly_coeff_rw;

    debugData->imageBRG2Gray = &imageAfterGRB2Gray;
    debugData->imageCanny = &imageCanny;
    debugData->imageEdges = &imageEdges;
    debugData->polynomalFit = &poly_coeff;
    debugData->slidingWindow = &slidingWindow;
    debugData->laneDirection = &direction_lane;
    debugData->curveDirection = &direction_curvature;
    debugData->laneCurvature = &curv;
    debugData->polynomalFit_rw = &poly_coeff_rw;

    // Winkel berechnen und Debug-Bilder speichern
    int angle = processor.calculateAngle(inputImage, debugData);

    // Ergebnisse ausgeben
    std::cout << "coeff right fit: " << poly_coeff.first << std::endl;
    std::cout << "coeff right rw fit: " << poly_coeff_rw.first << std::endl;
    std::cout << "\n\n" << std::endl;
    std::cout << "coeff left fit: " << poly_coeff.second << std::endl;
    std::cout << "coeff left rw fit: " << poly_coeff_rw.second << std::endl;
    std::cout << "\n\n" << std::endl;
    std::cout << "Winkel: " << angle << std::endl;    
    std::cout << "slope : " << getDirectionString(direction_lane) << std::endl;
    std::cout << "curv dir: " << getDirectionString(direction_curvature) << std::endl;
    std::cout << "curv: " << getCurvatureString(curv) << std::endl;
    cv::imshow("image_after_preprocess.jpg", imageAfterGRB2Gray);
    cv::imshow("image_with_sliding_window.jpg", imageCanny);
    cv::imshow("image_with_fitted_lines.jpg", imageEdges);
    cv::imshow("slidingwindow.jpg", slidingWindow);
    cv::waitKey(0);


    return 0;
}