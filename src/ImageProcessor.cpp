#include "../header/ImageProcessor.hpp"

    // Funktion zum Berechnen des Winkels
    int ImageProcessor::calculateAngle(const cv::Mat& inputImage, DebugData* debugData) {        
        cv::Mat bgr2grayImage;
        processImageBRG2Gray(inputImage, bgr2grayImage, debugData);

        cv::Mat birdeyeImage;
        processImage2BirdsEye(bgr2grayImage, birdeyeImage, debugData);

        cv::Mat cannyImage;
        processImage2Canny(birdeyeImage, cannyImage, debugData);

        cv::Mat edgeImage;
        processImage2Edges(cannyImage, edgeImage, debugData);
        std::tuple<std::vector<std::pair<cv::Mat, cv::Mat>>, std::vector<std::pair<cv::Mat, cv::Mat>>> lane_tuple;
        std::vector<std::pair<cv::Mat, cv::Mat>> left_lane;
        std::vector<std::pair<cv::Mat, cv::Mat>> right_lane;
        processImage2DetectedLanePixels(edgeImage, birdeyeImage, lane_tuple, debugData);
        left_lane = std::get<0>(lane_tuple);
        right_lane = std::get<1>(lane_tuple);

        std::pair<cv::Mat, cv::Mat> poly_coeffs;
        cv::Mat right_fit;
        cv::Mat left_fit;
        processImage2Polynomial(left_lane, right_lane, poly_coeffs, debugData);
        right_fit = poly_coeffs.first;
        left_fit = poly_coeffs.second;

        Curvature curvature;
        processFitting2straight_or_curved(right_fit, left_fit, curvature, debugData);

        std::pair<cv::Mat, cv::Mat> poly_coeffs_rw;
        cv::Mat right_fit_rw;
        cv::Mat left_fit_rw;
        processImage2PolynomialRealWorld(left_lane, right_lane, poly_coeffs_rw, debugData);
        right_fit_rw = poly_coeffs_rw.first;
        left_fit_rw = poly_coeffs_rw.second;

        Direction direction;
        int angle = 0;

        switch(curvature) {
            case Curvature::STRAIGHT:
                processDetermineLineDirection(left_lane, right_lane, direction, debugData);
                processFitting2Angle(left_fit, right_fit, birdeyeImage, direction, angle, debugData);
            break;
            case Curvature::CURVED:
                processDetermineCurveDirection(left_fit_rw, right_fit_rw, direction, debugData);
                processFitting2Angle(left_fit, right_fit, birdeyeImage, direction, angle, debugData);
                break;

            case Curvature::UNDETERMINED:

            break;

            default:
            break;

        }


        if (debugData) {
            debugData->debug = &ImageProcessor::debug;
            debugData->debug(debugData);    
        }

        return angle;
    }


    void ImageProcessor::processImageBRG2Gray(const cv::Mat& inputImage, cv::Mat& outputImage, DebugData* debugData) {
        cv::cvtColor(inputImage, outputImage, cv::COLOR_BGR2GRAY);

        if (debugData && debugData->imageBRG2Gray) {
            *debugData->imageBRG2Gray = outputImage.clone();
        }
    }


    // Funktion zum Vorverarbeiten des Bildes
    void ImageProcessor::processImage2BirdsEye(const cv::Mat& inputImage, cv::Mat& outputImage, DebugData* debugData) {
        int width = inputImage.cols;
        int height = inputImage.rows;
        cv::Point2f source[] = {cv::Point2f(20, 460), cv::Point2f(140, 240), cv::Point2f(500, 240), cv::Point2f(620, 460)};
        cv::Point2f destination[] = {cv::Point2f(0, 480), cv::Point2f(0, 0), cv::Point2f(640, 0), cv::Point2f(640, 480)};

        cv::Mat matrix = cv::getPerspectiveTransform(source, destination);
        cv::warpPerspective(inputImage, outputImage, matrix, cv::Size(width, height), cv::INTER_LINEAR);


        if (debugData && debugData->imageBirdsEye) {
            *debugData->imageBirdsEye = outputImage.clone();
        }
    }


        // Funktion zum Vorverarbeiten des Bildes
    void ImageProcessor::processImage2Canny(const cv::Mat& inputImage, cv::Mat& outputImage, DebugData* debugData) {
        int threshold_low = 10;
        int threshold_high = 100;
        cv::Mat image_copy = inputImage.clone();
        cv::Mat image_blurred;
        cv::GaussianBlur(image_copy, image_blurred, cv::Size(7, 7), 0);
        cv::Canny(image_blurred, outputImage, threshold_low, threshold_high);


        if (debugData && debugData->imageCanny) {
            *debugData->imageCanny = outputImage.clone();
        }
    }


        // Funktion zum Vorverarbeiten des Bildes
    void ImageProcessor::processImage2Edges(const cv::Mat& inputImage, cv::Mat& outputImage, DebugData* debugData) {
        cv::Mat image = inputImage.clone();
        cv::Mat edges;
        cv::Canny(image, edges, 50, 200);

        // Define kernel for dilation operation
        cv::Mat kernel = cv::Mat::ones(5, 5, CV_8U);

        // Apply dilation on the edges
        cv::dilate(edges, outputImage, kernel, cv::Point(-1, -1), 1);

        if (debugData && debugData->imageEdges) {
            *debugData->imageEdges = outputImage.clone();
        }
    }

        
    void ImageProcessor::processImage2DetectedLanePixels(const cv::Mat& dilated_edges, const cv::Mat& warped, std::tuple<std::vector<std::pair<cv::Mat, cv::Mat>>, std::vector<std::pair<cv::Mat, cv::Mat>>>& output , DebugData* debugData) {
        cv::Mat thresh = dilated_edges.clone();
        int nwindows = 10;
        int window_length = 70;
        int minpix = 20;
        int window_height = static_cast<int>(thresh.rows / nwindows);

        // cv::Mat bottom_fourth(thresh, bottom_fourth_rect);
        cv::Mat bottom_fourth = thresh(cv::Range(thresh.rows * 7 / 8, thresh.rows), cv::Range::all());

        //cv::Mat bottom_half = thresh(cv::Range(thresh.rows / 2, thresh.rows), cv::Range::all());
        cv::Mat histogram;
        cv::reduce(bottom_fourth, histogram, 0, cv::REDUCE_SUM, CV_32S);

        cv::Mat out_img = warped.clone();

        std::vector<cv::Point> nonzero;
        cv::findNonZero(thresh, nonzero);
        std::vector<int> nonzeroy(nonzero.size()), nonzerox(nonzero.size());
        for (size_t i = 0; i < nonzero.size(); i++) {
            nonzeroy[i] = nonzero[i].y;
            nonzerox[i] = nonzero[i].x;
        }

        int midpoint = histogram.cols / 2;
        cv::Mat left_histogram = histogram.colRange(0, midpoint);
        cv::Mat right_histogram = histogram.colRange(midpoint, histogram.cols);

        int threshold = 100;
        cv::Point left_peak, right_peak;
        cv::minMaxLoc(left_histogram, nullptr, nullptr, nullptr, &left_peak);
        cv::minMaxLoc(right_histogram, nullptr, nullptr, nullptr, &right_peak);
        right_peak.x += midpoint;

        cv::Mat leftx_peak_mask = left_histogram > threshold;
        cv::Mat rightx_peak_mask = right_histogram > threshold;

        int leftx_current = -1;
        int rightx_current = -1;

        if (cv::countNonZero(leftx_peak_mask) > 0) {
            leftx_current = left_peak.x;
        }

            if (cv::countNonZero(rightx_peak_mask) > 0) {
            rightx_current = right_peak.x;
        }

        std::vector<std::vector<int>> left_lane_inds, right_lane_inds;

        for (int window = 0; window < nwindows; window++) {
            int win_y_low = thresh.rows - (window + 1) * window_height;
            int win_y_high = thresh.rows - window * window_height;

            if (leftx_current != -1) {
                int win_xleft_low = leftx_current - window_length;
                int win_xleft_high = leftx_current + window_length;

                cv::rectangle(out_img, cv::Point(win_xleft_low, win_y_low), cv::Point(win_xleft_high, win_y_high), cv::Scalar(0, 255, 0), 4);

                std::vector<int> left_lane;
                for (size_t i = 0; i < nonzero.size(); i++) {
                    if (nonzeroy[i] >= win_y_low && nonzeroy[i] < win_y_high && nonzerox[i] >= win_xleft_low && nonzerox[i] < win_xleft_high) {
                        left_lane.push_back(i);
                    }
                }

                left_lane_inds.push_back(left_lane);

                if (left_lane.size() > minpix) {
                    int sum = 0;
                    for (int idx : left_lane) {
                        sum += nonzerox[idx];
                    }
                    leftx_current = static_cast<int>(sum / left_lane.size());
                }
            }

            if (rightx_current != -1) {
                int win_xright_low = rightx_current - window_length;
                int win_xright_high = rightx_current + window_length;

                cv::rectangle(out_img, cv::Point(win_xright_low, win_y_low), cv::Point(win_xright_high, win_y_high), cv::Scalar(0, 255, 0), 4);

                std::vector<int> right_lane;
                for (size_t i = 0; i < nonzero.size(); i++) {
                    if (nonzeroy[i] >= win_y_low && nonzeroy[i] < win_y_high && nonzerox[i] >= win_xright_low && nonzerox[i] < win_xright_high) {
                        right_lane.push_back(i);
                    }
                }

                right_lane_inds.push_back(right_lane);

                if (right_lane.size() > minpix) {
                    int sum = 0;
                    for (int idx : right_lane) {
                        sum += nonzerox[idx];
                    }
                    rightx_current = static_cast<int>(sum / right_lane.size());
                }
            }
        }

        std::vector<int> leftlane_x, leftlane_y, rightlane_x, rightlane_y;
        for (const auto& lane : left_lane_inds) {
            for (int idx : lane) {
                leftlane_x.push_back(nonzerox[idx]);
                leftlane_y.push_back(nonzeroy[idx]);
            }
        }
        for (const auto& lane : right_lane_inds) {
            for (int idx : lane) {
                rightlane_x.push_back(nonzerox[idx]);
                rightlane_y.push_back(nonzeroy[idx]);
            }
        }

        std::vector<std::pair<cv::Mat, cv::Mat>> left_lane, right_lane;

        for (const auto& lane : left_lane_inds) {
            for (int idx : lane) {
                left_lane.emplace_back(cv::Mat(1, 1, CV_32S, cv::Scalar(nonzeroy[idx])), cv::Mat(1, 1, CV_32S, cv::Scalar(nonzerox[idx])));
            }
        }

        for (const auto& lane : right_lane_inds) {
            for (int idx : lane) {
                right_lane.emplace_back(cv::Mat(1, 1, CV_32S, cv::Scalar(nonzeroy[idx])), cv::Mat(1, 1, CV_32S, cv::Scalar(nonzerox[idx])));
            }
        }

        output = std::make_tuple(left_lane, right_lane);

        if (debugData && (debugData->detectedLanePixels != nullptr)) {
            *debugData->detectedLanePixels = output;
        }

        if (debugData && (debugData->slidingWindow != nullptr)) {
            *debugData->slidingWindow = out_img;
        }
    }

    void ImageProcessor::processImage2Polynomial(std::vector<std::pair<cv::Mat, cv::Mat>>& left_lane, std::vector<std::pair<cv::Mat, cv::Mat>>& right_lane, std::pair<cv::Mat, cv::Mat>& output, DebugData* debugData) {
        cv::Mat right_fit, left_fit;
        PolynomialFitting poly;

        if (!left_lane.empty()) {
            cv::Mat leftlane_x, leftlane_y;
            for (const auto& left : left_lane) {
                leftlane_x.push_back(left.first);
                leftlane_y.push_back(left.second);
            }        

            left_fit = poly.polyfit(leftlane_y, leftlane_x, 2);
        }

        if (!right_lane.empty()) {
            cv::Mat rightlane_x, rightlane_y;
            for (const auto& right : right_lane) {
                rightlane_x.push_back(right.first);
                rightlane_y.push_back(right.second);
            }
            right_fit = poly.polyfit(rightlane_y, rightlane_x, 2);
        }

        output = std::make_pair(right_fit, left_fit);


        if (debugData && (debugData->polynomalFit != nullptr)) {
            *debugData->polynomalFit = std::make_pair(right_fit, left_fit);
        }
    }

    void ImageProcessor::processImage2PolynomialRealWorld(std::vector<std::pair<cv::Mat, cv::Mat>>& left_lane, std::vector<std::pair<cv::Mat, cv::Mat>>& right_lane, std::pair<cv::Mat, cv::Mat>& output, DebugData* debugData) {
        cv::Mat right_fit, left_fit;
        PolynomialFitting poly;

        double ym_per_pix = 1.0 / 600.0;
        double xm_per_pix = 1.0 / 600.0;


        if (!left_lane.empty()) {
            cv::Mat leftlane_x, leftlane_y;
            for (const auto& left : left_lane) {
                leftlane_x.push_back(left.first);
                leftlane_y.push_back(left.second);
            }

            leftlane_y.convertTo(leftlane_y, CV_64F);
            leftlane_x.convertTo(leftlane_x, CV_64F);

            leftlane_y *= ym_per_pix;
            leftlane_x *= xm_per_pix;


            left_fit = poly.polyfit(leftlane_y, leftlane_x, 2);
        }

        if (!right_lane.empty()) {
            cv::Mat rightlane_x, rightlane_y;
            for (const auto& right : right_lane) {
                rightlane_x.push_back(right.first);
                rightlane_y.push_back(right.second);
            }

            rightlane_y.convertTo(rightlane_y, CV_64F);
            rightlane_x.convertTo(rightlane_x, CV_64F);

            rightlane_x *= xm_per_pix;
            rightlane_y *= ym_per_pix;

            right_fit = poly.polyfit(rightlane_y, rightlane_x, 2);
        }

        output = std::make_pair(right_fit, left_fit);


        if (debugData && (debugData->polynomalFit_rw != nullptr)) {
            *debugData->polynomalFit_rw = std::make_pair(right_fit, left_fit);
        }
    }

    void ImageProcessor::processDetermineLineDirection(std::vector<std::pair<cv::Mat, cv::Mat>>& left_lane, std::vector<std::pair<cv::Mat, cv::Mat>>& right_lane, Direction& output , DebugData *debugData) {
        PolynomialFitting poly;
        cv::Mat slope_fit;
        double ym_per_pix = 1.0 / 600.0;
        double xm_per_pix = 1.0 / 600.0;
        Direction direction = Direction::UNDETERMINED;
        double slope_coeff = 0.0;
        PolynomialFitting polyfit_obj; // Assuming you have PolynomialFitting class with polyfit method
        cv::Mat left_slope_fit, right_slope_fit;

        // Calculate the slope coefficient for left lane
        if (!left_lane.empty()) {
            cv::Mat leftlane_x, leftlane_y;
            for (auto& pair : left_lane) {
                leftlane_x.push_back(pair.first);
                leftlane_y.push_back(pair.second);
            }
            leftlane_y.convertTo(leftlane_y, CV_64F);
            leftlane_x.convertTo(leftlane_x, CV_64F);
            cv::Mat slopex = leftlane_x * xm_per_pix;
            cv::Mat slopey = leftlane_y * ym_per_pix;


            left_slope_fit = polyfit_obj.polyfit(slopey, slopex, 1);
            slope_coeff = left_slope_fit.at<double>(0);

        }

        // Calculate the slope coefficient for right lane
        if (!right_lane.empty()) {
            cv::Mat rightlane_x, rightlane_y;
            for (auto& pair : right_lane) {
                rightlane_x.push_back(pair.first);
                rightlane_y.push_back(pair.second);
            }

            rightlane_x.convertTo(rightlane_x, CV_64F);
            rightlane_y.convertTo(rightlane_y, CV_64F);
            cv::Mat slopex = rightlane_x * xm_per_pix;
            cv::Mat slopey = rightlane_y * ym_per_pix;


            right_slope_fit = polyfit_obj.polyfit(slopey, slopex, 1);
            slope_coeff = right_slope_fit.at<double>(0);
        }

            // If both lanes are not empty, calculate the average slope coefficient
        if (!left_lane.empty() && !right_lane.empty()) {
            cv::Mat average_slope_fit = (right_slope_fit + left_slope_fit) / 2;
            slope_coeff = average_slope_fit.at<double>(0);
        }

        // If both lanes are empty, return UNDETERMINED
        else if (left_lane.empty() && right_lane.empty()) {
                direction = Direction::UNDETERMINED;
            }

        std::cout << "slope coeff " << slope_coeff << std::endl;


        if (slope_coeff < 0) {
            direction = Direction::RIGHT;
        } else if (slope_coeff > 0) {
            direction = Direction::LEFT;
        } else {
        }

        output = direction;

        if (debugData && debugData->laneDirection) {
            *debugData->laneDirection = direction;
        }

    }


    Direction getDirection(const cv::Mat &coeffs) {
        if (coeffs.cols == 0) {
            return Direction::UNDETERMINED;
        }

        if (coeffs.at<double>(0, 0) > 0) {
            return Direction::RIGHT;
        } 
        else if (coeffs.at<double>(0, 0) < 0) {
            return Direction::LEFT;
        } 

        return Direction::UNDETERMINED;
    }


    void ImageProcessor::processDetermineCurveDirection(cv::Mat& left_fit_real_world, cv::Mat& right_fit_real_world, Direction& output, DebugData* debugData) {
            PolynomialFitting poly;
            
            cv::Mat left_coeffs = left_fit_real_world.clone();
            cv::Mat right_coeffs = right_fit_real_world.clone();

            if (!right_coeffs.empty() && !left_coeffs.empty()) {
                cv::Mat average_coeffs = right_coeffs + left_coeffs / 2.0;
                cv::Mat curvedirectionCoeff = poly.polyder(average_coeffs, 2);
                output = getDirection(curvedirectionCoeff);
            }
            else if (!right_coeffs.empty()) {
                cv::Mat right_t = right_coeffs.t();
                cv::Mat  curvedirectionCoeff = poly.polyder(right_t, 2);
                output = getDirection(curvedirectionCoeff);
            }
            
            else if (!left_coeffs.empty()) {
                
                cv::Mat left_t = left_coeffs.t();
                cv::Mat curvedirectionCoeff = poly.polyder(left_t, 2);
                output = getDirection(curvedirectionCoeff);
            }

            else {
                output = Direction::UNDETERMINED;
            }

        if (debugData && (debugData->curveDirection != nullptr)) {
            *debugData->curveDirection = output;
        }
    }


    void ImageProcessor::processFitting2straight_or_curved(const cv::Mat& right_fit, const cv::Mat& left_fit, Curvature& output, DebugData* debugData)  {
        cv::Mat right_first = right_fit.clone();
        cv::Mat  left_first = left_fit.clone();

        if (!right_first.empty() && !left_first.empty()) {
            cv::Mat average_fit = left_first + right_first / 2;
            if (std::abs(average_fit.at<double>(0,0)) < 0.001) {
                output = Curvature::STRAIGHT;
            } else {
                output = Curvature::CURVED;
            }
        
        } else if (!right_fit.empty() && std::abs(right_first.at<double>(0,0)) < 0.001) {
            output = Curvature::STRAIGHT;
            
        } else if (!left_first.empty() && std::abs(left_first.at<double>(0,0)) < 0.001) {         
            output = Curvature::STRAIGHT;
        } else {
            output = Curvature::CURVED;
        }

        if (debugData && (debugData->laneCurvature != nullptr)) {
            *debugData->laneCurvature = output;
        }
    }


    template<typename T>
    const T& clamp(const T& value, const T& min, const T& max) {
        return std::max(min, std::min(value, max));
    }


    void ImageProcessor::processFitting2Angle(const cv::Mat& left_fit, const cv::Mat& right_fit,
                                const cv::Mat& warped, Direction& direction , int& angle, DebugData* debugData) {
        
        double ym_per_pix = 1.0 / 600.0;
        double xm_per_pix = 1.0 / 600.0;
        int lane_width_px = 300;
        int image_height = warped.rows;
        int image_middle = warped.cols / 2;
        int steering_angle = 0;
        
        cv::Mat left_fit_temp, right_fit_temp;
        
        // Check if left_fit and right_fit are empty, if so, return 0
        if (left_fit.empty() && right_fit.empty()) {
            return;
        }

        // Check if left_fit is empty, if so, define it
        if (left_fit.empty()) {
            left_fit_temp = (cv::Mat_<double>(3,1) << right_fit.at<double>(0,0), 
                                                    right_fit.at<double>(1,0), 
                                                    right_fit.at<double>(2,0) - lane_width_px);
        } else {
            left_fit_temp = left_fit.clone();
        }
        
        // Check if right_fit is empty, if so, define it
        if (right_fit.empty()) {
            right_fit_temp = (cv::Mat_<double>(3,1) << left_fit_temp.at<double>(0,0), 
                                                    left_fit_temp.at<double>(1,0), 
                                                    left_fit_temp.at<double>(2,0) + lane_width_px);
        } else {
            right_fit_temp = right_fit.clone();
        }

        // Calculate x coordinates of the lane line
        double bottom_left = left_fit_temp.at<double>(0,0) * pow(image_height, 2) + left_fit_temp.at<double>(1,0) * image_height + left_fit_temp.at<double>(2,0);
        double bottom_right = right_fit_temp.at<double>(0,0) * pow(image_height, 2) + right_fit_temp.at<double>(1,0) * image_height + right_fit_temp.at<double>(2,0);

        double lane_centre_px = (bottom_right + bottom_left) / 2;

        // Calculate offset and convert to meters
        double centre_offset_pixels = abs(lane_centre_px - image_middle);
        double centre_offset_mtrs = xm_per_pix * centre_offset_pixels;

        // Calculate steering angle
        double angle_to_vertical_line_rad = atan(centre_offset_mtrs / ((image_height) * ym_per_pix));
        double angle_to_vertical_line_deg = static_cast<double>(angle_to_vertical_line_rad * 180.0 / CV_PI);

        if (direction == Direction::LEFT) {
            angle_to_vertical_line_deg =  -angle_to_vertical_line_deg;
        } else if (direction == Direction::UNDETERMINED) {
            steering_angle = 0;
        }

        const double MIN_STEERING_ANGLE = -45.0;
        const double MAX_STEERING_ANGLE = 45.0;
        steering_angle = clamp(angle_to_vertical_line_deg, MIN_STEERING_ANGLE, MAX_STEERING_ANGLE);


        if (debugData && (debugData->polynomialFit2Angle != nullptr)) {
            *debugData->polynomialFit2Angle = steering_angle;
        }

        angle = (int)steering_angle;
    }
    

    
    void ImageProcessor::debug(DebugData *debugData) {
        // hier bilder, winkel, coeffs ausgeben, bearbeiten etc..

    }