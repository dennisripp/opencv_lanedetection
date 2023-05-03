#ifndef POLY_HPP
#define POLY_HPP

#include <vector>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <cmath>
#include <cassert>
#include <iostream>
#include <opencv2/core.hpp>

class PolynomialFitting {
public:
    PolynomialFitting() = default;
    ~PolynomialFitting() = default;

    std::vector<double> polyfit_old(const std::vector<int>& t, const std::vector<int>& v, int order);
    cv::Mat polyfit(const cv::Mat& src_x, const cv::Mat& src_y, int order);
    cv::Mat polyder(cv::Mat& p, int m);

private:
    Eigen::VectorXd solve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
};

#endif /* POLY_HPP */
