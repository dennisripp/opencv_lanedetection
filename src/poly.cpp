#include "../header/poly.hpp"

std::vector<double> PolynomialFitting::polyfit_old(
    const std::vector<int>& t,
    const std::vector<int>& v,
    int order
) {
    std::vector<double> t_double(t.begin(), t.end());
    std::vector<double> v_double(v.begin(), v.end());
    std::vector<double> coeff;

    // Create Matrix Placeholder of size n x k, n= number of datapoints, k = order of polynomial, for exame k = 3 for cubic polynomial
    Eigen::MatrixXd T(t.size(), order + 1);
    Eigen::VectorXd V = Eigen::VectorXd::Map(&v_double.front(), v_double.size());
    Eigen::VectorXd result;

    // check to make sure inputs are correct
    assert(t.size() == v.size());
    assert(t.size() >= order + 1);

    // Populate the matrix
    for (size_t i = 0; i < t.size(); ++i)
    {
        for (size_t j = 0; j < order + 1; ++j)
        {
            T(i, j) = std::pow(t_double.at(i), j);
        }
    }

    // Solve for linear least square fit
    result = solve(T, V);

    coeff.resize(order + 1);
    for (int k = 0; k < order + 1; k++)
    {
        coeff[k] = result[k];
    }

    return coeff;
}

// cv::Mat PolynomialFitting::polyfit(const cv::Mat& src_x, const cv::Mat& src_y, int order) {
//     int npoints = src_x.rows;
//     int ncols = order + 1;

//     Eigen::MatrixXd A = Eigen::MatrixXd::Ones(npoints, ncols);
//     Eigen::VectorXd y = Eigen::VectorXd::Map(src_y.ptr<double>(), npoints);
//     Eigen::VectorXd x = Eigen::VectorXd::Map(src_x.ptr<double>(), npoints);

//     for (int j = 1; j < ncols; ++j) {
//         A.col(j) = A.col(j - 1).cwiseProduct(x);
//     }

//     Eigen::VectorXd coeffs = A.householderQr().solve(y);
//     cv::Mat coeffs_cv(coeffs.rows(), 1, CV_64F);
//     for (int i = 0; i < coeffs.rows(); ++i) {
//         coeffs_cv.at<double>(i) = coeffs[i];
//         std::cout << "coeffs :"<< coeffs[i] << std::endl;

//     }

//     return coeffs_cv;
// }

cv::Mat PolynomialFitting::polyfit(const cv::Mat& src_y, const cv::Mat& src_x, int order) {
    assert(src_x.rows == src_y.rows);
    assert(src_x.rows >= order + 1);

    std::vector<double> t_double, v_double;

    src_x.copyTo(t_double);
    src_y.copyTo(v_double);

    Eigen::MatrixXd T(src_x.rows, order + 1);
    Eigen::VectorXd V = Eigen::VectorXd::Map(&v_double.front(), v_double.size());
    Eigen::VectorXd result;

    for (size_t i = 0; i < src_x.rows; ++i) {
        for (size_t j = 0; j < order + 1; ++j) {
            T(i, j) = std::pow(t_double.at(i), j);
        }
    }

    result = solve(T, V);

    cv::Mat coeff(order + 1, 1, CV_64F);
    for (int k = 0; k < order + 1; k++) {
        coeff.at<double>(k, 0) = result[k];
    }

    // Reverse elements of coeff
    int num_elements = coeff.rows;
    for (int i = 0; i < num_elements / 2; ++i) {
        double temp = coeff.at<double>(i, 0);
        coeff.at<double>(i, 0) = coeff.at<double>(num_elements - 1 - i, 0);
        coeff.at<double>(num_elements - 1 - i, 0) = temp;
    }

    return coeff;
}


Eigen::VectorXd PolynomialFitting::solve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b)
{
    return A.householderQr().solve(b);
}


cv::Mat PolynomialFitting::polyder(cv::Mat& p, int m = 1) {
    if (m < 0) {
        std::cerr << "Order of derivative must be positive (see polyint)\n";
        return cv::Mat();
    }

    if (m >= p.cols) {
        // The derivative order is higher than or equal to the polynomial degree,
        // resulting in a zero polynomial
        return cv::Mat::zeros(1, 1, CV_64F);
    }

    int n = p.cols - 1;
    cv::Mat y = cv::Mat::zeros(1, n, CV_64F);
    for (int i = 0; i < n; ++i) {
        y.at<double>(0, i) = p.at<double>(0, i) * (n - i);
    }

    if (m == 0) {
        return p;
    } else {
        return polyder(y, m - 1);
    }
}

