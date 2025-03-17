#ifndef POST_PROCESSING_HPP
#define POST_PROCESSING_HPP

#include <opencv2/opencv.hpp>

// Kalman 필터 기반 트래킹 보정 알고리즘
cv::Rect2d customAlgorithm(const cv::Mat& frame, const cv::Rect2d& currentBox);

// EKF 변수 전역 선언 (정의는 post_processing.cpp에서 수행)
extern cv::KalmanFilter kf;
extern cv::Mat state;
extern cv::Mat meas;
extern bool isFirstRun;
extern cv::Rect2d prevBox;

#endif // POST_PROCESSING_HPP
