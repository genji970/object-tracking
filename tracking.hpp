#ifndef TRACKING_HPP
#define TRACKING_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>  // 🔹 OpenCV Tracking 모듈 추가
#include <opencv2/core.hpp>      // 🔹 cv::Rect2d을 위해 core 모듈 추가

class ObjectTracker {
public:
    ObjectTracker();
    void init(cv::Mat& frame, cv::Rect2d& boundingBox);
    bool update(cv::Mat& frame, cv::Rect2d& boundingBox);

private:
    cv::Ptr<cv::Tracker> tracker;
};

#endif // TRACKING_HPP
