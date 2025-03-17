#include "tracking.hpp"

ObjectTracker::ObjectTracker() {
    tracker = nullptr; // 초기에는 null
    tracker = cv::TrackerMIL::create(); // ✅ KCF 대신 MIL 사용
}

void ObjectTracker::init(cv::Mat& frame, cv::Rect2d& boundingBox) {
    tracker->init(frame, boundingBox);
}

bool ObjectTracker::update(cv::Mat& frame, cv::Rect2d& boundingBox) {
    if (!tracker) return false;

    cv::Rect boundingBoxInt = boundingBox; // ✅ cv::Rect2d → cv::Rect 변환
    bool success = tracker->update(frame, boundingBoxInt);

    if (success) {
        boundingBox = boundingBoxInt; // ✅ 변환된 boundingBoxInt 값을 boundingBox에 반영
    }

    return success;
}
