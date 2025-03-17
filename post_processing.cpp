#include "post_processing.hpp"
#include <cmath>

// 🔹 EKF 변수 정의 (중복 정의 방지)
cv::KalmanFilter kf(4, 2, 0);
cv::Mat state(4, 1, CV_32F);
cv::Mat meas(2, 1, CV_32F);
bool isFirstRun = true;
cv::Rect2d prevBox;

// 🔹 비선형 상태 전이 함수 (EKF 적용)
cv::Mat transitionFunction(const cv::Mat& x) {
    cv::Mat f_x(4, 1, CV_32F);

    float dt = 0.2f;    // 프레임 간 시간 간격
    float damping = 0.9f; // 속도 감쇠 계수

    f_x.at<float>(0) = x.at<float>(0) + x.at<float>(2) * dt;  // x' = x + dx * dt
    f_x.at<float>(1) = x.at<float>(1) + x.at<float>(3) * dt;  // y' = y + dy * dt
    f_x.at<float>(2) = x.at<float>(2) * damping;  // dx' = dx * damping
    f_x.at<float>(3) = x.at<float>(3) * damping;  // dy' = dy * damping

    return f_x;
}

// 🔹 EKF 기반 객체 추적 함수
cv::Rect2d customAlgorithm(const cv::Mat& frame, const cv::Rect2d& currentBox) {
    if (isFirstRun) {
        // 초기 박스가 비어 있는 경우 초기화 방지
        if (currentBox.width == 0 || currentBox.height == 0) {
            return currentBox;
        }

        // 🔹 칼만 필터 초기화 (최초 1회 실행)
        kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);

        kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0);

        kf.processNoiseCov = (cv::Mat_<float>(4, 4) <<
            1e-1, 0, 0, 0,
            0, 1e-1, 0, 0,
            0, 0, 5e-2, 0,
            0, 0, 0, 5e-2);

        kf.measurementNoiseCov = (cv::Mat_<float>(2, 2) <<
            5e-2, 0,
            0, 5e-2);

        kf.errorCovPost = (cv::Mat_<float>(4, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);

        // 초기 상태 설정
        state.at<float>(0) = currentBox.x;
        state.at<float>(1) = currentBox.y;
        state.at<float>(2) = 0; // 초기 속도
        state.at<float>(3) = 0;

        kf.statePost = state;
        prevBox = currentBox;
        isFirstRun = false;
    }

    // 🔹 1️⃣ 예측 단계 (Predict)
    state = kf.predict(); // 칼만 필터 예측
    kf.statePre = transitionFunction(state); // EKF 전이 함수 적용 (이전 값 덮어쓰기 방지)

    double predictedX = kf.statePre.at<float>(0);
    double predictedY = kf.statePre.at<float>(1);

    // 🔹 2️⃣ 측정값 갱신 (Update)
    if (currentBox.width == 0 || currentBox.height == 0) {
        // 객체 감지가 실패한 경우 예측값 유지
        meas.at<float>(0) = predictedX;
        meas.at<float>(1) = predictedY;
    }
    else {
        meas.at<float>(0) = currentBox.x;
        meas.at<float>(1) = currentBox.y;
    }

    cv::Mat correctedState = kf.correct(meas); // EKF 보정

    // 🔹 3️⃣ 속도 업데이트 (보정된 속도 반영)
    float alpha = 0.8f;
    correctedState.at<float>(2) = alpha * (meas.at<float>(0) - prevBox.x) + (1 - alpha) * correctedState.at<float>(2);
    correctedState.at<float>(3) = alpha * (meas.at<float>(1) - prevBox.y) + (1 - alpha) * correctedState.at<float>(3);

    // 🔹 4️⃣ 보정된 상태를 필터에 반영
    kf.statePost = correctedState;
    prevBox = currentBox;

    // 🔹 5️⃣ 최종 Bounding Box 결정
    cv::Rect2d adjustedBox(predictedX, predictedY, currentBox.width, currentBox.height);

    // 🔹 6️⃣ 이미지 경계를 벗어나지 않도록 조정
    adjustedBox.x = std::max(0.0, std::min(adjustedBox.x, (double)(frame.cols - adjustedBox.width)));
    adjustedBox.y = std::max(0.0, std::min(adjustedBox.y, (double)(frame.rows - adjustedBox.height)));

    return adjustedBox;
}
