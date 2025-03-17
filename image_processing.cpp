#include "image_processing.hpp"

// 🔹 Blur 감소 및 Edge 강조
cv::Mat enhanceImage(const cv::Mat& frame) {
    cv::Mat sharpened, edges, enhanced;

    // 1️⃣ GaussianBlur로 노이즈 제거 후 선명도 증가
    cv::Mat blurred;
    cv::GaussianBlur(frame, blurred, cv::Size(3, 3), 0);

    // 2️⃣ Laplacian 연산으로 Edge 강조
    cv::Mat laplacian;
    cv::Laplacian(blurred, laplacian, CV_16S, 3);
    cv::convertScaleAbs(laplacian, edges);

    // 3️⃣ 원본 이미지와 Edge를 결합하여 선명한 이미지 생성
    cv::addWeighted(frame, 1.5, edges, -0.5, 0, enhanced);

    return enhanced;
}
