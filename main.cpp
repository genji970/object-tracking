#include <opencv2/opencv.hpp>
#include "tracking.hpp"
#include "post_processing.hpp"
#include "image_processing.hpp"  // ✅ 이미지 처리 모듈 추가

int main() {
    cv::VideoCapture cap(0); // 웹캠 사용
    if (!cap.isOpened()) {
        std::cerr << "웹캠을 열 수 없습니다!" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FPS, 5); // 60 FPS 설정 (카메라 지원 여부 확인)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // ✅ 비디오 저장을 위한 VideoWriter 설정
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter video("C:/Users/home/source/repos/object/object/output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(frame_width, frame_height));

    if (!video.isOpened()) {
        std::cerr << "비디오 파일을 저장할 수 없습니다!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cap >> frame;

    // ✅ 프레임 선명화 및 Edge 강조 적용
    frame = enhanceImage(frame);

    cv::Rect2d boundingBox = cv::selectROI("Select Object", frame);
    if (boundingBox.width == 0 || boundingBox.height == 0) return -1;

    ObjectTracker tracker;
    tracker.init(frame, boundingBox);

    double startTime = cv::getTickCount(); // FPS 계산용

    while (cap.read(frame)) {
        double fps = cv::getTickFrequency() / (cv::getTickCount() - startTime);
        startTime = cv::getTickCount(); // 프레임 시간 갱신

        // ✅ 매 프레임마다 이미지 선명화 적용
        frame = enhanceImage(frame);

        cv::Rect2d trackerBox = boundingBox;
        bool trackingSuccess = tracker.update(frame, trackerBox);

        if (!trackingSuccess) break;

        boundingBox = customAlgorithm(frame, trackerBox);

        // ✅ 바운딩 박스를 추가한 후 저장
        cv::rectangle(frame, boundingBox, cv::Scalar(255, 0, 0), 2);  // 파란색 바운딩 박스
        cv::putText(frame, "Tracking", cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2); // 상태 표시
        cv::putText(frame, "FPS: " + std::to_string(int(fps)), cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2); // FPS 표시

        video.write(frame); // ✅ 바운딩 박스 포함된 프레임 저장

        // ✅ 화면 출력
        cv::imshow("Tracking", frame);

        if (cv::waitKey(1) == 27) break; // ESC 키 종료
    }

    cap.release();
    video.release(); // ✅ 비디오 저장 종료
    cv::destroyAllWindows();
    return 0;
}
