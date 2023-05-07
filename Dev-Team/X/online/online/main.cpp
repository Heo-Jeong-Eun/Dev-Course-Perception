// main.cpp

#include <iostream>

#include "opencv2/opencv.hpp"

#include "image_setting.h"
#include "sliding_window.h"

int main()
{
    image_setting setting;
    // Video load
    cv::VideoCapture capture("/Users/1001l1000/Documents/Lab/sub_project.avi");
    cv::Mat mask_image = cv::imread("/Users/1001l1000/Documents/Dev-Course/Dev-Team/X-Code/online/online/mask_image.png", cv::COLOR_BGR2GRAY);

    if (!capture.isOpened()) {
        std::cerr << "Image laod failed!" << std::endl;
        return -1;
    }

    cv::Mat src;

    /*
    * FPS 세팅함수를 사용해서 배속조정이 가능한지 실험해보았는데, 해당 함수로는 배속 조정이 불가합니다.
    capture.set(CAP_PROP_FPS, 50);
    */
    // Video width, height 설정
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // FPS 측정을 위한 변수 선언
    int capture_frame = capture.get(cv::CAP_PROP_POS_FRAMES);

    SlidingWindow window;

    // mask image setting
    cv::Mat calibrated_mask = window.calibrate_image(mask_image, map1, map2, roi);
    cv::Mat warped_mask_image = window.warp_image(calibrated_mask);
    // cv::Mat binary_mask_image = window.binary_image(warped_mask_image);
    
    while (true) {
        // sliding_window window;

        capture >> src;

        if (src.empty()) {
            std::cerr << "Frame empty!" << std::endl;
            break;
        }

        // clahe image
        cv::Mat clahe_image = window.contrast_clihe(src);

        // calibrate image
        cv::Mat calibrated_image = window.calibrate_image(clahe_image, map1, map2, roi);

        
        // warp image
        cv::Mat warped_image = window.warp_image(calibrated_image);

        cv::Mat lane = window.binary_image_with_adaptivethreshold(warped_image);

        cv::Mat morphological_transformation_image = window.morphological_transformation(lane);

        cv::Mat binary_non_lidar_image;

        cv::subtract(morphological_transformation_image, warped_mask_image, binary_non_lidar_image);

        cv::Mat sliding_window_image = window.draw_sliding_window(binary_non_lidar_image, capture_frame);

        std::vector<cv::Point2f> lpos;
        std::vector<cv::Point2f> rpos;

        if (capture_frame % 30 == 0) {
            lpos = window.warp_point(l_pos);
            rpos = window.warp_point(r_pos);
        }
        
        // Image 출력
        imshow("src", src);
        imshow("calibrated image", calibrated_image);
        imshow("warped image", warped_image);
        imshow("adap_thresh", lane);
        imshow("morph_image", morphological_transformation_image);
        imshow("sliding_window_image_adap", sliding_window_image);

        // waitKey(n)의 n값에 따라 동영상의 배속, 감속됨
        if (cv::waitKey(20) == 27) // ESC
            break;
    }
}
