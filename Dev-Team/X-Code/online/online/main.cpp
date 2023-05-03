//
//  main.cpp
//  online
//
//  Created by J on 2023/05/03.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "sliding_window.h"
#include "image_setting.h"

int main()
{
    image_setting setting;
    // Video load
    // cv::VideoCapture capture("/Users/1001l1000/Documents/Dev-Course/Dev-Team/X-Code/online/online/track2.avi");
    cv::VideoCapture capture("/Users/1001l1000/Documents/Dev-Course/Dev-Team/X-Code/online/online/sub_project.avi");

    if (!capture.isOpened())
    {
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
    int capture_fps = capture.get(cv::CAP_PROP_FPS);
    
    while (true)
    {
        sliding_window window;

        capture >> src;

        if (src.empty())
        {
            std::cerr << "Frame empty!" << std::endl;
            break;
        }

        // FPS 출력
        std::cout << "fps:" << capture_fps << std::endl;

        // calibrate image
        cv::Mat calibrated_image = window.calibrate_image(src, map1, map2, roi);

        // warp image
        cv::Mat warped_image = window.warp_image(calibrated_image);

        cv::Mat lane = window.warp_process_image(warped_image);
        //cv::Mat lane = warp_process_image(warped_image);

        // Image 출력
        imshow("src", src);
        imshow("calibrated image", calibrated_image);
        imshow("warped image", warped_image);
        imshow("bin", lane);

        // waitKey(n)의 n값에 따라 동영상의 배속, 감속됨, 27 == ESC
        if (cv::waitKey(20) == 27) break;
    }
}
