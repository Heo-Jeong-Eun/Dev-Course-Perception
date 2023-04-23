//
//  main.cpp
//  warp_process_image
//
//  Created by J on 2023/04/23.
// 

#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

Size imageSize = Size(640, 480);

// Calibrate 관련 변수 선언
double calibrate_mtx_data[9] = {
    350.354184, 0.0, 328.104147,
    0.0, 350.652653, 236.540676,
    0.0, 0.0, 1.0
};
double dist_data[5] = { -0.289296, 0.061035, 0.001786, 0.015238, 0.0 };

Mat calibrate_mtx(3, 3, CV_64FC1, calibrate_mtx_data);
Mat distCoeffs(1, 4, CV_64FC1, dist_data);
Mat cameraMatrix = getOptimalNewCameraMatrix(calibrate_mtx, distCoeffs, imageSize, 1, imageSize);

// calibrate 함수
Mat calibrate_image(Mat src, Mat map1, Mat map2)
{
    // python과의 차이점은 getOptimalNewCameraMatrix함수가 roi값을 리턴하지 않는다는 것인데,
    // 깔끔한 영상처리를 위해 python에서 return받은 roi값을 가져왔습니다.
    Rect roi(34, 61, 585, 360);

    // image calibrating
    Mat temp_calibrate = src.clone();
    Mat calibrated_image;
    remap(src, temp_calibrate, map1, map2, INTER_LINEAR);

    // image slicing & resizing
    temp_calibrate = temp_calibrate(roi);
    resize(temp_calibrate, calibrated_image, imageSize);

    return calibrated_image;
};

// warp 함수
Mat warp_image(Mat image)
{
    // Warping 관련 변수 선언
    int warp_image_width = 320;
    int warp_image_height = 240;

    int warp_x_margin = 30;
    int warp_y_margin = 3;

    Point src_pts1 = Point2f(290 - warp_x_margin, 290 - warp_y_margin);
    Point src_pts2 = Point2f(100 - warp_x_margin, 410 + warp_y_margin);
    Point src_pts3 = Point2f(440 + warp_x_margin, 290 - warp_y_margin);
    Point src_pts4 = Point2f(580 + warp_x_margin, 400 + warp_y_margin);

    Point dist_pts2 = Point2f(0, warp_image_height);
    Point dist_pts3 = Point2f(warp_image_width, 0);
    Point dist_pts4 = Point2f(warp_image_width, warp_image_height);
    Point dist_pts1 = Point2f(0, 0);

    vector<Point2f> warp_src_mtx = { src_pts1, src_pts2, src_pts3, src_pts4 };
    vector<Point2f> warp_dist_mtx = { dist_pts1, dist_pts2, dist_pts3, dist_pts4 };

    Mat src_to_dist_mtx = getPerspectiveTransform(warp_src_mtx, warp_dist_mtx);

    Mat warped_image;
    warpPerspective(image, warped_image, src_to_dist_mtx, Size(warp_image_width, warp_image_height), INTER_LINEAR);

    // warp 기준점 확인
    circle(image, src_pts1, 20, Scalar(255, 0, 0), -1);
    circle(image, src_pts2, 20, Scalar(255, 0, 0), -1);
    circle(image, src_pts3, 20, Scalar(255, 0, 0), -1);
    circle(image, src_pts4, 20, Scalar(255, 0, 0), -1);

    return warped_image;
};

int main()
{
    // Video load
    VideoCapture capture("/Users/1001l1000/Documents/Dev-Course/Dev-Team/VS-Code/track2.avi");

    if (!capture.isOpened()) {
        cerr << "Image laod failed!" << endl;
        return -1;
    }

    Mat src;

    /*
    * FPS 세팅함수를 사용해서 배속조정이 가능한지 실험해보았는데, 해당 함수로는 배속 조정이 불가합니다.
    capture.set(CAP_PROP_FPS, 50);
    */

    // Video width, height 설정
    capture.set(CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CAP_PROP_FRAME_HEIGHT, 480);

    // FPS 측정을 위한 변수 선언
    int capture_fps = capture.get(CAP_PROP_FPS);

    // 기존 undistort함수를 initUndistortRectifyMat(), remap()으로 나눠 loop 밖에서 initUndistortRectifyMat() 선언
    Mat map1, map2;
    initUndistortRectifyMap(calibrate_mtx, distCoeffs, Mat(), cameraMatrix, imageSize, CV_32FC1, map1, map2);

    while (true) {
        capture >> src;

        if (src.empty()) {
            cerr << "Frame empty!" << endl;
            break;
        }

        // FPS 출력
        cout << "fps:" << capture_fps << endl;

        // calibrate image
        Mat calibrated_image = calibrate_image(src, map1, map2);

        // warp image
        Mat warped_image = warp_image(calibrated_image);

        // Image 출력
        imshow("src", src);
        imshow("calibrated image", calibrated_image);
        imshow("warped image", warped_image);

        // waitKey(n)의 n값에 따라 동영상의 배속, 감속됨
        if (waitKey(20) == 27) // ESC
            break;
    }
}