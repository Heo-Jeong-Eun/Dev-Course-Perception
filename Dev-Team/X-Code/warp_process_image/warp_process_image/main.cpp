//
//  main.cpp
//  warp_process_image
//
//  Created by J on 2023/04/26.
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
double dist_data[5] = {-0.289296, 0.061035, 0.001786, 0.015238, 0.0};

Mat calibrate_mtx(3, 3, CV_64FC1, calibrate_mtx_data);
Mat distCoeffs(1, 4, CV_64FC1, dist_data);
Mat cameraMatrix = getOptimalNewCameraMatrix(calibrate_mtx, distCoeffs, imageSize, 1, imageSize);

// calibrate 함수
Mat calibrate_image(Mat src, Mat map1, Mat map2)
{
    // python과의 차이점은 getOptimalNewCameraMatrix 함수가 roi값을 리턴하지 않는다는 것
    // 깔끔한 영상처리를 위해 python에서 return받은 roi값을 가져왔다.
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

    vector<Point2f> warp_src_mtx = {src_pts1, src_pts2, src_pts3, src_pts4};
    vector<Point2f> warp_dist_mtx = {dist_pts1, dist_pts2, dist_pts3, dist_pts4};

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

Mat warp_process_image(Mat image)
{
    int num_sliding_window = 20;
    int width_sliding_window = 20;
    int min_points = 5;
    int lane_bin_th = 145;

    Mat blur;
    GaussianBlur(image, blur, Size(5, 5), 0);

    Mat hls;
    cvtColor(blur, hls, COLOR_BGR2HLS);
    vector<Mat> L;
    split(hls, L);

    Mat lane;
    // L에 3채널 데이터가 들어와서 원하는 L채널 데이터를 추출한다. 
    threshold(L[1], lane, lane_bin_th, 255, THRESH_BINARY);
    // threshold(L[1], lane, lane_bin_th, 255, THRESH_BINARY_INV);

    vector<Mat> dividen_lane;
    int rows_per_window = lane.rows / num_sliding_window;
    for (int i = 0; i < num_sliding_window; i++) 
    {
        Rect roi(0, i * rows_per_window, lane.cols, rows_per_window);
        dividen_lane.push_back(lane(roi));
    }

    return lane;

    /*
    int window_margin = 20;
    int lane_width = 200;
    vector<int> lx, ly, rx, ry;
    vector<int> l_box_center, r_box_center;

    bool before_l_detected = true;
    bool before_r_detected = true;

    // ? -> CV_8UC3 
    Mat out_img = Mat::zeros(lane.size(), CV_8UC3);
    
    for (int window = num_sliding_window - 1; window > 0; --window) 
    {
        vector<int> left_lane_inds, right_lane_inds;

        Mat histogram = sum(dividen_lane[window])[0];

        if (window == num_sliding_window - 1) 
        {
            leftx_current = Point2f(0, Mat(histogram.rowRange(0, midpoint)).dot( Mat(Range::all(), Range(0, 1))));
            rightx_current = Point2f(midpoint, Mat(histogram.rowRange(midpoint, histogram.rows)).dot(Mat(Range::all(), Range(0, 1)))) + midpoint;
        }
        else if (before_l_detected == true && before_r_detected == true) 
        {
            leftx_current = Point2f(0, Mat(histogram.rowRange(0, lx.back() + window_margin)).dot(Mat(Range::all(), Range(0, 1))));
            rightx_current = Point2f(rx.back() - window_margin, Mat(histogram.rowRange(rx.back() - window_margin, rx.back() + window_margin)).dot(Mat(Range::all(), Range(0, 1)))) + rx.back() - window_margin;
        }
        else if (before_l_detected == false && before_r_detected == true) 
        {
            if (rx.back() - lane_width > 0) 
            {
                leftx_current = Point2f(0, Mat(histogram.rowRange(0, rx.back() - lane_width)).dot(Mat(Range::all(), Range(0, 1))));
                rightx_current = Point2f(rx.back() - window_margin, Mat(histogram.rowRange(rx.back() - window_margin, histogram.rows)).dot(Mat(Range::all(), Range(0, 1)))) + rx.back() - window_margin;
                if (abs(leftx_current.x - rightx_current.x) < 100) 
                {
                    l_min_points = (width_sliding_window * 2) * (warp_image_height / num_sliding_window);
                }
            }
        }
        else if (before_l_detected == true && before_r_detected == false) 
        {
            leftx_current = Point2f(max(0, lx.back() - window_margin), Mat(histogram.rowRange(max(0, lx.back() - window_margin), lx.back() + window_margin)).dot(deleteMat(deleteRange::all(), deleteRange(0, 1)))) + max(0, lx.back() - window_margin);
            rightx_current = Point2f(min(lx.back() + lane_width, histogram.cols - 1), Mat(histogram.rowRange(min(lx.back() + lane_width, histogram.cols - 1), histogram.rows)).dot(Mat(Range::all(), Range(0, 1)))) + min(lx.back() + lane_width, histogram.cols - 1);
            if (rightx_current.x - leftx_current.x < 100) 
            {
                r_min_points = (width_sliding_window * 2) * (warp_image_height / num_sliding_window);
            }
        }
        else if (before_l_detected == false && before_r_detected == false) 
        {
            leftx_current = Point2f(0, Mat(histogram.rowRange(0, midpoint)).dot(Mat(Range::all(), Range(0, 1))));
            rightx_current = Point2f(midpoint, Mat(histogram.rowRange(midpoint, histogram.rows)).dot(Mat(Range::all(), Range(0, 1)))) + midpoint;
        }

        int window_height = static_cast<int>(lane.rows / num_sliding_window);

        Mat nz;
        findNonZero(divide_n_lane[window], nz);

        int win_yl = (window + 1) * window_height;
        int win_yh = window * window_height;

        int win_xll = leftx_current - width_sliding_window;
        int win_xlh = leftx_current + width_sliding_window;
        int win_xrl = rightx_current - width_sliding_window;
        int win_xrh = rightx_current + width_sliding_window;

        Mat nz;
        findNonZero(dividen_lane[window], nz);

        Mat good_left_inds = ((nz.col(0).row(0) >= 0) & (nz.col(0).row(0) < window_height) & (nz.col(1).row(0) >= win_xll) & (nz.col(1).row(0) < win_xlh));
        Mat good_right_inds = ((nz.col(0).row(0) >= 0) & (nz.col(0).row(0) < window_height) & (nz.col(1).row(0) >= win_xrl) & (nz.col(1).row(0) < win_xrh));

        vector<int> left_lane_inds, right_lane_inds;
        for (int i = 0; i < good_left_inds.cols; ++i) 
        {
            if (good_left_inds.at<bool>(0, i)) 
            {
                left_lane_inds.push_back(nz.at<Point>(0, i).y);
            }
        }
        for (int i = 0; i < good_right_inds.cols; ++i) 
        {
            if (good_right_inds.at<bool>(0, i))
            {
                right_lane_inds.push_back(nz.at<Point>(0, i).y);
            }
        }

        if (good_left_inds.size() > l_min_points) 
        {
            leftx_current = static_cast<int>(mean(nz.row(1).col(good_left_inds))[0]);
            rectangle(out_image, Rect(win_xll, win_yl, win_xlh - win_xll, win_yh - win_yl), Scalar(0, 255, 0), 2);
            l_box_center.push_back({(win_xll + win_xlh) / 2, (win_yl + win_yh) / 2});
            before_l_detected = true;
        } 
        else before_l_detected = false;

        if (good_right_inds.size() > r_min_points) 
        {
            rightx_current = static_cast<int>(mean(nz.row(1).col(good_right_inds))[0]);
            rectangle(out_image, Rect(win_xrl, win_yl, win_xrh - win_xrl, win_yh - win_yl), Scalar(0, 255, 0), 2);
            r_box_center.push_back({(win_xrl + win_xrh) / 2, (win_yl + win_yh) / 2});
            before_r_detected = true;
        } 
        else before_r_detected = false;

        lx.push_back(leftx_current);
        ly.push_back((win_yl + win_yh) / 2);
        rx.push_back(rightx_current);
        ry.push_back((win_yl + win_yh) / 2);

        # 10번의 loop가 끝나면 그동안 모은 점들을 저장한다. 
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # 기존 흰색 차선 픽셀을 왼쪽과 오른쪽 각각 파란색과 빨간색으로 색 변경
        out_image[(window * window_height) + nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
        out_image[(window * window_height) + nz[0][right_lane_inds], nz[1][right_lane_inds]] = [0, 0, 255]

    # 슬라이딩 윈도우의 중심점(x좌표) 9개를 가지고 2차 함수를 만들어낸다. 
    # 2차 함수 -> x = ay^2 + by + c
    lfit = np.polyfit(np.array(ly), np.array(lx), 2)
    rfit = np.polyfit(np.array(ry), np.array(rx), 2)

    return lfit, rfit, np.mean(lx), np.mean(rx), out_image, l_box_center, r_box_center, lane
    }
    */
};

int main()
{
    // Video load
    VideoCapture capture("/Users/1001l1000/Documents/Dev-Course/Dev-Team/VS-Code/track2.avi");

    if (!capture.isOpened())
    {
        cerr << "Image laod failed!" << endl;
        return -1;
    }
    
    /*
    * FPS 세팅함수를 사용해서 배속조정이 가능한지 실험, 해당 함수로는 배속 조정이 불가
    capture.set(CAP_PROP_FPS, 50);
    */
    
    Mat src;

    // Video width, height 설정
    capture.set(CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CAP_PROP_FRAME_HEIGHT, 480);

    // FPS 측정을 위한 변수 선언
    int capture_fps = capture.get(CAP_PROP_FPS);

    // 기존 undistort함수를 initUndistortRectifyMat(), remap()으로 나눠 loop 밖에서 initUndistortRectifyMat() 선언
    Mat map1, map2;
    initUndistortRectifyMap(calibrate_mtx, distCoeffs, Mat(), cameraMatrix, imageSize, CV_32FC1, map1, map2);

    while (true)
    {
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
        
        Mat lane = warp_process_image(warped_image);

        // Image 출력
        imshow("src", src);
        imshow("calibrated image", calibrated_image);
        imshow("warped image", warped_image);
        
        imshow("bin", lane);

        // waitKey(n)의 n값에 따라 동영상의 배속, 감속된다. 
        if (waitKey(20) == 27) // ESC
            break;
    }
}
