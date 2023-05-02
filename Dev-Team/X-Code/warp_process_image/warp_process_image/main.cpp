//
//  main.cpp
//  warp_process_image
//
//  Created by J on 2023/05/03
// 

#include <iostream>
#include "opencv2/opencv.hpp"

cv::Size image_size = cv::Size(640, 480);
constexpr int HALF_WIDTH = 320;
constexpr int HALF_HEIGHT = 240;

// Calibrate 관련 변수 선언
double calibrate_mtx_data[9] = {
	350.354184, 0.0, 328.104147,
	0.0, 350.652653, 236.540676,
	0.0, 0.0, 1.0
};
double dist_data[5] = {-0.289296, 0.061035, 0.001786, 0.015238, 0.0};

cv::Rect roi;

cv::Mat calibrate_mtx(3, 3, CV_64FC1, calibrate_mtx_data);
cv::Mat distCoeffs(1, 4, CV_64FC1, dist_data);
cv::Mat cameraMatrix = getOptimalNewCameraMatrix(calibrate_mtx, distCoeffs, image_size, 1, image_size, &roi);

// calibrate 함수
cv::Mat calibrate_image(cv::Mat const& src, cv::Mat const& map1, cv::Mat const& map2)
{
	// image calibrating
	cv::Mat mapping_image = src.clone();
	cv::Mat calibrated_image;
	remap(src, mapping_image, map1, map2, cv::INTER_LINEAR);

	// image slicing & resizing
	mapping_image = mapping_image(roi);
	resize(mapping_image, calibrated_image, image_size);

	return calibrated_image;
};

// warp 함수
cv::Mat warp_image(cv::Mat image)
{
	// Warping 관련 변수 선언
	int warp_image_width = HALF_WIDTH;
	int warp_image_height = HALF_HEIGHT;

	int warp_x_margin = 30;
	int warp_y_margin = 3;

	cv::Point src_pts1 = cv::Point2f(290 - warp_x_margin, 290 - warp_y_margin);
	cv::Point src_pts2 = cv::Point2f(100 - warp_x_margin, 410 + warp_y_margin);
	cv::Point src_pts3 = cv::Point2f(440 + warp_x_margin, 290 - warp_y_margin);
	cv::Point src_pts4 = cv::Point2f(580 + warp_x_margin, 400 + warp_y_margin);

	cv::Point dist_pts2 = cv::Point2f(0, warp_image_height);
	cv::Point dist_pts3 = cv::Point2f(warp_image_width, 0);
	cv::Point dist_pts4 = cv::Point2f(warp_image_width, warp_image_height);
	cv::Point dist_pts1 = cv::Point2f(0, 0);

	std::vector<cv::Point2f> warp_src_mtx = { src_pts1, src_pts2, src_pts3, src_pts4 };
	std::vector<cv::Point2f> warp_dist_mtx = { dist_pts1, dist_pts2, dist_pts3, dist_pts4 };

	cv::Mat src_to_dist_mtx = getPerspectiveTransform(warp_src_mtx, warp_dist_mtx);

	cv::Mat warped_image;
	warpPerspective(image, warped_image, src_to_dist_mtx, cv::Size(warp_image_width, warp_image_height), cv::INTER_LINEAR);

	// warp 기준점 확인
	circle(image, src_pts1, 20, cv::Scalar(255, 0, 0), -1);
	circle(image, src_pts2, 20, cv::Scalar(255, 0, 0), -1);
	circle(image, src_pts3, 20, cv::Scalar(255, 0, 0), -1);
	circle(image, src_pts4, 20, cv::Scalar(255, 0, 0), -1);

	return warped_image;
};

cv::Mat warp_process_image(cv::Mat image)
{
    int num_sliding_window = 20;
    int width_sliding_window = 20;
    int min_points = 5;
    int lane_bin_th = 145;
    
    cv::Mat blur;
    GaussianBlur(image, blur, cv::Size(5, 5), 0);
    
    cv::Mat hls;
    cvtColor(blur, hls, cv::COLOR_BGR2HLS);
    std::vector<cv::Mat> L;
    split(hls, L);
    
    cv::Mat lane;
    threshold(L[1], lane, lane_bin_th, 255, cv::THRESH_BINARY);
    // threshold(L, lane, lane_bin_th, 255, THRESH_BINARY_INV);
    
    // C++ 변환
    // out_image = np.dstack((lane, lane, lane)) * 255
    
    int window_margin = 20;
    
    int lane_width = 200;
    
    std::vector<int> lx, ly, rx, ry;
    std::vector<int> l_box_center, r_box_center;
        
    // 자른 window 갯수만큼 for loop
    for (int i = num_sliding_window - 1; i >= 0; i--)
    {
        cv::Rect roi(0, i * lane.rows / num_sliding_window, lane.cols, lane.rows/num_sliding_window);
        cv::Mat window = lane(roi);
        
        cv::Mat histogram;
        cv::reduce(window, histogram, 0, cv::REDUCE_SUM, CV_32S);
        
        std::vector<int> left_lane_inds, right_lane_inds;
        
        int midpoint = lane.cols / 2;
        
        int r_min_points = min_points;
        int l_min_points = min_points;
        
        bool before_l_detected = true;
        bool before_r_detected = true;
        
        // 첫 window를 기반으로 그리기 위해 첫 window는 화면 중앙을 기준으로 좌, 우 차선을 나눈다.
        if (i == num_sliding_window - 1)
        {
            cv::Mat left_histogram = histogram(cv::Range::all(), cv::Range(0, midpoint));
            cv::Mat right_histogram = histogram(cv::Range::all(), cv::Range(midpoint, histogram.cols));
            // std::cout << "left_histogram: " << left_histogram <<std::endl;
            // std::cout << "right_histogram: " << right_histogram <<std::endl;
            
            // minMaxLoc(image, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc, mask = noarray())
            double left_max_val;
            cv::Point left_max_loc;
            cv::minMaxLoc(left_histogram, NULL, &left_max_val, NULL, &left_max_loc);
            int left_current = left_max_loc.x;
            
            // std::cout << "left_current: " << left_current <<std::endl;
            
            double right_max_val;
            cv::Point right_max_loc;
            cv::minMaxLoc(right_histogram, NULL, &right_max_val, NULL, &right_max_loc);
            int right_current = right_max_loc.x;
            
            // std::cout << "right_current: " << right_current <<std::endl;
        }
        
        // ? -> 충돌 에러 O
        // 이전 window에서 차선을 둘 다 인식한 경우
        else if (before_l_detected == true && before_r_detected == true)
        {
            cv::Mat left_histogram = histogram(cv::Range::all(), cv::Range(std::max(0, lx.back() - window_margin), lx.back() + window_margin)
                                        + std::max(0, lx.back() - window_margin));
            cv::Mat right_histogram = histogram(cv::Range::all(), cv::Range(rx.back() - window_margin, std::min(histogram.size[0] - 1, rx.back() + window_margin)) 
                                        + rx.back() - window_margin);
            
            // std::cout << "left_histogram: " << left_histogram <<std::endl;
            // std::cout << "right_histogram: " << right_histogram <<std::endl;

            double left_max_val;
            cv::Point left_max_loc;
            cv::minMaxLoc(left_histogram, NULL, &left_max_val, NULL, &left_max_loc);
            int left_current = left_max_loc.x;

            // std::cout << "left_current: " << left_current <<std::endl;

            double right_max_val;
            cv::Point right_max_loc;
            cv::minMaxLoc(right_histogram, NULL, &right_max_val, NULL, &right_max_loc);
            int right_current = right_max_loc.x;

            // std::cout << "right_current: " << right_current <<std::endl;
        }
        
        // ? -> 문법 오류 X, cout X
        // 이전 window에서 왼쪽 차선 인식 X, 오른쪽 차선만 인식 O한 경우
        else if (before_l_detected == false && before_r_detected == true)
        {
            if (rx[rx.size() - 1] - lane_width > 0)
            // if (rx.back() - lane_width > 0)
            {
                cv::Mat left_histogram = histogram(cv::Range::all(), cv::Range(0, rx.back() - lane_width));

                // std::cout << "left_histogram: " << left_histogram <<std::endl;

                double left_max_val;
                cv::Point left_max_loc;
                cv::minMaxLoc(left_histogram, NULL, &left_max_val, NULL, &left_max_loc);
                int left_current = left_max_loc.x;

                // std::cout << "left_current: " << left_current <<std::endl;
            }
            else
            {
                // ? -> None X, nullptr or NULL
                int leftx_current = NULL;
            }
        }
        
        // 이전 window에서 왼쪽 차선은 인식 O, 오른쪽 차선은 인식 X한 경우
        else if (before_l_detected == true && before_r_detected == false)
        {
            cv::Mat left_histogram = histogram(cv::Range::all(), cv::Range(std::max(0, lx.back() - window_margin), lx.back() + window_margin)
                                        + std::max(0, lx.back() - window_margin));
            
            double left_max_val;
            cv::Point left_max_loc;
            cv::minMaxLoc(left_histogram, NULL, &left_max_val, NULL, &left_max_loc);
            int left_current = left_max_loc.x;
            
            if (lx[lx.size() - 1] + lane_width < histogram.size[0])
            // if (lx.back() + lane_width < histogram.size())
            {
                cv::Mat right_histogram = histogram(cv::Range::all(), cv::Range(rx.back() + lane_width, histogram.cols)) + (lx.back() + lane_width);
            }
            else
            {
                // ? -> None X, nullptr or NULL
                int rightx_current = NULL;
            }
        }
        
        // 이전 window에서 차선을 둘 다 인식하지 못한 경우
        else if (before_l_detected == false && before_r_detected == false)
        {
            cv::Mat left_histogram = histogram(cv::Range::all(), cv::Range(0, midpoint));
            cv::Mat right_histogram = histogram(cv::Range::all(), cv::Range(midpoint, histogram.cols));
            // std::cout << "left_histogram: " << left_histogram <<std::endl;
            // std::cout << "right_histogram: " << right_histogram <<std::endl;
            
            // minMaxLoc(image, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc, mask = noarray())
            double left_max_val;
            cv::Point left_max_loc;
            cv::minMaxLoc(left_histogram, NULL, &left_max_val, NULL, &left_max_loc);
            int left_current = left_max_loc.x;
            
            // std::cout << "left_current: " << left_current <<std::endl;
            
            double right_max_val;
            cv::Point right_max_loc;
            cv::minMaxLoc(right_histogram, NULL, &right_max_val, NULL, &right_max_loc);
            int right_current = right_max_loc.x;
            
            // std::cout << "right_current: " << right_current <<std::endl;
        }
        
        /*
        int window_height = static_cast<int>(lane.rows / num_sliding_window);
        
        cv::Mat nz;
        findNonZero(divide_lane[window], nz);
        
        int win_yl = (i + 1) * window_height;
        int win_yh = i * window_height;
        
        // 오른쪽 차선의 경우
        if (rightx_current != None)
        {
            int win_xrl = rightx_current - width_sliding_window;
            int win_xrh = rightx_current + width_sliding_window;
            
            cv::Mat good_right_inds = ((nz.col(0).row(0) >= 0) & (nz.col(0).row(0) < window_height) & (nz.col(1).row(0) >= win_xrl) & (nz.col(1).row(0) < win_xrh));
            
            // C++ 변환
            right_lane_inds.append(good_right_inds)
            
            if (good_right_inds.size() > r_min_points)
            {
            
            }
            
            else
            {
                before_r_detected = False
            }
            
            // C++ 변환
            rx.append(rightx_current)
            ry.append((win_yl + win_yh) / 2)
            
            // C++ 변환
            right_lane_inds = np.concatenate(right_lane_inds)
            
            // C++ 변환
            out_image[(i * window_height) + nz[0][right_lane_inds], nz[1][right_lane_inds]] = [0, 0, 255]
        }

        // C++ 변환
        else
        {
            before_r_detected = False
        }
        
        if (leftx_current != None)
        {
            int win_xll = leftx_current - width_sliding_window;
            int win_xlh = leftx_current + width_sliding_window;
            
            cv::Mat good_left_inds = ((nz.col(0).row(0) >= 0) & (nz.col(0).row(0) < window_height) & (nz.col(1).row(0) >= win_xll) & (nz.col(1).row(0) < win_xlh));
            
            if (good_left_inds.size() > l_min_points)
            {
            
            }
            
            else
            {
            
            }
        }
        else 
        {
            
        }
        
        // C++ 변환
        lfit = np.polyfit(np.array(ly), np.array(lx), 2)
        rfit = np.polyfit(np.array(ry), np.array(rx), 2)
        */
    }
    return lane;
    // return lfit, rfit, np.mean(lx), np.mean(rx), out_image, l_box_center, r_box_center, lane
}

int main()
{
    // Video load
    cv::VideoCapture capture("/Users/1001l1000/Documents/Dev-Course/Dev-Team/VS-Code/track2.avi");
    
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
    
    // 기존 undistort함수를 initUndistortRectifyMat(), remap()으로 나눠 loop 밖에서 initUndistortRectifyMat() 선언
    cv::Mat map1, map2;
    initUndistortRectifyMap(calibrate_mtx, distCoeffs, cv::Mat(), cameraMatrix, image_size, CV_32FC1, map1, map2);
    
    
    while (true)
    {
        capture >> src;
        
        if (src.empty())
        {
            std::cerr << "Frame empty!" << std::endl;
            break;
        }
        
        // FPS 출력
        std::cout << "fps: " << capture_fps << std::endl;
        
        // calibrate image
        cv::Mat calibrated_image = calibrate_image(src, map1, map2);
        
        // warp image
        cv::Mat warped_image = warp_image(calibrated_image);
        
        cv::Mat lane = warp_process_image(warped_image);
        
        // Image 출력
        /*
        imshow("src", src);
        imshow("calibrated image", calibrated_image);
        imshow("warped image", warped_image);
        imshow("bin", lane);
        */
        imshow("bin", lane);
        
        // waitKey(n)의 n값에 따라 동영상의 배속, 감속됨
        if (cv::waitKey(20) == 27) // ESC
            break;
    }
}
