//
//  main.cpp
//  warp_process_image
//
//  Created by J on 2023/05/03
// 

#include <iostream>
#include <numeric>
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
    
    int window_margin = 20;
    int lane_width = 200;
    
    cv::Mat out_img;
    
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
        
        int rightx_current;
        int leftx_current;
        cv::Mat left_histogram;
        cv::Mat right_histogram;
        
        // 첫 window를 기반으로 그리기 위해 첫 window는 화면 중앙을 기준으로 좌, 우 차선을 나눈다.
        if (i == num_sliding_window - 1)
        {
            cv::Mat left_histogram = histogram(cv::Range::all(), cv::Range(0, midpoint));
            cv::Mat right_histogram = histogram(cv::Range::all(), cv::Range(midpoint, histogram.cols)) + midpoint;
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
        
        // ! ERROR -> Thread 1: EXC_BAD_ACCESS (code=1, address=0xfffffffffffffffc)
        // 이전 window에서 차선을 둘 다 인식한 경우
        else if (before_l_detected == true && before_r_detected == true)
        {
            cv::Mat left_histogram = histogram(cv::Range::all(), cv::Range(std::max(0, lx.back() - window_margin), lx.back() + window_margin));
            // cv::Mat left_histogram = histogram(cv::Range::all(), cv::Range(std::max(0, lx.back() - window_margin), lx.back() + window_margin) + std::max(0, lx.back() - window_margin));
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
        
        // 이전 window에서 왼쪽 차선 인식 X, 오른쪽 차선만 인식 O한 경우
        else if (before_l_detected == false && before_r_detected == true)
        {
            // if (rx[rx.size() - 1] - lane_width > 0)
            if (rx.back() - lane_width > 0)
            {
                cv::Mat left_histogram = histogram(cv::Range::all(), cv::Range(0, rx.back() - lane_width));
                
                double left_max_val;
                cv::Point left_max_loc;
                cv::minMaxLoc(left_histogram, NULL, &left_max_val, NULL, &left_max_loc);
                int left_current = left_max_loc.x;
            }
            else
            {
                int leftx_current = NULL;
            }
            
            cv::Mat right_histogram = histogram(cv::Range::all(), cv::Range(rx.back()-window_margin, std::min(320,rx.back()+window_margin)));
            
            double right_max_val;
            cv::Point right_max_loc;
            cv::minMaxLoc(right_histogram, NULL, &right_max_val, NULL, &right_max_loc);
            rightx_current = right_max_loc.x + midpoint;
        }
        
        // 이전 window에서 왼쪽 차선은 인식 O, 오른쪽 차선은 인식 X한 경우
        else if (before_l_detected == true && before_r_detected == false)
        {
            // if (lx[lx.size() - 1] + lane_width < histogram.size[0])
            if (lx.back() + lane_width < HALF_WIDTH)
            {
                // cv::Mat right_histogram = histogram(cv::Range::all(), cv::Range(lx.back() + lane_width, histogram.cols)) + (lx.back() + lane_width);
                cv::Mat right_histogram = histogram(cv::Range::all(), cv::Range(lx.back() + lane_width, histogram.cols));
            }
            else
            {
                int rightx_current = NULL;
            }
            
            // cv::Mat left_histogram = histogram(cv::Range::all(), cv::Range(std::max(0, lx.back() - window_margin), lx.back() + window_margin) + std::max(0, lx.back() - window_margin));
            cv::Mat left_histogram = histogram(cv::Range::all(), cv::Range(std::max(0, lx.back() - window_margin), lx.back() + window_margin));
            
            double left_max_val;
            cv::Point left_max_loc;
            cv::minMaxLoc(left_histogram, NULL, &left_max_val, NULL, &left_max_loc);
            int left_current = left_max_loc.x;
        }
        
        // 이전 window에서 차선을 둘 다 인식하지 못한 경우
        else if (before_l_detected == false && before_r_detected == false)
        {
            cv::Mat left_histogram = histogram(cv::Range::all(), cv::Range(0, midpoint));
            cv::Mat right_histogram = histogram(cv::Range::all(), cv::Range(midpoint, histogram.cols)) + midpoint;
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
        
        int window_height = lane.rows / num_sliding_window;
        
        // nz 나눈 이유
        // 화면을 midpoint를 기준으로 나누었을 때 r_current, l_current를 구하는 0점의 좌표가
        // 파이썬 코드에서는(0, 0)이기 때문에 값을 구하기 위해 연산이 더 생기는 것 같아서
        // midpoint를 기준으로 좌, 우 0점값을 따로 계산하는 것으로 계산의 복잡함을 줄이도록 했다.
        // 즉 좌, 우 좌표 값을 따로 계산한다.
        
        // 왼쪽에서 흰색 찾기
        cv::Mat left_nz;
        cv::findNonZero(left_histogram, left_nz);
        std::vector<int> left_nonzeros;
        
        for(int i = 0; i < left_nz.total(); i++)
        {
            left_nonzeros.push_back(left_nz.at<cv::Point>(i).x);
        }
        
        // 오른쪽에서 흰색 찾기
        cv::Mat right_nz;
        cv::findNonZero(right_histogram, right_nz);
        std::vector<int> right_nonzeros;
        
        for(int i = 0; i < right_nz.total(); i++)
        {
            right_nonzeros.push_back(right_nz.at<cv::Point>(i).x);
        }
        
        int win_yl = (i + 1) * window_height;
        int win_yh = i * window_height;
        
        // 오른쪽 차선의 경우
        if (rightx_current != NULL)
        {
            int win_xrl = rightx_current - width_sliding_window;
            int win_xrh = rightx_current + width_sliding_window;
            
            // 차선 픽셀 정보 저장
            // nz에서 지정 범위내에 있는 좌표값들만 차선이라고 인식
            // good_right_inds에 저장
            
            // 검출된 차선의 픽셀이 최소 픽셀의 개수보다 많은지
            if(right_nz.rows > min_points)
            {
                // 흰색 차선의 평균값을 right_current로 지정
                rightx_current = int(std::accumulate(right_nonzeros.begin(), right_nonzeros.end(), 0.0) / right_nonzeros.size());
                // 오른쪽 슬라이딩 윈도우 그리기
                cv::rectangle(out_img, cv::Point(win_xrl, win_yl), cv::Point(win_xrh, win_yh), cv::Scalar(0,255,0), 2);
                before_r_detected = true;
            }
            // 최소 픽셀 개수보다 검출한 차선의 픽셀 개수가 적다면
            else before_r_detected = false;
            
            // right_current가 담긴 lx 생성
            rx.push_back(rightx_current);
        }
        else before_r_detected = false;
        
        if (leftx_current != NULL)
        {
            // 슬라이딩을 그리기 위한 오른쪽 x좌표의 low, high값
            int win_xll = leftx_current - width_sliding_window;
            int win_xlh = leftx_current + width_sliding_window;
            
            // 검출된 차선의 픽셀이 최소 픽셀의 개수보다 많은지
            if(left_nz.rows > min_points)
            {
                // 흰색 차선의 평균값을 right_current로 지정
                leftx_current = int(std::accumulate(left_nonzeros.begin(), left_nonzeros.end(), 0.0) / left_nonzeros.size());
                // 오른쪽 슬라이딩 윈도우 그리기
                cv::rectangle(out_img, cv::Point(win_xll, win_yl), cv::Point(win_xlh, win_yh), cv::Scalar(0, 0, 255), 2);
                before_l_detected = true;
            }
            // 최소 픽셀 개수보다 검출한 차선의 픽셀 개수가 적다면
            else before_l_detected = false;
            
            // right_current가 담긴 lx 생성
            lx.push_back(leftx_current);
            
        }
        else before_l_detected = false;
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
