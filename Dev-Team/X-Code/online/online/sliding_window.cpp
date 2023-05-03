//
//  sliding_window.cpp
//  online
//
//  Created by J on 2023/05/03.
//

#include "opencv2/opencv.hpp"
#include "sliding_window.h"

constexpr int HALF_WIDTH = 320;
constexpr int HALF_HEIGHT = 240;

cv::Size image_size = cv::Size(640, 480);

// mask_image를 만들어 나가면서 진행하는 것 보다 만들어진 mask_image를 가져와서 subtract연산만을 수행하면서 진행하도록 수정이 필요하다.
cv::Mat mask_image = cv::Mat(cv::Size(320, 240),CV_8UC1, cv::Scalar(255));

// calibrate 함수
cv::Mat sliding_window::calibrate_image(cv::Mat const& src, cv::Mat const& map1, cv::Mat const& map2, cv::Rect const& roi)
{
    // image calibrating
    cv::Mat mapping_image = src.clone();
    cv::Mat calibrated_image;
    remap(src, mapping_image, map1, map2, cv::INTER_LINEAR);

    // image slicing & resizing
    mapping_image = mapping_image(roi);
    resize(mapping_image, calibrated_image, image_size);

    return calibrated_image;
}

// warp 함수
cv::Mat sliding_window::warp_image(cv::Mat image)
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

// warp process image 함수
cv::Mat sliding_window::warp_process_image(cv::Mat image)
{
    int num_sliding_window = 20;
    int width_sliding_window = 20;
    int min_points = 5;
    int lane_bin_th = 180;

    int margin = 12;
    int minpix = 5;
    int window_margin = 30;
    int lane_width = 200;

    cv::Mat blur;
    cv::GaussianBlur(image, blur, cv::Size(5, 5), 0);

    cv::Mat hls;
    cv::cvtColor(blur, hls, cv::COLOR_BGR2HLS);

    std::vector<cv::Mat> L;
    cv::split(hls, L);

    cv::Mat lane;
    // L에 3채널 데이터가 들어와서 원하는 L채널 데이터를 추출한다.
    // cv::threshold(L[1], lane, lane_bin_th, 255, cv::THRESH_BINARY);
    cv::threshold(L[1], lane, lane_bin_th, 255, cv::THRESH_BINARY_INV);
    
    mask_image = cv::min(lane, mask_image);
    cv::subtract(lane, mask_image, lane);
    
    // 정빈 코드 -> imshow test
    int window_height = lane.cols / num_sliding_window;
    int window_width = lane.rows / num_sliding_window;
    std::queue<int> lx;
    std::queue<int> ly;
    std::queue<int> rx;
    std::queue<int> ry;
    cv::Mat out_img = cv::Mat(lane.size(),CV_8UC3, cv::Scalar(0,0,0));
    
    for (int i = 0; i < num_sliding_window; i++)
    {
        cv::Mat divide_lane = lane(cv::Rect(0,i * window_width, lane.cols, window_width));
        std::vector<int> left_lane_inds, right_lane_inds;
        std::vector<int> hist;
        std::vector<int>::iterator leftx_result;
        std::vector<int>::iterator rightx_result;
        long leftx_current;
        long rightx_current;
        
        for (int j = 0; j < divide_lane.cols; j++)
        {
            int sum = 0;
            
            for (int k = 0; k < divide_lane.rows; k++)
            {
                sum += divide_lane.at<uchar>(k,j);
            }
            hist.push_back(sum);
        }
        
        int win_yl = lane.cols - (i+1)*window_height;
        int win_yh = lane.cols - i*window_height;
        int left_check = *std::max_element(hist.begin(), hist.begin()+hist.size()/2);
        int right_check = *std::max_element(hist.begin()+hist.size()/2, hist.end());
        std::cout << "left : " << left_check;
        std::cout << "right : " << right_check << std::endl;
        
        if (left_check > 2500)
        {
            leftx_result = std::max_element(hist.begin(), hist.begin()+hist.size()/2);
            leftx_current = std::distance(hist.begin(), leftx_result);
            int win_xll = (int)leftx_current - margin;
            int win_xlh = (int)leftx_current + margin;
            cv::rectangle(out_img, cv::Point(win_xll, win_yl), cv::Point(win_xlh, win_yh), cv::Scalar(0,255,0), 2);
        }
        
        if (right_check > 2500)
        {
            rightx_result = std::max_element(hist.begin()+hist.size()/2, hist.end());
            rightx_current = std::distance(hist.begin()+hist.size()/2, rightx_result)+hist.size()/2;
            int win_xrl = (int)rightx_current - margin;
            int win_xrh = (int)rightx_current + margin;
            cv::rectangle(out_img, cv::Point(win_xrl, win_yl), cv::Point(win_xrh, win_yh), cv::Scalar(0,255,0), 2);
        }
    }
    cv::Mat result;
    cv::flip(out_img, result, 0);
    cv::imshow("test", result);
    
    return lane;
}
