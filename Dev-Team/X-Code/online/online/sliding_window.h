//
//  sliding_window.h
//  online
//
//  Created by J on 2023/05/03.
//

#include "opencv2/opencv.hpp"

extern cv::Size image_size;

#pragma once
#ifndef SLIDING_WINDOW_H
#define SLIDING_WINDOW_H
class sliding_window
{
    public:
        cv::Mat calibrate_image(cv::Mat const& src, cv::Mat const& map1, cv::Mat const& map2, cv::Rect const& roi);
        cv::Mat warp_image(cv::Mat image);
        cv::Mat warp_process_image(cv::Mat image);
};
#endif
