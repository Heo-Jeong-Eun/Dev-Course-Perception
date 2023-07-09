// header file
#include <opencv2/opencv.hpp>

class yolov3_trt
{
public:
    int rgb_detection(cv::Mat image_raw, std::vector<BoundingBox> bboxes_info) 
    {
        // get image
        cv::Mat image = image_raw.clone();

        // if no bounding box info, return original image
        if (bboxes_info.empty()) 
        {
            return image_raw;
        }

        // bounding box info for traffic lights
        // box score, xmin, xmax, ymin, ymax => bboxes infos
        std::vector<BoundingBox> traffic_roi;

        for (const auto& bbox : bboxes_info) 
        {
            // if the bbox is a traffic light
            if (bbox.id == 5) 
            {
                // add bbox to traffic_roi
                traffic_roi.push_back(bbox);
            }
        }

        // process each ROI
        for (const auto& roi : traffic_roi) 
        {
            // crop image
            cv::Mat cropped_image = image(cv::Range(roi.ymin, roi.ymax), cv::Range(roi.xmin, roi.xmax));

            // blur processing for accurate circle detection (using median blur)
            cv::Mat blur_image;
            cv::medianBlur(cropped_image, blur_image, 5);

            // BGR -> HSV
            cv::Mat hsv;
            cv::cvtColor(blur_image, hsv, cv::COLOR_BGR2HSV);

            // split H, S, V channels (to use V)
            std::vector<cv::Mat> channels;
            cv::split(hsv, channels);
            cv::Mat v = channels[2];

            // detect circles
            std::vector<cv::Vec3f> circles;
            cv::HoughCircles(v, circles, cv::HOUGH_GRADIENT, 1, 20, 25, 25, 1, 100);

            if (circles.empty()) 
            {
                continue;
            }

            // define lower and upper boundaries for red, green, and yellow colors
            cv::Scalar red_lower(0, 100, 100);
            cv::Scalar red_upper(10, 255, 255);
            cv::Scalar green_lower(45, 90, 90);
            cv::Scalar green_upper(100, 255, 255);
            cv::Scalar yellow_lower(20, 100, 120);
            cv::Scalar yellow_upper(35, 255, 255);

            for (const auto& circle : circles) {
                // circle coordinates and radius
                int x = cvRound(circle[0]);
                int y = cvRound(circle[1]);
                int radius = cvRound(circle[2]);

                // crop region around the circle
                cv::Mat cr_image_h = channels[0](cv::Range(y - 10, y + 10), cv::Range(x - 10, x + 10));
                cv::Mat cr_image_s = channels[1](cv::Range(y - 10, y + 10), cv::Range(x - 10, x + 10));
                cv::Mat cr_image_v = channels[2](cv::Range(y - 10, y + 10), cv::Range(x - 10, x + 10));

                cv::Mat cr_image;
                cv::merge(std::vector<cv::Mat>{cr_image_h, cr_image_s, cr_image_v}, cr_image);

                if (cr_image.empty()) 
                {
                    continue;
                }

                // draw the circle
                cv::circle(cropped_image, cv::Point(x, y), radius, cv::Scalar(0, 255, 0), 2);

                // resize images to the same size
                cv::resize(cropped_image, cr_image, cr_image.size());

                // use mask images to detect colors within the HSV image
                // red version
                cv::Mat red_mask, red_result;
                cv::inRange(cr_image, red_lower, red_upper, red_mask);
                cv::bitwise_and(cropped_image, cropped_image, red_result, red_mask);

                // green version
                cv::Mat green_mask, green_result;
                cv::inRange(cr_image, green_lower, green_upper, green_mask);
                cv::bitwise_and(cropped_image, cropped_image, green_result, green_mask);

                // yellow version
                cv::Mat yellow_mask, yellow_result;
                cv::inRange(cr_image, yellow_lower, yellow_upper, yellow_mask);
                cv::bitwise_and(cropped_image, cropped_image, yellow_result, yellow_mask);

                // result values: off = 0, red = 1, yellow = 2, green = 3
                int result = 0;

                // if the pixel mean value is nonzero, a color has been detected
                if (cr_image.mean() >= 90) 
                {
                    if (cv::mean(red_result).val[0] >= 2) result = 1;
                    if (cv::mean(yellow_result).val[0] >= 2) result = 2;
                    if (cv::mean(green_result).val[0] >= 2) result = 3;
                }

                // if the cropped image mean value is less than or equal to 60, traffic light is off (result = 0)
                if (cropped_image.mean() <= 60) 
                {
                    result = 0;
                }

                return result;
            }
        }
    }
}