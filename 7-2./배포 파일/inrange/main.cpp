#include <iostream>
#include <algorithm>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int pos_hue1 = 50, pos_hue2 = 80, pos_sat1 = 150, pos_sat2 = 255;
Mat src, src_hsv, dst, mask;

void on_hsv_changed(int, void*)
{
	Scalar lowerb(pos_hue1, pos_sat1, 0);
	Scalar upperb(pos_hue2, pos_sat2, 255);
	inRange(src_hsv, lowerb, upperb, mask);

	cvtColor(src, dst, COLOR_BGR2GRAY);
	cvtColor(dst, dst, COLOR_GRAY2BGR);
	src.copyTo(dst, mask);

	imshow("mask", mask);
	imshow("dst", dst);
}

int main()
{
	src = imread("candies.png", IMREAD_COLOR);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	cvtColor(src, src_hsv, COLOR_BGR2HSV);

	namedWindow("src");
	namedWindow("mask");
	namedWindow("dst");

	imshow("src", src);

	createTrackbar("Lower Hue", "dst", &pos_hue1, 179, on_hsv_changed);
	createTrackbar("Upper Hue", "dst", &pos_hue2, 179, on_hsv_changed);
	createTrackbar("Lower Sat", "dst", &pos_sat1, 255, on_hsv_changed);
	createTrackbar("Upper Sat", "dst", &pos_sat2, 255, on_hsv_changed);
	on_hsv_changed(0, 0);

	waitKey();
}
