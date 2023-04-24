#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("contours.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(src, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);

	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
		Scalar color(rand() & 255, rand() & 255, rand() & 255);
		drawContours(dst, contours, idx, color, 2, LINE_8, hierarchy);
	}

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}
