#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void MaskOp1();
void MaskOp2();

int main()
{
//	MaskOp1();
	MaskOp2();
}

void MaskOp1()
{
	Mat src = imread("airplane.bmp", IMREAD_COLOR);
	Mat mask = imread("mask_plane.bmp", IMREAD_GRAYSCALE);
	Mat dst = imread("field.bmp", IMREAD_COLOR);

	if (src.empty() || mask.empty() || dst.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	//copyTo(src, dst, mask);
	src.copyTo(dst, mask);

	imshow("src", src);
	imshow("dst", dst);
	imshow("mask", mask);
	waitKey();
	destroyAllWindows();
}

void MaskOp2()
{
	Mat src = imread("cat.bmp", IMREAD_COLOR);
	Mat logo = imread("opencv-logo-white.png", IMREAD_UNCHANGED);

	if (src.empty() || logo.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	vector<Mat> planes;
	split(logo, planes);

	Mat mask = planes[3];
	merge(vector<Mat>(planes.begin(), planes.begin() + 3), logo);
	Mat crop = src(Rect(10, 10, logo.cols, logo.rows));

	logo.copyTo(crop, mask);

	imshow("src", src);
	waitKey();
	destroyAllWindows();
}