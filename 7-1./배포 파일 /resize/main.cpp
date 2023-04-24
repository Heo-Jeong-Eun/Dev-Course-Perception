#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void resize1();
void resize2();
void resize3();
void resize4();

int main()
{
	resize1();
	resize2();
	resize3();
	resize4();
}

void resize1()
{
	Mat src = imread("camera.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst = Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC1);

	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			int x_ = x * 2;
			int y_ = y * 2;

			dst.at<uchar>(y_, x_) = src.at<uchar>(y, x);
		}
	}

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}

void resize2()
{
	Mat src = imread("camera.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst = Mat::zeros(src.rows * 2, src.cols * 2, src.type());

	for (int y_ = 0; y_ < dst.rows; y_++) {
		for (int x_ = 0; x_ < dst.cols; x_++) {
			int x = x_ / 2;
			int y = y_ / 2;
			dst.at<uchar>(y_, x_) = src.at<uchar>(y, x);
		}
	}

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}

void resizeBilinear(const Mat& src, Mat& dst, Size size)
{
	dst.create(size.height, size.width, CV_8U);

	int x1, y1, x2, y2;	double rx, ry, p, q, value;
	double sx = static_cast<double>(src.cols - 1) / (dst.cols - 1);
	double sy = static_cast<double>(src.rows - 1) / (dst.rows - 1);

	for (int y = 0; y < dst.rows; y++) {
		for (int x = 0; x < dst.cols; x++) {
			rx = sx * x;			ry = sy * y;
			x1 = cvFloor(rx);		y1 = cvFloor(ry);
			x2 = x1 + 1; if (x2 == src.cols) x2 = src.cols - 1;
			y2 = y1 + 1; if (y2 == src.rows) y2 = src.rows - 1;
			p = rx - x1;			q = ry - y1;

			value = (1. - p) * (1. - q) * src.at<uchar>(y1, x1)
				+ p * (1. - q) * src.at<uchar>(y1, x2)
				+ (1. - p) * q * src.at<uchar>(y2, x1)
				+ p * q * src.at<uchar>(y2, x2);

			dst.at<uchar>(y, x) = static_cast<uchar>(value + .5);
		}
	}
}

void resize3()
{
	Mat src = imread("camera.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst;
	resizeBilinear(src, dst, Size(600, 300));

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}

void resize4()
{
	Mat src = imread("rose.bmp");

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst1, dst2, dst3, dst4;
	resize(src, dst1, Size(), 4, 4, INTER_NEAREST);
	resize(src, dst2, Size(1920, 1280));
	resize(src, dst3, Size(1920, 1280), 0, 0, INTER_CUBIC);
	resize(src, dst4, Size(1920, 1280), 0, 0, INTER_LANCZOS4);

	imshow("src", src);
	imshow("dst1", dst1(Rect(400, 500, 400, 400)));
	imshow("dst2", dst2(Rect(400, 500, 400, 400)));
	imshow("dst3", dst3(Rect(400, 500, 400, 400)));
	imshow("dst4", dst4(Rect(400, 500, 400, 400)));
	waitKey();
}
