#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#define RGB2GRAY(r, g, b) ((4899*(r) + 9617*(g) + 1868*(b)) >> 14)

void ColorOp1();
void ColorOp2();

int main()
{
	ColorOp1();
	ColorOp2();
}

void ColorOp1()
{
	Mat src = imread("mandrill.bmp");

	if (src.empty()) {
		cerr << "Image laod failed!" << endl;
		return;
	}

#if 1
	Mat dst = Scalar(255, 255, 255) - src;
#else
	Mat dst(src.rows, src.cols, CV_8UC3);

	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			dst.at<Vec3b>(y, x) = Vec3b(255, 255, 255) - src.at<Vec3b>(y, x);
		}
	}
#endif

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
	destroyAllWindows();
}

void ColorOp2()
{
	Mat src = imread("mandrill.bmp");

	if (src.empty()) {
		cerr << "Image laod failed!" << endl;
		return;
	}

#if 0
	Mat dst;
	cvtColor(src, dst, COLOR_BGR2GRAY);
#else
	Mat dst(src.rows, src.cols, CV_8UC1);

	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			uchar b = src.at<Vec3b>(y, x)[0];
			uchar g = src.at<Vec3b>(y, x)[1];
			uchar r = src.at<Vec3b>(y, x)[2];

			//uchar gray = (uchar)(0.299 * r + 0.587 * g + 0.114 * b + 0.5);
			//dst.at<uchar>(y, x) = gray;

			dst.at<uchar>(y, x) = (uchar)RGB2GRAY(r, g, b);
		}
	}
#endif

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
	destroyAllWindows();
}
