#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void sobel_impl();
void sobel_func();

int main()
{
	sobel_impl();
	sobel_func();
}

void sobel_impl()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dx = Mat::zeros(src.size(), CV_8UC1);
	Mat dy = Mat::zeros(src.size(), CV_8UC1);
	Mat mag = Mat::zeros(src.size(), CV_8UC1);

	for (int y = 1; y < src.rows - 1; y++) {
		for (int x = 1; x < src.cols - 1; x++) {
			int v1 = src.at<uchar>(y - 1, x + 1)
				+ src.at<uchar>(y, x + 1) * 2
				+ src.at<uchar>(y + 1, x + 1)
				- src.at<uchar>(y - 1, x - 1)
				- src.at<uchar>(y, x - 1) * 2
				- src.at<uchar>(y + 1, x - 1);
			int v2 = src.at<uchar>(y + 1, x + 1)
				+ src.at<uchar>(y + 1, x) * 2
				+ src.at<uchar>(y + 1, x + 1)
				- src.at<uchar>(y - 1, x + 1)
				- src.at<uchar>(y - 1, x) * 2
				- src.at<uchar>(y - 1, x + 1);
			dx.at<uchar>(y, x) = saturate_cast<uchar>(v1 + 128);
			dy.at<uchar>(y, x) = saturate_cast<uchar>(v2 + 128);
			mag.at<uchar>(y, x) = saturate_cast<uchar>(sqrt(v1 * v1 + v2 * v2));
		}
	}

	Mat edge = mag > 120;

	imshow("src", src);
	imshow("dx", dx);
	imshow("dy", dy);
	imshow("mag", mag);
	imshow("edge", edge);

	waitKey();
	destroyAllWindows();
}

void sobel_func()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dx, dy;
	Sobel(src, dx, CV_32FC1, 1, 0);
	Sobel(src, dy, CV_32FC1, 0, 1);

	Mat mag;
	magnitude(dx, dy, mag);
	mag.convertTo(mag, CV_8UC1);

	Mat edge = mag > 150;

	imshow("src", src);
	imshow("mag", mag);
	imshow("edge", edge);

	waitKey();
	destroyAllWindows();
}