#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

#if 1
	Mat dst = Mat::zeros(src.size(), CV_8UC1);

	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			int x_ = x + 100;
			int y_ = y + 100;
			if (x_ < 0 || x_ >= dst.cols) continue;
			if (y_ < 0 || y_ >= dst.rows) continue;
			dst.at<uchar>(y_, x_) = src.at<uchar>(y, x);
		}
	}
#else
	Mat trans = (Mat_<float>(2, 3) << 1, 0, 100, 0, 1, 100);

	Mat dst;
	warpAffine(src, dst, trans, Size());
#endif

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}
