#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("rice.png", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat dst1;
	threshold(src, dst1, 0, 255, THRESH_BINARY | THRESH_OTSU);

	int bw = src.cols / 4;
	int bh = src.rows / 4;

	Mat dst2 = Mat::zeros(src.rows, src.cols, CV_8UC1);

	for (int y = 0; y < 4; y++) {
		for (int x = 0; x < 4; x++) {
			Mat src_ = src(Rect(x*bw, y*bh, bw, bh));
			Mat dst_ = dst2(Rect(x*bw, y*bh, bw, bh));
			threshold(src_, dst_, 0, 255, THRESH_BINARY | THRESH_OTSU);
		}
	}

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);
	waitKey();
}
