#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image laod failed!" << endl;
		return -1;
	}

	Mat dst = 255 - src;

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}
