#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

#if 0
	float alpha = 1.0f;
	Mat dst = src + (src - 128) * alpha;
#else
	float alpha = 1.0f;
	int m = (int)mean(src)[0];
	Mat dst = src + (src - m) * alpha;
#endif

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
}
