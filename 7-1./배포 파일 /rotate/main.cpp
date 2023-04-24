#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void on_rotate(int pos, void* data);

int main()
{
	Mat src = imread("lenna.bmp");

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	imshow("src", src);

	namedWindow("dst");
	createTrackbar("angle", "dst", 0, 360, on_rotate, (void*)&src);
	on_rotate(0, (void*)&src);

	waitKey();
}

void on_rotate(int pos, void* data)
{
	Mat src = *(Mat*)data;

	float degree = (float)pos;
	Point2f pt(src.cols / 2.f, src.rows / 2.f);
	Mat rot = getRotationMatrix2D(pt, degree, 1.0);

	Mat dst;
	warpAffine(src, dst, rot, Size());

	imshow("dst", dst);
}
