#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"

using namespace std;
using namespace cv;

int main()
{
	ocl::setUseOpenCL(false);

	Mat src = imread("hongkong.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}
	
	cout << "getNumberOfCPUs(): " << getNumberOfCPUs() << endl;
	cout << "getNumThreads(): " << getNumThreads() << endl;
	cout << "Image size: " << src.size() << endl;

	namedWindow("src", WINDOW_NORMAL);
	namedWindow("dst", WINDOW_NORMAL);
	resizeWindow("src", 1280, 720);
	resizeWindow("dst", 1280, 720);

	Mat dst;
	TickMeter tm;

	// 1. Operator overloading
	tm.start();

	dst = 2 * src - 128;

	tm.stop();
	cout << "1. Operator overloading: " << tm.getTimeMilli() << " ms." << endl;

	imshow("src", src);
	imshow("dst", dst);
	waitKey();

	// 2. Pixel access by ptr()
	tm.reset();
	tm.start();

	dst = Mat::zeros(src.rows, src.cols, src.type());
	for (int j = 0; j < src.rows; j++) {
		uchar* pSrc = src.ptr<uchar>(j);
		uchar* pDst = dst.ptr<uchar>(j);
		for (int i = 0; i < src.cols; i++) {
			pDst[i] = saturate_cast<uchar>(2 * pSrc[i] - 128);
		}
	}

	tm.stop();
	cout << "2. Pixel access by ptr(): " << tm.getTimeMilli() << " ms." << endl;

	imshow("src", src);
	imshow("dst", dst);
	waitKey();

	// 3. LUT() function
	Mat lut(1, 256, CV_8U);
	uchar* p = lut.ptr(0);
	for (int i = 0; i < 256; i++) {
		p[i] = saturate_cast<uchar>(2 * i - 128);
	}

	tm.reset();
	tm.start();

	LUT(src, lut, dst);
	
	tm.stop();
	cout << "3. LUT() function: " << tm.getTimeMilli() << " ms." << endl;

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}
