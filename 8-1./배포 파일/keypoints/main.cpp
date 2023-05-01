#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"

using namespace std;
using namespace cv;

int main()
{
	ocl::setUseOpenCL(false); // for ORB time check

	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	TickMeter tm;
	tm.start();

	Ptr<Feature2D> detector = SIFT::create(); // SIFT, KAZE, AKAZE, ORB

	vector<KeyPoint> keypoints;
	detector->detect(src, keypoints);

	tm.stop();
	cout << "Elapsed time: " << tm.getTimeMilli() << "ms." << endl;
	cout << "keypoints.size(): " << keypoints.size() << endl;

	Mat dst;
	drawKeypoints(src, keypoints, dst, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("dst", dst);
	waitKey();
}