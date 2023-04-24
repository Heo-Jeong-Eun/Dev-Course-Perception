#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void split_bgr();
void split_hsv();
void split_ycrcb();

int main()
{
	split_bgr();
	split_hsv();
	split_ycrcb();
}

void split_bgr()
{
	Mat src = imread("candies.png", IMREAD_COLOR);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	vector<Mat> bgr_planes;
	split(src, bgr_planes);

	imshow("src", src);
	imshow("B plane", bgr_planes[0]);
	imshow("G plane", bgr_planes[1]);
	imshow("R plane", bgr_planes[2]);

	waitKey();
	destroyAllWindows();
}

void split_hsv()
{
	Mat src = imread("candies.png", IMREAD_COLOR);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat src_hsv;
	cvtColor(src, src_hsv, COLOR_BGR2HSV);

	vector<Mat> hsv_planes;
	split(src_hsv, hsv_planes);

	imshow("src", src);
	imshow("H plane", hsv_planes[0]);
	imshow("S plane", hsv_planes[1]);
	imshow("V plane", hsv_planes[2]);

	waitKey();
	destroyAllWindows();
}

void split_ycrcb()
{
	Mat src = imread("candies.png", IMREAD_COLOR);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat src_ycrcb;
	cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb);

	vector<Mat> ycrcb_planes;
	split(src_ycrcb, ycrcb_planes);

	imshow("src", src);
	imshow("Y plane", ycrcb_planes[0]);
	imshow("Cr plane", ycrcb_planes[1]);
	imshow("Cb plane", ycrcb_planes[2]);

	waitKey();
	destroyAllWindows();
}
