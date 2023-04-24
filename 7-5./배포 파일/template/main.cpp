#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("circuit.bmp", IMREAD_GRAYSCALE);
	Mat tmpl = imread("crystal.bmp", IMREAD_GRAYSCALE);

#if 0
	src = imread("wheres_wally.jpg", IMREAD_GRAYSCALE); // IMREAD_COLOR
	tmpl = imread("wally.bmp", IMREAD_GRAYSCALE);
#endif

	if (src.empty() || tmpl.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

#if 1
	src = src + 50;

	Mat noise(src.size(), CV_32S);
	randn(noise, 0, 10); // 10, 50, 100
	add(src, noise, src, noArray(), CV_8U);
#endif

#if 1
	GaussianBlur(src, src, Size(), 1);
	GaussianBlur(tmpl, tmpl, Size(), 1);
#endif

#if 0
	resize(src, src, Size(), 0.9, 0.9); // 0.8, 0.7
#endif

#if 0
	Point2f cp(src.cols / 2.f, src.rows / 2.f);
	Mat rot = getRotationMatrix2D(cp, 10, 1); // 20, 30
	warpAffine(src, src, rot, src.size());
#endif

	Mat res, res_norm;
	matchTemplate(src, tmpl, res, TM_CCOEFF_NORMED);
	normalize(res, res_norm, 0, 255, NORM_MINMAX, CV_8U);

	double maxv;
	Point maxloc;
	minMaxLoc(res, 0, &maxv, 0, &maxloc);

	cout << "maxv: " << maxv << endl;
	cout << "maxloc: " << maxloc << endl;

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);

	rectangle(dst, Rect(maxloc.x, maxloc.y, tmpl.cols, tmpl.rows), Scalar(0, 0, 255), 2);

//	imshow("src", src);
	imshow("tmpl", tmpl);
	imshow("res_norm", res_norm);
	imshow("dst", dst);
	waitKey();
}