#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void sharpen_mean();
void sharpen_gaussian();

int main(void)
{
	sharpen_mean();
	sharpen_gaussian();
}

void sharpen_mean()
{
	Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	float sharpen[] = {
		-1 / 9.f, -1 / 9.f, -1 / 9.f,
		-1 / 9.f, 17 / 9.f, -1 / 9.f,
		-1 / 9.f, -1 / 9.f, -1 / 9.f
	};
	Mat kernel(3, 3, CV_32F, sharpen);

	Mat dst;
	filter2D(src, dst, -1, kernel);

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void sharpen_gaussian()
{
	Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	imshow("src", src);

	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	for (int sigma = 1; sigma <= 5; sigma++) {
		Mat blr;
		GaussianBlur(srcf, blr, Size(), sigma);

		float alpha = 1.0f;
		Mat dst = (1.f + alpha) * srcf - alpha * blr;

		dst.convertTo(dst, CV_8UC1);

		String desc = format("sigma: %d", sigma);
		putText(dst, desc, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);

		imshow("dst", dst);
		waitKey();
	}

	destroyAllWindows();
}