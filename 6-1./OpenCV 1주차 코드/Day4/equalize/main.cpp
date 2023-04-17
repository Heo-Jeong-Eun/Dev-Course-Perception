#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

Mat calcGrayHist(const Mat& img)
{
	CV_Assert(img.type() == CV_8UC1);

	Mat hist;
	int channels[] = { 0 };
	int dims = 1;
	const int histSize[] = { 256 };
	float graylevel[] = { 0, 256 };
	const float* ranges[] = { graylevel };

	calcHist(&img, 1, channels, noArray(), hist, dims, histSize, ranges, true);

	return hist;
}

Mat getGrayHistImage(const Mat& hist)
{
	CV_Assert(!hist.empty());
	CV_Assert(hist.type() == CV_32FC1);
	CV_Assert(hist.size() == Size(1, 256));

	double histMax = 0.;
	minMaxLoc(hist, 0, &histMax);

	Mat imgHist(100, 256, CV_8UC1, Scalar(255));
	for (int i = 0; i < 256; i++) {
		line(imgHist, Point(i, 100),
			Point(i, 100 - cvRound(hist.at<float>(i) * 100 / histMax)), Scalar(0));
	}

	return imgHist;
}

int main()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

#if 1
	Mat dst;
	equalizeHist(src, dst);
#else
	Mat dst(src.rows, src.cols, src.type());

	int hist[256] = {};
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			hist[src.at<uchar>(y, x)]++;

	int size = (int)src.total();
	float cdf[256] = {};
	cdf[0] = float(hist[0]) / size;
	for (int i = 1; i < 256; i++)
		cdf[i] = cdf[i - 1] + float(hist[i]) / size;

	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			dst.at<uchar>(y, x) = uchar(cdf[src.at<uchar>(y, x)] * 255);
		}
	}
#endif

	imshow("src", src);
	imshow("dst", dst);
	imshow("hist_src", getGrayHistImage(calcGrayHist(src)));
	imshow("hist_dst", getGrayHistImage(calcGrayHist(dst)));

	waitKey();
}
