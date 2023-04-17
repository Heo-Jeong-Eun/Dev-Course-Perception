#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(void)
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	// 히스토그램
	int hist[256] = {};
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			hist[src.at<uchar>(y, x)]++;
		}
	}

	// 정규화된 히스토그램
	int size = (int)src.total();
	float nhist[256] = {};
	for (int i = 0; i < 256; i++) {
		nhist[i] = (float)hist[i] / size;
	}

	// 히스토그램 그래프 그리기
	int histMax = 0;
	for (int i = 0; i < 256; i++) {
		if (hist[i] > histMax) histMax = hist[i];
	}

	Mat imgHist(100, 256, CV_8UC1, Scalar(255));
	for (int i = 0; i < 256; i++) {
		line(imgHist, Point(i, 100),
			Point(i, 100 - cvRound(hist[i] * 100 / histMax)), Scalar(0));
	}

	imshow("src", src);
	imshow("hist", imgHist);
	waitKey();
}
