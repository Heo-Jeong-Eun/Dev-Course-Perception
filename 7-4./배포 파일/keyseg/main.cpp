#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(void)
{
	Mat src = imread("keyboard.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat src_bin;
	threshold(src, src_bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

	Mat labels, stats, centroids;
	int cnt = connectedComponentsWithStats(src_bin, labels, stats, centroids);
	cout << cnt << endl;

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);

	for (int i = 1; i < cnt; i++) {
		int* p = stats.ptr<int>(i);

		if (p[4] < 20) continue;

		rectangle(dst, Rect(p[0], p[1], p[2], p[3]), Scalar(0, 255, 255));
	}

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}
