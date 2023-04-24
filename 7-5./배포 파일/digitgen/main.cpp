#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("digit_consolas.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat src_bin;
	threshold(src, src_bin, 128, 255, THRESH_BINARY_INV);

	Mat labels, stats, centroids;
	int label_cnt = connectedComponentsWithStats(src_bin, labels, stats, centroids);
	
	for (int i = 1; i < label_cnt; i++) {
		int sx = stats.at<int>(i, 0);
		int sy = stats.at<int>(i, 1);
		int w = stats.at<int>(i, 2);
		int h = stats.at<int>(i, 3);

		Mat digit;
		resize(src(Rect(sx, sy, w, h)), digit, Size(100, 150));
		String filename = cv::format("temp%d.bmp", i - 1);
		imwrite(filename, digit);
		cout << filename << " file is generated!" << endl;
		//imshow("digit", );
		//waitKey();
	}
}