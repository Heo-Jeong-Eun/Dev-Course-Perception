#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(void)
{
	Mat src = imread("rice.png", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	// 지역 이진화
	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
	int bw = src.cols / 4;
	int bh = src.rows / 4;

	for (int y = 0; y < 4; y++) {
		for (int x = 0; x < 4; x++) {
			Mat src_ = src(Rect(x*bw, y*bh, bw, bh));
			Mat dst_ = dst(Rect(x*bw, y*bh, bw, bh));
			threshold(src_, dst_, 0, 255, THRESH_BINARY | THRESH_OTSU);
		}
	}

	// 레이블링 함수를 이용한 흰색 객체 갯수 구하기
	Mat labels;
	
	int cnt1 = connectedComponents(dst, labels);
	cout << "# of objects in dst: " << cnt1 - 1 << endl;

	Mat dst2;
	morphologyEx(dst, dst2, MORPH_OPEN, Mat());
	
	int cnt2 = connectedComponents(dst2, labels);
	cout << "# of objects in dst2: " << cnt2 - 1 << endl;

	imshow("src", src);
	imshow("dst", dst);
	imshow("dst2", dst2);
	waitKey();
}
