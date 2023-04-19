#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(void)
{
	Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	imshow("src", src);

	Mat dst;
	for (int sigma = 1; sigma <= 5; sigma++) {
		TickMeter tm;
		tm.start();

		GaussianBlur(src, dst, Size(0, 0), (double)sigma);
		
		tm.stop();
		cout << "sigma: " << sigma << ", time: " << tm.getTimeMilli() << " ms." << endl;

		String desc = format("Sigma = %d", sigma);
		putText(dst, desc, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);

		imshow("dst", dst);
		waitKey();
	}
}
