#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("lenna.bmp");

	if (img.empty()) {
		cerr << "Image laod failed!" << endl;
		return -1;
	}

	namedWindow("image");
	imshow("image", img);
	waitKey();
	destroyAllWindows();
}
