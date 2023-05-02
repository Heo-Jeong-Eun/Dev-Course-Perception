#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(void)
{
	Mat src = imread("korea.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	VideoCapture cap1(0);

	if (!cap1.isOpened()) {
		cerr << "Camera open failed!" << endl;
		return -1;
	}

	VideoCapture cap2("korea.mp4");

	if (!cap2.isOpened()) {
		cerr << "Video open failed!" << endl;
		return -1;
	}
	
	// TODO:
}
