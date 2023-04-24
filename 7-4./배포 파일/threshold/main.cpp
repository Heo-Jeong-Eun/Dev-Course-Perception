#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int t_value = 128;
void on_trackbar_threshold(int, void*);
Mat src, dst;

int main(int argc, char* argv[])
{
	String filename = "neutrophils.png";

	if (argc > 1) {
		filename = argv[1];
	}

	src = imread(filename, IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	namedWindow("src");
	imshow("src", src);

	namedWindow("dst");
	createTrackbar("Threshold", "dst", &t_value, 255, on_trackbar_threshold);
	on_trackbar_threshold(0, 0); // Call the function to initialize

	waitKey();
}

void on_trackbar_threshold(int, void*)
{
	threshold(src, dst, t_value, 255, THRESH_BINARY);
	imshow("dst", dst);
}
