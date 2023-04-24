#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int block_size = 51;
Mat src, dst;

void on_trackbar(int, void*)
{
	int bsize = block_size;

	if ((bsize & 0x00000001) == 0)
		bsize--;

	if (bsize < 3)
		bsize = 3;

	adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, bsize, 5);
	imshow("dst", dst);
}

int main()
{
	src = imread("sudoku.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	namedWindow("src");
	imshow("src", src);

	namedWindow("dst");
	createTrackbar("Block Size", "dst", &block_size, 201, on_trackbar);
	on_trackbar(0, 0); // Call the function to initialize

	waitKey();
}
