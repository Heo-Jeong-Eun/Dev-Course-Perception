#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;

void on_mouse(int event, int x, int y, int flags, void* userdata);

Mat norm_digit(Mat& src)
{
	CV_Assert(!src.empty() && src.type() == CV_8UC1);

	Mat src_bin;
	threshold(src, src_bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

	Mat labels, stats, centroids;
	int n = connectedComponentsWithStats(src_bin, labels, stats, centroids);

	Mat dst = Mat::zeros(src.rows, src.cols, src.type());
	for (int i = 1; i < n; i++) {
		if (stats.at<int>(i, 4) < 20) continue;

		int cx = cvRound(centroids.at<double>(i, 0));
		int cy = cvRound(centroids.at<double>(i, 1));

		double dx = 14 - cx;
		double dy = 14 - cy;

		Mat warpMat = (Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
		warpAffine(src, dst, warpMat, dst.size());
	}

	return dst;
}

int main()
{
	Net net = readNet("mnist.pb");
	//Net net = readNet("mnist.onnx");

	if (net.empty()) {
		cerr << "Network load failed!" << endl;
		return -1;
	}

	Mat img = Mat::zeros(400, 400, CV_8UC1);

	imshow("img", img);
	setMouseCallback("img", on_mouse, (void*)&img);

	while (true) {
		int c = waitKey();

		if (c == 27) {
			break;
		} else if (c == ' ') {
			Mat blr, resized;
			GaussianBlur(img, blr, Size(), 1.0);
			resize(blr, resized, Size(28, 28), 0, 0, INTER_AREA);

			Mat blob = blobFromImage(norm_digit(resized), 1/255.f, Size(28, 28));
			net.setInput(blob);
			Mat prob = net.forward();

			double maxVal;
			Point maxLoc;
			minMaxLoc(prob, NULL, &maxVal, NULL, &maxLoc);
			int digit = maxLoc.x;

			cout << digit << " (" << maxVal * 100 << "%)" <<endl;

			img.setTo(0);
			imshow("img", img);
		}
	}
}

Point ptPrev(-1, -1);

void on_mouse(int event, int x, int y, int flags, void* userdata)
{
	Mat img = *(Mat*)userdata;

	if (event == EVENT_LBUTTONDOWN) {
		ptPrev = Point(x, y);
	} else if (event == EVENT_LBUTTONUP) {
		ptPrev = Point(-1, -1);
	} else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {
		line(img, ptPrev, Point(x, y), Scalar::all(255), 40, LINE_AA, 0);
		ptPrev = Point(x, y);

		imshow("img", img);
	}
}
