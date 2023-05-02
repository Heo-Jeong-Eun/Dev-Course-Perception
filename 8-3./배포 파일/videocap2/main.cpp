#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	VideoCapture cap("korea.mp4");

	if (!cap.isOpened()) {
		cerr << "Video open failed!" << endl;
		return -1;
	}

	int frame_width = cvRound(cap.get(CAP_PROP_FRAME_WIDTH));
	int frame_height = cvRound(cap.get(CAP_PROP_FRAME_HEIGHT));
	int frame_count = cvRound(cap.get(CAP_PROP_FRAME_COUNT));
	int fps = cvRound(cap.get(CAP_PROP_FPS));
	cout << "Frame width: " << frame_width << endl;
	cout << "Frame height: " << frame_height << endl;
	cout << "Frame count: " << frame_count << endl;
	cout << "FPS: " << fps << endl;

	namedWindow("src");
	namedWindow("dst");

	Mat frame, gray, blr, edge;
	while (true) {
		cap >> frame;
		if (frame.empty())
			break;

		TickMeter tm;
		tm.start();

		cvtColor(frame, gray, COLOR_BGR2GRAY);
		bilateralFilter(gray, blr, -1, 10, 3);
		Canny(blr, edge, 50, 150);

		tm.stop();
		cout << "It took " << tm.getTimeMilli() << "ms." << endl;

		imshow("src", frame);
		imshow("dst", edge);

		if (waitKey(10) == 27)
			break;
	}
}