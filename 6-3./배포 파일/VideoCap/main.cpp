#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	VideoCapture cap;
	cap.open(0);
//	cap.open("../../data/test_video.mp4");

	if (!cap.isOpened()) {
		cerr << "Camera open failed!" << endl;
		return -1;
	}

	int w = cvRound(cap.get(CAP_PROP_FRAME_WIDTH));
	int h = cvRound(cap.get(CAP_PROP_FRAME_HEIGHT));
	
	cout << "width: " << w << endl;
	cout << "height: " << h << endl;

	Mat frame, edge;
	while (true) {
		cap >> frame;

		if (frame.empty()) {
			cerr << "Empty frame!" << endl;
			break;
		}

		Canny(frame, edge, 50, 150);
		imshow("frame", frame);
		imshow("edge", edge);

		if (waitKey(0) == 27)
			break;
	}
	
	cap.release();
	destroyAllWindows();
}