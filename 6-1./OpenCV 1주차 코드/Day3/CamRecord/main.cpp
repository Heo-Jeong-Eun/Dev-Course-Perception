#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cerr << "Camera open failed!" << endl;
		return -1;
	}

	int  fourcc = VideoWriter::fourcc('X', 'V', 'I', 'D');
	double fps = 30;
	Size sz((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));

	cout << "FPS = " << fps << endl;
	cout << "Size = " << sz << endl;

	VideoWriter output("output.avi", fourcc, fps, sz);

	if (!output.isOpened()) {
		cerr << "output.avi open failed!" << endl;
		return -1;
	}

	int delay = cvRound(1000 / fps);
	Mat frame, edge;
	while (true) {
		cap >> frame;
		if (frame.empty())
			break;

		Canny(frame, edge, 50, 150);
		cvtColor(edge, edge, COLOR_GRAY2BGR);

		output << edge;

		imshow("frame", frame);
		imshow("edge", edge);

		if (waitKey(delay) == 27)
			break;
	}

	cout << "output.avi file is created!!!" << endl;
	
	output.release();
	cap.release();
	destroyAllWindows();
}
