#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"

using namespace std;
using namespace cv;

int main()
{
	ocl::setUseOpenCL(false);

	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	CascadeClassifier face_cascade("haarcascade_frontalface_default.xml");

	if (face_cascade.empty()) {
		cerr << "Failed to open (face) xml file!" << endl;
		return -1;
	}

	TickMeter tm;
	tm.start();

	vector<Rect> faces;
	face_cascade.detectMultiScale(src, faces);

	tm.stop();
	cout << "Face detect: " << tm.getTimeMilli() << " ms." << endl;

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);

	for (size_t i = 0; i < faces.size(); i++) {
		rectangle(dst, faces[i], Scalar(255, 0, 255), 2, LINE_AA);
	}

#if 0
	CascadeClassifier eyes_cascade("haarcascade_eye.xml");
	if (eyes_cascade.empty()) {
		cerr << "Failed to open (eye) xml file!" << endl;
		return -1;
	}

	for (size_t i = 0; i < faces.size(); i++) {
		Mat faceROI = src(faces[i]);
		vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes);

		for (size_t j = 0; j < eyes.size(); j++) {
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
			circle(dst, eye_center, radius, Scalar(255, 0, 0), 2, LINE_AA);
		}
	}
#endif

//	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}
