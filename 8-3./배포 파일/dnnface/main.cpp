#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace cv::dnn;
using namespace std;

const String model = "res10_300x300_ssd_iter_140000_fp16.caffemodel";
const String config = "deploy.prototxt";

//const String model = "opencv_face_detector_uint8.pb";
//const String config = "opencv_face_detector.pbtxt";

int main(void)
{
	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cerr << "Camera open failed!" << endl;
		return -1;
	}

	Net net = readNet(model, config);

	if (net.empty()) {
		cerr << "Net open failed!" << endl;
		return -1;
	}

	Mat frame;
	while (true) {
		cap >> frame;

		if (frame.empty())
			break;

		Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123));
		net.setInput(blob);         // "data"
		Mat res = net.forward();    // "detection_out"

		vector<double> layersTimings;
		double inf_ms = net.getPerfProfile(layersTimings) * 1000 / getTickFrequency();

		String info = format("FPS: %7.3f; time: %5.3fms", 1000 / inf_ms, inf_ms);
		putText(frame, info, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 1, LINE_AA);

		Mat detect(res.size[2], res.size[3], CV_32FC1, res.ptr<float>());

		// detect는 200x7 float 행렬.
		// 200개의 얼굴 후보 영역을 찾고, 각 얼굴마다 7개의 파라미터를 가짐.
		// [0, 1, confidence, x1, y1, x2, y2]
		// confidence는 0~1 사이의 실수. 내림차순으로 정렬되어 있음.
		// x1, y1, x2, y2도 0~1 사이의 실수. 실제 좌표는 영상의 width, height를 곱해야 함.

		for (int i = 0; i < detect.rows; i++) {
			float confidence = detect.at<float>(i, 2);

			if (confidence < 0.5f) break;

			int x1 = cvRound(detect.at<float>(i, 3) * frame.cols);
			int y1 = cvRound(detect.at<float>(i, 4) * frame.rows);
			int x2 = cvRound(detect.at<float>(i, 5) * frame.cols);
			int y2 = cvRound(detect.at<float>(i, 6) * frame.rows);

			rectangle(frame, Rect(Point(x1, y1), Point(x2, y2)), Scalar(0, 255, 0));

			String label = format("Face: %4.3f", confidence);
			putText(frame, label, Point(x1, y1 - 2), FONT_HERSHEY_SIMPLEX, 0.7, 
				Scalar(0, 255, 0), 1, LINE_AA);
		}

		imshow("frame", frame);
		if (waitKey(1) == 27)
			break;
	}
}
