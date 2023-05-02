#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;

int main()
{
	// These files can be downloaded here: https://pjreddie.com/darknet/yolo/

	const String config = "yolov3.cfg";
	const String model = "yolov3.weights";
	const float confThreshold = 0.5f;
	const float nmsThreshold = 0.4f;

	Net net = readNet(config, model);

	if (net.empty()) {
		cerr << "Network load failed!" << endl;
		return -1;
	}

#if 0
	net.setPreferableTarget(DNN_TARGET_CPU);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
#else
	net.setPreferableTarget(DNN_TARGET_CUDA);
	net.setPreferableBackend(DNN_BACKEND_CUDA);
#endif

	// Class name file can be downloaded here: https://github.com/pjreddie/darknet/tree/master/data/

	vector<string> classNamesVec;
	ifstream classNamesFile("coco.names");

	if (classNamesFile.is_open()) {
		string className = "";
		while (std::getline(classNamesFile, className))
			classNamesVec.push_back(className);
	}

	// Video file open

	VideoCapture cap("Pexels Videos 1721294.mp4");

	if (!cap.isOpened()) {
		cerr << "Video open failed!" << endl;
		return -1;
	}

	// Get the names of the output layers
	vector<String> outputLayers = net.getUnconnectedOutLayersNames();

	Mat frame;
	while (true) {
		cap >> frame;

		if (frame.empty())
			break;

		// Convert Mat to batch of images
		Mat blob = blobFromImage(frame, 1 / 255.f, Size(416, 416), Scalar(), true);

		// Set the network input
		net.setInput(blob);

		// compute output
		vector<Mat> outs;
		net.forward(outs, outputLayers);

		vector<double> layersTimings;
		double time_ms = net.getPerfProfile(layersTimings) * 1000 / getTickFrequency();
		putText(frame, format("FPS: %.2f ; time: %.2f ms", 1000.f / time_ms, time_ms),
			Point(20, 30), 0, 0.75, Scalar(0, 0, 255), 1, LINE_AA);

		vector<int> classIds;
		vector<float> confidences;
		vector<Rect> boxes;

		for (auto& out : outs) {
			// Scan through all the bounding boxes output from the network and keep only the
			// ones with high confidence scores. Assign the box's class label as the class
			// with the highest score for the box.
			float* data = (float*)out.data;
			for (int j = 0; j < out.rows; ++j, data += out.cols) {
				Mat scores = out.row(j).colRange(5, out.cols);
				double confidence;
				Point classIdPoint;
				
				// Get the value and location of the maximum score
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

				if (confidence > confThreshold) {
					int cx = (int)(data[0] * frame.cols);
					int cy = (int)(data[1] * frame.rows);
					int bw = (int)(data[2] * frame.cols);
					int bh = (int)(data[3] * frame.rows);
					int sx = cx - bw / 2;
					int sy = cy - bh / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(Rect(sx, sy, bw, bh));
				}
			}
		}

		// Perform non maximum suppression to eliminate redundant overlapping boxes with
		// lower confidences
		vector<int> indices;
		NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

		for (size_t i = 0; i < indices.size(); ++i) {
			int idx = indices[i];
			int sx = boxes[idx].x;
			int sy = boxes[idx].y;

			rectangle(frame, boxes[idx], Scalar(0, 255, 0));

			string label = format("%.2f", confidences[idx]);
			label = classNamesVec[classIds[idx]] + ":" + label;
			int baseLine = 0;
			Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
			rectangle(frame, Rect(sx, sy, labelSize.width, labelSize.height + baseLine),
				Scalar(0, 255, 0), FILLED);
			putText(frame, label, Point(sx, sy + labelSize.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(), 1, LINE_AA);
		}

		imshow("frame", frame);
		if (waitKey(1) == 27)
			break;
	}
}
