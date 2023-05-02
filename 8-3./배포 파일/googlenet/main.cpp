#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main(int argc, char* argv[])
{
	// Image load

	String filename = "space_shuttle.jpg";

	if (argc >= 2)
		filename = argv[1];

	Mat img = imread(filename, IMREAD_COLOR);

	if (img.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	// Network load

	String model = "bvlc_googlenet.caffemodel";
	String config = "deploy.prototxt";

	Net net = readNet(model, config);

	if (net.empty()) {
		cerr << "Network load failed!" << endl;
		return -1;
	}

	// Classes load

	ifstream fs("classification_classes_ILSVRC2012.txt");

	if (!fs.is_open()) {
		cerr << "Class file load failed!" << endl;
		return -1;
	}

	vector<String> classNames;
	string name;
	while (!fs.eof()) {
		getline(fs, name);
		if (name.length())
			classNames.push_back(name);
	}

	fs.close();

	// Inference

	Mat inputBlob = blobFromImage(img, 1, Size(224, 224), Scalar(104, 117, 123));
	net.setInput(inputBlob);
	Mat prob = net.forward();

	// Result check & display

	double maxVal;
	Point maxLoc;
	minMaxLoc(prob, NULL, &maxVal, NULL, &maxLoc);

	String str = format("%s (%4.2lf%%)", classNames[maxLoc.x].c_str(), maxVal * 100);
	putText(img, str, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 1, LINE_AA);

	imshow("img", img);
	waitKey();
}
