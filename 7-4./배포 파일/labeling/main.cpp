#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	uchar data[] = {
		0, 0, 1, 1, 0, 0, 0, 0,
		1, 1, 1, 1, 0, 0, 1, 0,
		1, 1, 1, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 1, 0,
		0, 0, 0, 1, 1, 1, 1, 0,
		0, 0, 1, 1, 0, 0, 1, 0,
		0, 0, 1, 1, 1, 1, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	};

	Mat src(8, 8, CV_8UC1, data);

#if 0
	Mat labels;
	int num_labels = connectedComponents(src, labels);

	cout << "src:\n" << src << endl;
	cout << "number of labels: " << num_labels << endl;
	cout << "labels:\n" << labels << endl;
#else
	Mat labels, stats, centroids;
	int num_labels = connectedComponentsWithStats(src, labels, stats, centroids);

	cout << "src:\n" << src << endl;
	cout << "number of labels: " << num_labels << endl;
	cout << "labels:\n" << labels << endl;
	cout << "stats:\n" << stats << endl;
	cout << "centroids:\n" << centroids << endl;
#endif
}
