#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

int main()
{
	Mat train = Mat_<float>({ 8, 2 }, {
		150, 200, 200, 250, 100, 250, 150, 300,
		350, 100, 400, 200, 400, 300, 350, 400 });
	Mat label = Mat_<int>({ 8, 1 }, { 0, 0, 0, 0, 1, 1, 1, 1 });

	Ptr<SVM> svm = SVM::create();

#if 1
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->trainAuto(train, ROW_SAMPLE, label);
#else
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->trainAuto(train, ROW_SAMPLE, label);
#endif

	cout << svm->getC() << endl;
	cout << svm->getGamma() << endl;

	Mat img = Mat::zeros(Size(500, 500), CV_8UC3);

	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			Mat test = Mat_<float>({ 1, 2 }, { (float)x, (float)y });
			int res = cvRound(svm->predict(test));

			if (res == 0)
				img.at<Vec3b>(y, x) = Vec3b(128, 128, 255); // R
			else
				img.at<Vec3b>(y, x) = Vec3b(128, 255, 128); // G
		}
	}

	for (int i = 0; i < train.rows; i++) {
		int x = cvRound(train.at<float>(i, 0));
		int y = cvRound(train.at<float>(i, 1));
		int l = label.at<int>(i, 0);

		if (l == 0)
			circle(img, Point(x, y), 5, Scalar(0, 0, 128), -1, LINE_AA); // R
		else
			circle(img, Point(x, y), 5, Scalar(0, 128, 0), -1, LINE_AA); // G
	}

	imshow("svm", img);
	waitKey();
}
