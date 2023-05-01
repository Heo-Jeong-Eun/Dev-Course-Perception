#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

Mat img;
Point ptPrev(-1, -1);

void on_mouse(int event, int x, int y, int flags, void*)
{
	if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
		return;
	if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
		ptPrev = Point(-1, -1);
	else if (event == EVENT_LBUTTONDOWN)
		ptPrev = Point(x, y);
	else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
	{
		Point pt(x, y);
		if (ptPrev.x < 0)
			ptPrev = pt;
		line(img, ptPrev, pt, Scalar::all(255), 40, LINE_AA, 0);
		ptPrev = pt;

		imshow("img", img);
	}
}

Mat norm_digit(Mat& src)
{
	CV_Assert(!src.empty() && src.type() == CV_8UC1);

	Mat src_bin;
	threshold(src, src_bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

	Mat labels, stats, centroids;
	int n = connectedComponentsWithStats(src_bin, labels, stats, centroids);

	Mat dst = Mat::zeros(src.rows, src.cols, src.type());
	for (int i = 1; i < n; i++) {
		if (stats.at<int>(i, 4) < 10) continue;

		int cx = cvRound(centroids.at<double>(i, 0));
		int cy = cvRound(centroids.at<double>(i, 1));

		double dx = 10 - cx;
		double dy = 10 - cy;

		Mat warpMat = (Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
		warpAffine(src, dst, warpMat, dst.size());
	}

	return dst;
}

int main()
{
	Mat digits = imread("digits.png", IMREAD_GRAYSCALE);

	if (digits.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

#if _DEBUG
	HOGDescriptor hog(Size(20, 20), // _winSize
		Size(8, 8),		// _blockSize
		Size(4, 4),		// _blockStride,
		Size(4, 4),		// _cellSize,
		9);				// _nbins,
#else
	HOGDescriptor hog(Size(20, 20), // _winSize
		Size(10, 10),	// _blockSize
		Size(5, 5),		// _blockStride,
		Size(5, 5),		// _cellSize,
		9);				// _nbins,
#endif

	size_t descriptor_size = hog.getDescriptorSize();
	cout << "Descriptor Size : " << descriptor_size << endl;

	Mat train_hog, train_labels;

	for (int j = 0; j < 50; j++) {
		for (int i = 0; i < 100; i++) {
			Mat roi = digits(Rect(i * 20, j * 20, 20, 20)).clone();

			vector<float> desc;
			hog.compute(roi, desc);

			Mat desc_mat(desc, true);
			train_hog.push_back(desc_mat.t());
			train_labels.push_back(j / 5);
		}
	}

	Ptr<SVM> svm = SVM::create();
//	Ptr<SVM> svm = SVM::load("svmdigits.yml");

	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);

#if 1
	svm->setGamma(0.50625);
	svm->setC(2.5);
	svm->train(train_hog, ROW_SAMPLE, train_labels);
#else
	svm->trainAuto(train_hog, ROW_SAMPLE, train_labels);
#endif
//	svm->save("svmdigits.yml");

	// 입력 이미지 생성
	img = Mat::zeros(400, 400, CV_8U);

	imshow("img", img);
	setMouseCallback("img", on_mouse);

	while (true) {
		int c = waitKey();

		if (c == 27) {
			break;
		} else if (c == ' ') {
			Mat img_blur, img_resize;
			GaussianBlur(img, img_blur, Size(), 1);
			resize(img_blur, img_resize, Size(20, 20), 0, 0, INTER_AREA);

			vector<float> desc;
			hog.compute(img_resize, desc);

			Mat desc_mat(desc, true);
			float res = svm->predict(desc_mat.t());
			cout << cvRound(res) << endl;

			img.setTo(0);
			imshow("img", img);
		} else if (c == 'c') {
			img.setTo(0);
			imshow("img", img);
		}
	}
}
