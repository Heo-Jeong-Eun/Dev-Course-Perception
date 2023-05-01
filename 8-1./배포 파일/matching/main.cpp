#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"

using namespace std;
using namespace cv;

void keypoint_matching();
void good_matching();
void find_homography();

string file1 = "box.png";
string file2 = "box_in_scene.png";
Ptr<Feature2D> feature;

int main()
{
	ocl::setUseOpenCL(false);

	//file1 = "graf1.png";
	//file2 = "graf3.png";

	feature = SIFT::create();
	//feature = ORB::create();

	keypoint_matching();
	good_matching();
	find_homography();
}

void keypoint_matching()
{
	Mat src1 = imread(file1, IMREAD_GRAYSCALE);
	Mat src2 = imread(file2, IMREAD_GRAYSCALE);

	if (src1.empty() || src2.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	vector<KeyPoint> keypoints1, keypoints2;
	Mat desc1, desc2;
	feature->detectAndCompute(src1, Mat(), keypoints1, desc1);
	feature->detectAndCompute(src2, Mat(), keypoints2, desc2);

	Ptr<DescriptorMatcher> matcher = BFMatcher::create();
//	Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING); // for binary descriptors

	vector<DMatch> matches;
	matcher->match(desc1, desc2, matches);

	Mat dst;
	drawMatches(src1, keypoints1, src2, keypoints2, matches, dst);

	imshow("dst", dst);
	waitKey();
	destroyAllWindows();
}

void good_matching()
{
	Mat src1 = imread(file1, IMREAD_GRAYSCALE);
	Mat src2 = imread(file2, IMREAD_GRAYSCALE);

	if (src1.empty() || src2.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	vector<KeyPoint> keypoints1, keypoints2;
	Mat desc1, desc2;
	feature->detectAndCompute(src1, Mat(), keypoints1, desc1);
	feature->detectAndCompute(src2, Mat(), keypoints2, desc2);

	Ptr<DescriptorMatcher> matcher = BFMatcher::create();

#if 1
	vector<DMatch> matches;
	matcher->match(desc1, desc2, matches);

	std::sort(matches.begin(), matches.end());
	vector<DMatch> good_matches(matches.begin(), matches.begin() + 80);
#else
	vector<vector<DMatch>> matches;
	matcher->knnMatch(desc1, desc2, matches, 2);

	vector<DMatch> good_matches;
	for (const auto& m : matches) {
		if (m[0].distance / m[1].distance < 0.7)
			good_matches.push_back(m[0]);
	}
#endif

	Mat dst;
	drawMatches(src1, keypoints1, src2, keypoints2, good_matches, dst,
		Scalar::all(-1), Scalar::all(-1), vector<char>(),
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imshow("dst", dst);
	waitKey();
	destroyAllWindows();
}

void find_homography()
{
	Mat src1 = imread(file1, IMREAD_GRAYSCALE);
	Mat src2 = imread(file2, IMREAD_GRAYSCALE);

	if (src1.empty() || src2.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	TickMeter tm;
	tm.start();

	vector<KeyPoint> keypoints1, keypoints2;
	Mat desc1, desc2;
	feature->detectAndCompute(src1, Mat(), keypoints1, desc1);
	feature->detectAndCompute(src2, Mat(), keypoints2, desc2);

	Ptr<DescriptorMatcher> matcher = BFMatcher::create();

#if 1
	vector<DMatch> matches;
	matcher->match(desc1, desc2, matches);

	std::sort(matches.begin(), matches.end());
	vector<DMatch> good_matches(matches.begin(), matches.begin() + 80);
#else
	vector<vector<DMatch>> matches;
	matcher->knnMatch(desc1, desc2, matches, 2);

	vector<DMatch> good_matches;
	for (const auto& m : matches) {
		if (m[0].distance / m[1].distance < 0.7)
			good_matches.push_back(m[0]);
	}
#endif



	vector<Point2f> pts1, pts2;
	for (size_t i = 0; i < good_matches.size(); i++) {
		pts1.push_back(keypoints1[good_matches[i].queryIdx].pt);
		pts2.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(pts1, pts2, RANSAC);

	tm.stop();
	cout << "time: " << tm.getTimeMilli() << endl;

	Mat dst;
	drawMatches(src1, keypoints1, src2, keypoints2, good_matches, dst,
		Scalar::all(-1), Scalar::all(-1), vector<char>(),
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	vector<Point2f> corners1, corners2;
	corners1.push_back(Point2f(0, 0));
	corners1.push_back(Point2f(src1.cols - 1.f, 0));
	corners1.push_back(Point2f(src1.cols - 1.f, src1.rows - 1.f));
	corners1.push_back(Point2f(0, src1.rows - 1.f));
	perspectiveTransform(corners1, corners2, H);

	vector<Point> corners_dst;
	for (Point2f pt : corners2) {
		corners_dst.push_back(Point(cvRound(pt.x + src1.cols), cvRound(pt.y)));
	}

	polylines(dst, corners_dst, true, Scalar(0, 255, 0), 2, LINE_AA);

	imshow("dst", dst);
	waitKey();
	destroyAllWindows();
}