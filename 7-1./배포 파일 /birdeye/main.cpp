#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	VideoCapture cap("../../data/test_video.mp4");

	if (!cap.isOpened()) {
		cerr << "Video open failed!" << endl;
		return -1;
	}

	Mat src;
	while (true) {
		cap >> src;

		if (src.empty())
			break;

		int w = 500, h = 260;

		vector<Point2f> src_pts(4);
		vector<Point2f> dst_pts(4);

		src_pts[0] = Point2f(474, 400);	src_pts[1] = Point2f(710, 400);
		src_pts[2] = Point2f(866, 530); src_pts[3] = Point2f(366, 530);

		dst_pts[0] = Point2f(0, 0);		dst_pts[1] = Point2f(w - 1, 0);
		dst_pts[2] = Point2f(w - 1, h - 1);	dst_pts[3] = Point2f(0, h - 1);

		Mat per_mat = getPerspectiveTransform(src_pts, dst_pts);

		Mat dst;
		warpPerspective(src, dst, per_mat, Size(w, h));

#if 1
		vector<Point> pts;
		for (auto pt : src_pts) {
			pts.push_back(Point(pt.x, pt.y));
		}
		polylines(src, pts, true, Scalar(0, 0, 255), 2, LINE_AA);
#endif

		imshow("src", src);
		imshow("dst", dst);

		if (waitKey(10) == 27)
			break;
	}
}
