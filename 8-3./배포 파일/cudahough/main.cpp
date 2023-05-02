#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/cudaimgproc.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

void houghLinesCPU(const Mat& src, Mat& dst);
void houghLinesGPU(const Mat& src, Mat& dst);

int main()
{
	// Print cuda device info.

	cout << "device count: " << getCudaEnabledDeviceCount() << endl;

	int device_id = getDevice();
	cout << "device id: " << device_id << endl;

	printShortCudaDeviceInfo(device_id);
	cout << endl;

	// Hough line detection

	Mat src = imread("building.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat dst_cpu, dst_gpu;

	houghLinesCPU(src, dst_cpu);
	houghLinesGPU(src, dst_gpu);

	imshow("dst_cpu", dst_cpu);
	imshow("dst_gpu", dst_gpu);
	waitKey();
}

void houghLinesCPU(const Mat& src, Mat& dst)
{
	int64 t1 = getTickCount();

	Mat edges;
	Canny(src, edges, 100, 200, 3);

	vector<Vec4i> lines;
	HoughLinesP(edges, lines, 1, CV_PI / 180, 160, 50, 5);
	
	int64 t2 = getTickCount();
	cout << "Canny & Hough (CPU version) took " << (t2 - t1) * 1000 / getTickFrequency() << " ms." << endl;
	cout << "It found " << lines.size() << " lines." << endl;
	cout << endl;

	cv::cvtColor(src, dst, COLOR_GRAY2BGR);

	for (size_t i = 0; i < lines.size(); i++) {
		Vec4i l = lines[i];
		line(dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA);
	}
}

void houghLinesGPU(const Mat& src, Mat& dst)
{
	GpuMat d_src, d_tmp;
	int64 t1, t2;
	double ms;

	t1 = getTickCount();
	d_tmp.upload(src);
	t2 = getTickCount();
	ms = (t2 - t1) * 1000 / getTickFrequency();
	cout << "GpuMat upload (1st try) took " << ms << " ms." << endl << endl;

	t1 = getTickCount();
	d_src.upload(src);
	t2 = getTickCount();
	ms = (t2 - t1) * 1000 / getTickFrequency();
	cout << "GpuMat upload (2nd try) took " << ms << " ms." << endl << endl;

	t1 = getTickCount();

	GpuMat d_edges;
	Ptr<CannyEdgeDetector> canny = createCannyEdgeDetector(100, 200, 3);
	canny->detect(d_src, d_edges);

	GpuMat d_lines;
	Ptr<cuda::HoughSegmentDetector> hough = cuda::createHoughSegmentDetector(1.0f, (float)(CV_PI / 180.0f), 50, 5);
	hough->detect(d_edges, d_lines);

	t2 = getTickCount();
	ms = (t2 - t1) * 1000 / getTickFrequency();
	cout << "Canny & Hough (GPU version) took " << ms << " ms." << endl;
	cout << "It found " << d_lines.cols << " lines." << endl;

	cv::cvtColor(src, dst, COLOR_GRAY2BGR);

	vector<Vec4i> lines_gpu;
	if (!d_lines.empty())
	{
		lines_gpu.resize(d_lines.cols);
		Mat h_lines(1, d_lines.cols, CV_32SC4, &lines_gpu[0]);
		d_lines.download(h_lines);
	}

	for (size_t i = 0; i < lines_gpu.size(); ++i)
	{
		Vec4i l = lines_gpu[i];
		line(dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
	}
}
