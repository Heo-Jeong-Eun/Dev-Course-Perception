#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"

using namespace std;
using namespace cv;

class ParallelContrast : public ParallelLoopBody
{
public:
	ParallelContrast(Mat& src, Mat& dst, const float alpha)
		: m_src(src), m_dst(dst), m_alpha(alpha)
	{
		m_dst = Mat::zeros(src.rows, src.cols, src.type());
	}

	virtual void operator ()(const Range& range) const
	{
		for (int r = range.start; r < range.end; r++)
		{
			uchar* pSrc = m_src.ptr<uchar>(r);
			uchar* pDst = m_dst.ptr<uchar>(r);

			for (int x = 0; x < m_src.cols; x++)
				pDst[x] = saturate_cast<uchar>((1 + m_alpha)*pSrc[x] - 128 * m_alpha);
		}
	}

	ParallelContrast& operator =(const ParallelContrast &) {
		return *this;
	};

private:
	Mat& m_src;
	Mat& m_dst;
	float m_alpha;
};

int main()
{
	ocl::setUseOpenCL(false);

	Mat src = imread("hongkong.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	cout << "getNumberOfCPUs(): " << getNumberOfCPUs() << endl;
	cout << "getNumThreads(): " << getNumThreads() << endl;
	cout << "Image size: " << src.size() << endl;

	namedWindow("src", WINDOW_NORMAL);
	namedWindow("dst", WINDOW_NORMAL);
	resizeWindow("src", 1280, 720);
	resizeWindow("dst", 1280, 720);

	Mat dst;
	TickMeter tm;
	float alpha = 1.f;

	resize(src, dst, Size(100, 100));

	// 1. Operator overloading
	tm.start();

	dst = (1 + alpha) * src - 128 * alpha;

	tm.stop();
	cout << "1. Operator overloading: " << tm.getTimeMilli() << " ms." << endl;

	imshow("src", src);
	imshow("dst", dst);
	waitKey();

	// 2. Pixel access by at() (No parallel)
	dst = Mat::zeros(src.rows, src.cols, src.type());

	tm.reset();
	tm.start();

	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			dst.at<uchar>(y, x) = saturate_cast<uchar>((1 + alpha)*src.at<uchar>(y, x) - 128 * alpha);
		}
	}

	tm.stop();
	cout << "2. Pixel access by at(): " << tm.getTimeMilli() << " ms." << endl;

	imshow("src", src);
	imshow("dst", dst);
	waitKey();

	// 3. Pixel access by ptr() (No parallel)
	dst = Mat::zeros(src.rows, src.cols, src.type());
	
	tm.reset(); 
	tm.start();

	for (int y = 0; y < src.rows; y++) {
		uchar* pSrc = src.ptr<uchar>(y);
		uchar* pDst = dst.ptr<uchar>(y);

		for (int x = 0; x < src.cols; x++) {
			pDst[x] = saturate_cast<uchar>((1 + alpha)*pSrc[x] - 128 * alpha);
		}
	}

	tm.stop();
	cout << "3. Pixel access by ptr(): " << tm.getTimeMilli() << " ms." << endl;

	imshow("src", src);
	imshow("dst", dst);
	waitKey();

	// 4. cv::parallel_for_ with ParallelLoopBody subclass
	dst = Mat::zeros(src.rows, src.cols, src.type());
	tm.reset();
	tm.start();

	parallel_for_(Range(0, src.rows), ParallelContrast(src, dst, alpha));

	tm.stop();
	cout << "4. With parallel_for_ (ParallelLoopBody):  " << tm.getTimeMilli() << " ms." << endl;

	imshow("src", src);
	imshow("dst", dst);
	waitKey();

	// 5. cv::parallel_for_ with lambda expression
	dst = Mat::zeros(src.rows, src.cols, src.type());
	tm.reset();
	tm.start();

	parallel_for_(Range(0, src.rows), [&](const Range& range) {
		for (int r = range.start; r < range.end; r++) {
			uchar* pSrc = src.ptr<uchar>(r);
			uchar* pDst = dst.ptr<uchar>(r);

			for (int x = 0; x < src.cols; x++) {
				pDst[x] = saturate_cast<uchar>((1 + alpha)*pSrc[x] - 128 * alpha);
			}
		}
	});

	tm.stop();
	cout << "5. With parallel_for_ (lambda expression): " << tm.getTimeMilli() << " ms." << endl;

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}
