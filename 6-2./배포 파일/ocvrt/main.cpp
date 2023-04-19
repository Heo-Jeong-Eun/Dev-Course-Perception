#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	// TODO: 명령행 인자 개수가 3개보다 작으면 사용법을 출력하고 종료하세요.

	if (argc < 3) {
		cout << "Usage: ocvrt.exe <src_image> <dst_image>" << endl;
		return 0;
	}

	// TODO: 첫 번째 이미지 파일을 imread() 함수로 읽어서 img 변수에 저장하세요.
	Mat img = imread(argv[1], IMREAD_UNCHANGED);

	if (img.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	// TODO: 두 번째 이미지 파일 이름으로 img 영상을 저장하세요.
	// 저장이 제대로 되면 ret 변수에 true를, 실패하면 false를 저장하세요.

	bool ret = imwrite(argv[2], img);
	
	if (ret) {
		cout << argv[1] << " is successfully saved as " << argv[2] << endl;
	} else {
		cout << "File save failed!" << endl;
	}

	return 0;
}
