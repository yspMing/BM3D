#include <iostream>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <vector>
#include <time.h>

#include "bm3d.h"

using namespace cv;
using namespace std;

int main()
{
	//read picture and add noise
	Mat pic = imread("house.png",0);
	int sigma = 25;
	if (!pic.data)
	{
		cout << "load image error!";
		return -1;
	}
	//convert data type
	Mat Pic(pic.size(), CV_32FC1);
	Mat Noisy(pic.size(), CV_32FC1);
	Mat Basic(pic.size(), CV_32FC1);
	Mat Denoised(pic.size(), CV_32FC1);

	uchar2float(pic, Pic);
	addNoise(sigma, Pic, Noisy);

	//convert type for displaying
	Mat basic(pic.size(), CV_8U);
	Mat noisy(pic.size(), CV_8U);
	Mat denoised(pic.size(), CV_8U);

	float2uchar(Noisy, noisy);
	imshow("origin", pic);
	imshow("noisy", noisy);
	cout << "psnr for noisy image:" << cal_psnr(Pic, Noisy)<<endl;
	waitKey(10);

	//caiculate time used and psnr
	double start, stop, duration;
	start = clock();
	runBm3d(sigma, Noisy, Basic, Denoised);//main denoising method
	stop = clock();
	duration = double(stop - start) / 1000;
	cout << "psnr for basic estimate:" << cal_psnr(Pic, Basic)<<endl;
	cout << "psnr for final denoised:" << cal_psnr(Pic, Denoised) << endl;
	cout << "time for BM3D:" << duration << " s" << endl;

	float2uchar(Basic, basic);
	float2uchar(Denoised, denoised);
	namedWindow("basic", 1);
	imshow("basic", basic);
	imshow("denoised", denoised);
	cvWaitKey(0);

	return 0;
}