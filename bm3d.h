#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <vector>

using namespace cv;
using namespace std;

void addNoise(const int sigma,const Mat origin,Mat &noisy);
void uchar2float(const Mat tyuchar,Mat &tyfloat);
void float2uchar(const Mat tyfloat, Mat &tyuchar);
float cal_psnr(const Mat x, const Mat y);
int log2(const int N);

int runBm3d(const int sigma,const Mat image_noisy,
	Mat &image_basic,Mat &image_denoised);

void getPatches(const Mat img,const int width,const int height,const int channels,
	const int patchSize,const int step,vector<Mat>&block,vector<int>&row_idx,vector<int>&col_idx);

void tran2d( vector<Mat> &input,char* tran_mode,int patchsize);

void getSimilarPatch(const vector<Mat> block, vector<Mat>&sim_patch,vector<int>&sim_num,
	int i, int j, int bn_r, int bn_c, int area, int maxNum, int tao);

float cal_distance(const Mat a, const Mat b);

void tran1d(vector<Mat>&input,char* tran_mode,int patchSize);

void shrink(vector<Mat>&input, float threshhold);

float calculate_weight_hd(const vector<Mat>input,int sigma);
float calculate_weight_wien(const vector<Mat>input, int sigma);

void inv_tran_3d(vector<Mat>&input,char* mode2d, char *mode1d,int patchSize);

void aggregation(Mat &numerator, Mat &denominator, vector<int>idx_r, vector<int>idx_c, const vector<Mat> input,
	float weight,int patchSize,Mat window);

void gen_wienFilter(vector<Mat>&input, int sigma);

void wienFiltering(vector<Mat>&input, const vector<Mat>wien,int patchSize);

Mat gen_kaiser(int beta,int length);
void wavedec(float *input,int length);
void waverec(float* input, int length,int N);
