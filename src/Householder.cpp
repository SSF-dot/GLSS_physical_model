#include "Householder.h"

namespace cg
{
	void Householder::SetData(
		double alpha, double beta, double e,
		double ka1, double ka2, double ka3,
		double ka4, double ka5, double ka6,
		double kb1, double kb2, double kb3,	
		double kb4, double kb5, double kb6)
	{
		e_ = e;
		alpha_ = alpha;
		beta_ = beta;

		ka1_ = ka1;
		ka2_ = ka2;
		ka3_ = ka3;
		ka4_ = ka4;
		ka5_ = ka5;
		ka6_ = ka6;		

		kb1_ = kb1;
		kb2_ = kb2;
		kb3_ = kb3;
		kb4_ = kb4;
		kb5_ = kb5;
		kb6_ = kb6;
	}

	cv::Mat Householder::CalculateH1(double a, double b)
	{
		double alpha =
			alpha_ +
			ka1_ * a +
			ka2_ * a * a +
			ka3_ * a * a * a +
			ka4_ * b +
			ka5_ * b * b +
			ka6_ * b * b * b;
		// 三角函数只接受弧度
		alpha = alpha * CV_PI / 180;
		double cosAlpha = cos(alpha);
		double sinAlpha = sin(alpha);
		cosAlpha = std::clamp(cosAlpha, -1.0, 1.0);
		sinAlpha = std::clamp(sinAlpha, -1.0, 1.0);

		cv::Mat H1 = cv::Mat::eye(4, 4, CV_64FC1);
		H1.at<double>(0, 0) = 1 - 2 * cosAlpha * cosAlpha;
		H1.at<double>(0, 1) = 0 - 2 * cosAlpha * sinAlpha;
		H1.at<double>(0, 3) = 2 * e_* cosAlpha * sinAlpha;

		H1.at<double>(1, 0) = 0 - 2 * cosAlpha * sinAlpha;
		H1.at<double>(1, 1) = 1 - 2 * sinAlpha * sinAlpha;
		H1.at<double>(1, 3) = 2 * e_* sinAlpha * sinAlpha;
		return H1;
	}

	cv::Mat Householder::CalculateH2(double a, double b)
	{
		double beta =
			beta_ +
			kb1_ * b +
			kb2_ * b * b +
			kb3_ * b * b * b +
			kb4_ * a +
			kb5_ * a * a +
			kb6_ * a * a * a;
		beta = beta * CV_PI / 180;
		double cosBeta = cos(beta);
		double sinBeta = sin(beta);
		cosBeta = std::clamp(cosBeta, -1.0, 1.0);
		sinBeta = std::clamp(sinBeta, -1.0, 1.0);

		cv::Mat H2 = cv::Mat::eye(4, 4, CV_64FC1);
		H2.at<double>(1, 1) = 1 - 2 * sinBeta * sinBeta;
		H2.at<double>(1, 2) = 0 - 2 * sinBeta * cosBeta;

		H2.at<double>(2, 1) = 0 - 2 * sinBeta * cosBeta;
		H2.at<double>(2, 2) = 1 - 2 * cosBeta * cosBeta;
		return H2;
	}

}// namespace cg