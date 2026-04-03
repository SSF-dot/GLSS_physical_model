#pragma once
#include <opencv.hpp>
#include <algorithm>

namespace cg
{
	class Householder
	{
	public:
		Householder() = default;
		void SetData(double alpha, double beta, double e,
			double ka1, double ka2, double ka3,
			double ka4, double ka5, double ka6,
			double kb1, double kb2, double kb3,
			double kb4, double kb5, double kb6);
		cv::Mat CalculateH1(double a, double b);
		cv::Mat CalculateH2(double a, double b);

	private:
		double e_;
		double alpha_, beta_;
		double ka1_, ka2_, ka3_, ka4_, ka5_, ka6_;
		double kb1_, kb2_, kb3_, kb4_, kb5_, kb6_;
	};

}// namespace cg
