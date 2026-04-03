#pragma once

#include "Householder.h"
#include "Optimization.h"

#include <algorithm>
#include <numeric>
#include <fstream>
#include <random>

namespace glss
{
	double FitLine3d(const std::vector<cv::Point3d>& points, cv::Vec6d& line);

	void ComputeRv2g(const std::vector<lg::SignalLineMat>& slms,
		cv::Mat& R_v2g, cv::Vec3d& direction);

	void ComputeTranslationV2g(const std::vector<lg::SignalLineMat>& slms,
		const cv::Mat& R_v2g, const cv::Vec3d& direction,
		cv::Mat& translation_v2g, cv::Point3d& linePoint);

	// ideal model
	void ComputeRv2g(const std::vector<lg::SignalLineMat>& slms,
		const cv::Vec3d& direction, cv::Mat& R_v2g);

	// ideal model
	void ComputeTransV2g(const std::vector<lg::SignalLineMat>& slms,
		const cv::Mat& R_v2g, const cv::Vec3d& laserDirection,
		const cv::Point3d& laserPoint, cv::Mat& translation_v2g);

	void Update(const cv::Mat& G,
		std::vector<lg::SignalLineMat>& trainSlms,
		std::vector<lg::SignalLineMat>& testSlms);

	void ComputeAngleError(const std::vector<lg::SignalLineMat>& slms,
		const cv::Vec3d& optimizedDir, const cv::Mat& optimizedR_v2g);

	void ComputeP2L(const std::vector<lg::SignalLineMat>& slms,
		const cv::Mat& R_g2v, const cv::Vec3d& direction,
		cv::Mat& t_g2v, cv::Point3d& linePoint);

	void ComputeLinesApproximation(const std::vector<lg::SignalLineMat>& slms,
		const cv::Mat& R_g2v, const cv::Vec3d& direction,
		cv::Mat& t_g2v, cv::Point3d& linePoint);

	void ComputeLinesApproximation2(const std::vector<lg::SignalLineMat>& slms,
		const cv::Mat& R_g2v, const cv::Vec3d& direction,
		cv::Mat& t_g2v, cv::Point3d& linePoint);

	void ComputeLinesApproximation(
		const std::vector<cv::Vec6d>& line1,
		const std::vector<cv::Vec6d>& line2);

	void ComputeLinesApproximation2(
		const std::vector<cv::Vec6d>& line1,
		const std::vector<cv::Vec6d>& line2);

	void ComputeLinesApproForCamera(
		const std::vector<cv::Point2d>& signals,
		const std::vector<cv::Vec6d>& line1,
		const std::vector<cv::Vec6d>& line2);

	void ComputeLinesApproForCamera2(
		const std::vector<cv::Point2d>& signals,
		const std::vector<cv::Vec6d>& line1,
		const std::vector<cv::Vec6d>& line2);

	// 对单位向量添加噪声
	cv::Vec3d AddNoiseToUnitVector(const cv::Vec3d& v, double sigma);
	cv::Vec3d PerturbUnitVector(const cv::Vec3d& v, double angle);

	// 对旋转矩阵添加噪声
	cv::Mat AddNoiseToRotationMatrix(const cv::Mat& R, double sigma);
	cv::Mat PerturbRotation(const cv::Mat& R, double angle);

	void Write(const cv::Mat& G,
		const cv::Mat& R_g2v,
		const cv::Vec3d& direction,
		const cv::Mat& translation_g2v,
		const cv::Point3d& linePoint);

	void SolvePhyModel();

	void SolveNoisePhyModel();

	void SolveIdealModel();

	void SolveCameraModel();
}
