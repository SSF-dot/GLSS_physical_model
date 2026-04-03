#pragma once

#include "Optimization.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>

namespace lg
{
	double FitPlane(const std::vector<cv::Point3f>& objPoints,
		double& a, double& b, double& c, double& d);

	void ComputeConeNormalAngle(
		const std::map<std::string, cv::Vec6d>& lines,
		cv::Vec3d& normal, double& angle);

	void ComputeConeNormalVector(
		const std::map<std::string, cv::Vec6d>& lines, cv::Vec3d& normal);

	void ComputeDeltaAngle(const std::map<std::string, cv::Vec6d>& lines,
		const cv::Vec3d& normal, double coneHalfAngle,
		const cv::Vec3d& originGeneratrix,
		std::map<std::string, double>& deltaAngles);

	double FitLine3d(const std::vector<cv::Point3d>& points, cv::Vec6d& line);

	double FitLine2d(const std::vector<cv::Point2d>& points, cv::Vec4d& line);

	void Signal2Angle();
}