#pragma once

#include <opencv.hpp>

namespace lg {
	struct SignalLineMat
	{
		cv::Point2d signal_;
		cv::Vec6d line_;
		cv::Mat H_;
		cv::Mat invH_;
		cv::Mat antiH_;
	};
}