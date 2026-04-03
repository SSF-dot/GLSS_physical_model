#include "DeflectionAngle.h"


namespace lg
{
	double FitLine3d(const std::vector<cv::Point3d>& points, cv::Vec6d& line)
	{
		cv::fitLine(points, line, cv::DIST_L2, 0., 0.01, 0.01);
		if (line[2] > 0) {
			line[0] *= (-1);
			line[1] *= (-1);
			line[2] *= (-1);
		}

		double err = 0.;
		cv::Vec3d lineDir(line[0], line[1], line[2]);
		cv::Vec3d linePoint(line[3], line[4], line[5]);
		for (const auto& point : points) {
			cv::Vec3d v = cv::Vec3d(point.x, point.y, point.z) - linePoint;
			cv::Vec3d projection = (v.dot(lineDir) / (cv::norm(lineDir) * cv::norm(lineDir))) * lineDir;
			cv::Vec3d h = v - projection;
			err += cv::norm(h);
		}
		return err / points.size();
	}

	double FitLine2d(const std::vector<cv::Point2d>& points, cv::Vec4d& line)
	{
		cv::fitLine(points, line, cv::DIST_L2, 0., 0.01, 0.01);

		double err = 0.;
		cv::Vec2d lineDir(line[0], line[1]);
		cv::Vec2d linePoint(line[2], line[3]);
		for (const auto& point : points) {
			cv::Vec2d v = cv::Vec2d(point.x, point.y) - linePoint;
			cv::Vec2d projection = (v.dot(lineDir) / (cv::norm(lineDir) * cv::norm(lineDir))) * lineDir;
			cv::Vec2d h = v - projection;
			err += cv::norm(h);
		}
		return err / points.size();
	}

	double FitPlane(const std::vector<cv::Point3f>& objPoints,
		double& a, double& b, double& c, double& d)
	{
		cv::Mat data(objPoints.size(), 3, CV_64F);
		for (size_t i = 0; i < objPoints.size(); i++) {
			data.at<double>(i, 0) = objPoints[i].x;
			data.at<double>(i, 1) = objPoints[i].y;
			data.at<double>(i, 2) = objPoints[i].z;
		}

		// 计算点的均值
		cv::Mat colMeans;
		cv::reduce(data, colMeans, 0, cv::REDUCE_AVG);
		double meanX = colMeans.at<double>(0);
		double meanY = colMeans.at<double>(1);
		double meanZ = colMeans.at<double>(2);

		// 将点集中心化
		cv::Mat centered = data - cv::Mat::ones(data.rows, 1, CV_64F) *
			(cv::Mat_<double>(1, 3) << meanX, meanY, meanZ);

		// 对中心化后的矩阵进行 SVD 分解
		cv::Mat A, W, U, Vt;
		cv::SVD::compute(centered, W, U, Vt, cv::SVD::FULL_UV);

		// 提取法向量（Vt 的最后一行）
		cv::Mat normal = Vt.row(2);

		// 计算平面方程的系数
		a = normal.at<double>(0);
		b = normal.at<double>(1);
		c = normal.at<double>(2);
		d = -(a * meanX + b * meanY + c * meanZ);

		// 归一化法向量
		double norm = std::sqrt(a * a + b * b + c * c);
		if (norm < 1e-12) {
			return -1.; // 无效平面
		}
		a /= norm;
		b /= norm;
		c /= norm;
		d /= norm;

		double maxDist = 0.0;
		for (const auto& p : objPoints) {
			double dist = std::abs(a * p.x + b * p.y + c * p.z + d);
			if (dist > maxDist) {
				maxDist = dist;
			}
		}
		return maxDist;
	}

	void ComputeConeNormalAngle(
		const std::map<std::string, cv::Vec6d>& lines,
		cv::Vec3d& normal, double& angle)
	{
		std::vector<cv::Point3f> objPoints;
		// 单位向量
		for (const auto& line : lines) {
			objPoints.emplace_back(line.second[0],
				line.second[1], line.second[2]);
		}
		double a, b, c, d;
		double maxError = FitPlane(objPoints, a, b, c, d);
		//std::cout << "Fit plane max error: " << maxError << "\n";

		if (c > 0.) {
			normal = cv::Vec3d(a, b, c);
		}
		else {
			normal = cv::Vec3d(-a, -b, -c);
		}
		double norm = cv::norm(normal);
		if (fabs(norm) < 1e-7) {
			throw std::invalid_argument("Zero vector cannot be normalized.");
		}
		normal = normal / norm;

		std::vector<double> thetas;
		for (const auto& line : lines) {
			cv::Vec3d generatrix{ line.second[0],
				line.second[1], line.second[2] };
			double cosTheta = normal.dot(generatrix);

			cosTheta = std::clamp(cosTheta, -1.0, 1.0);
			thetas.emplace_back(acos(cosTheta));
		}
		angle = std::accumulate(thetas.begin(), thetas.end(), 0.) / thetas.size();
	}

	void ComputeConeNormalVector(
		const std::map<std::string, cv::Vec6d>& lines, cv::Vec3d& normal)
	{
		// 转换为 cv::Mat
		cv::Mat data(lines.size(), 3, CV_64F);
		int i = 0;
		for (const auto& line : lines) {
			data.at<double>(i, 0) = line.second[0];
			data.at<double>(i, 1) = line.second[1];
			data.at<double>(i, 2) = line.second[2];
			i++;
		}

		// 计算均值
		cv::Mat mean;
		cv::reduce(data, mean, 0, cv::REDUCE_AVG);

		// 中心化数据
		cv::Mat centered = data - cv::repeat(mean, data.rows, 1);

		// 计算协方差矩阵
		cv::Mat cov = centered.t() * centered / (data.rows - 1);

		// 特征值分解
		cv::Mat eigenvalues, eigenvectors;
		cv::eigen(cov, eigenvalues, eigenvectors);

		// 最大特征值对应的特征向量是法向量
		normal = eigenvectors.row(2); // 取最后一行
		normal *= (-1);
	}

	void ComputeDeltaAngle(const std::map<std::string, cv::Vec6d>& lines,
		const cv::Vec3d& normal, double coneHalfAngle,
		const cv::Vec3d& originGeneratrix,
		std::map<std::string, double>& deltaAngles)
	{
		for (const auto& line : lines) {
			cv::Vec3d generatrix{ line.second[0],
				line.second[1], line.second[2] };
			double cosGamma = originGeneratrix.dot(generatrix);
			cosGamma = std::clamp(cosGamma, -1.0, 1.0);

			double v = 1 - ((1 - cosGamma) / (sin(coneHalfAngle) * sin(coneHalfAngle)));
			double alpha = acos(v) * 0.5;

			deltaAngles[line.first] = alpha;
		}
	}

	void Signal2Angle()
	{
		std::map<int, std::map<std::string, cv::Point2i>> horizontalScan; // Y-mirror constant, X-mirror vary
		std::map<int, std::map<std::string, cv::Point2i>> verticalScan; // X-mirror constant, Y-mirror vary
		{
			std::ifstream file;
			file.open("../phy_model/table-221A.txt");
			std::string str;
			int x, y;
			while (file >> str >> x >> y) {
				if (str.empty()) {
					continue;
				}
				char last = str.back();
				if (last == 'A' || last == 'B' ||
					last == 'C' || last == 'D' || last == 'E') {
					horizontalScan[y].insert({ str, cv::Point2i(x, y) });
					verticalScan[x].insert({ str, cv::Point2i(x, y) });
				}
			}
			file.close();
		}

		// 拟合空间直线
		std::map<std::string, std::vector<cv::Point3d>> objPoints;
		{
			std::ifstream file;
			file.open("../phy_model/reconstructed_points.txt");
			std::string str;
			double x, y, z;
			while (file >> str >> x >> y >> z) {
				if (str.empty()) {
					continue;
				}
				objPoints[str].emplace_back(x, y, z);
			}
			file.close();
		}
		std::map<std::string, cv::Vec6d> allLines;
		std::map<std::string, double> allLineErrors;
		for (const auto& ele : objPoints) {
			if (ele.second.size() < 6) {
				continue;
			}
			cv::Vec6d line;
			double error = FitLine3d(ele.second, line);
			if (error > 0.2) {
				continue;
			}
			std::string name = ele.first;
			allLines.insert({ name, line });
			allLineErrors.insert({ name, error });
		}

		double alpha0, beta0, e; // 45 52.5 67.5
		double ka1, ka2, ka3, ka4, ka5, ka6, ka7, ka8;
		double kb1, kb2, kb3, kb4, kb5, kb6, kb7, kb8;
		{
			cv::FileStorage fs("../phy_model/init_galvo_parameters.yaml", cv::FileStorage::READ);
			// 检查文件是否成功打开
			if (!fs.isOpened()) {
				std::cerr << "无法打开文件!" << std::endl;
			}
			// 读取数据
			fs["galvo"]["alpha"] >> alpha0;
			fs["galvo"]["beta"] >> beta0;
			fs["galvo"]["e"] >> e;
			fs["galvo"]["ka1"] >> ka1;
			fs["galvo"]["ka2"] >> ka2;
			fs["galvo"]["ka3"] >> ka3;
			fs["galvo"]["ka4"] >> ka4;
			fs["galvo"]["ka5"] >> ka5;
			fs["galvo"]["ka6"] >> ka6;
			fs["galvo"]["ka7"] >> ka7;
			fs["galvo"]["ka8"] >> ka8;

			fs["galvo"]["kb1"] >> kb1;
			fs["galvo"]["kb2"] >> kb2;
			fs["galvo"]["kb3"] >> kb3;
			fs["galvo"]["kb4"] >> kb4;
			fs["galvo"]["kb5"] >> kb5;
			fs["galvo"]["kb6"] >> kb6;
			fs["galvo"]["kb7"] >> kb7;
			fs["galvo"]["kb8"] >> kb8;

			fs.release();
		}

		// Y-mirror constant, X-mirror vary
		{
			// Y镜面处于不同位置 都对应着一个系数
			// 求出所有系数 取平均值
			std::map<int, double> ks; // Y -- k  Y:0000
			std::map<int, std::vector<cv::Point2d>> allDeltaAngles; // Y -- (x, deltaAlpha) Y:0000 x:00.00
			for (const auto& hScan : horizontalScan) {
				int Y = hScan.first;

				auto signals = hScan.second;
				std::string origin = "";
				std::map<std::string, cv::Vec6d> lines;
				for (const auto& signal : signals) {
					std::string name = signal.first;
					auto iter = allLines.find(name);
					if (iter == allLines.end()) {
						continue;
					}
					lines.insert(*iter);

					if (signal.second.x == 0) {
						origin = name;
					}
				}
				if (lines.size() < 10) {					
					continue;
				}
				if (lines.find(origin) == lines.end()) {
					continue;
				}

				std::cout << "Y= " << Y << " succeed" << std::endl;

				// 计算圆锥曲面的法矢和半顶角
				cv::Vec3d coneNormal;
				double coneHalfAngle;
				lg::ComputeConeNormalAngle(lines, coneNormal, coneHalfAngle);

				cv::Vec3d optConeNormal = coneNormal;
				double optConeHalfAngle = coneHalfAngle;
				//lg::Optimization::Optimize(lines, optConeNormal, optConeHalfAngle);

				// 计算镜面偏转角度
				std::map<std::string, double> deltaAngles; // rad
				cv::Vec3d originGeneratrix{ lines[origin][0],
					lines[origin][1], lines[origin][2] };
				// deltaAngle 全是正值
				lg::ComputeDeltaAngle(lines, optConeNormal, optConeHalfAngle, originGeneratrix, deltaAngles);

				// x <--> deltaAngle
				std::vector<cv::Point2d> signalAngle;
				for (auto& angle : deltaAngles) {
					std::string key = angle.first;
					double x = signals[key].x / 100.;
					// rad --> degree
					double degree = angle.second * 180 / CV_PI;
					signalAngle.emplace_back(x, degree);
				}

				for (auto& sa : signalAngle) {
					// when x > 0, angle < 0
					if (sa.x > 0.) {
						sa.y = alpha0 - sa.y;
					}
					else {
						sa.y = alpha0 + sa.y;
					}
				}
				cv::Vec4d line;
				FitLine2d(signalAngle, line);
				double k = line[1] / line[0];
				ks.insert({ Y, k });
				allDeltaAngles.insert({ Y, signalAngle });
			}
			double averageK = 0.;
			for (const auto k : ks) {
				averageK += k.second;
			}
			averageK /= ks.size();

			// 总体优化
			// alpha = alpha0 + m1 * a + m2 * a^2 + m3 * a^3 + m4 * a^4 + n1 * b + n2 * b^2 + n3 * b^3 + n4 * b^4
			ka1 = averageK;
			lg::Optimization::OptimizeGalvoX(allDeltaAngles, alpha0, ka1, ka2, ka3, ka4, ka5, ka6, ka7, ka8);
		}

		// X-mirror constant, Y-mirror vary
		{
			std::map<int, double> ks; // X -- k  X:0000
			std::map<int, std::vector<cv::Point2d>> allDeltaAngles; // X -- (y, deltaBeta) X:0000 y:00.00
			for (const auto& vScan : verticalScan) {
				int X = vScan.first;

				auto signals = vScan.second;
				std::string origin = "";
				std::map<std::string, cv::Vec6d> lines;
				for (const auto& signal : signals) {
					std::string name = signal.first;
					auto iter = allLines.find(name);
					if (iter == allLines.end()) {
						continue;
					}
					lines.insert(*iter);

					if (signal.second.y == 0) {
						origin = name;
					}
				}
				if (lines.size() < 10) {					
					continue;
				}
				if (lines.find(origin) == lines.end()) {					
					continue;
				}
				std::cout << "X= " << X <<" succeed" << std::endl;

				// 计算圆锥曲面的法矢和半顶角
				cv::Vec3d coneNormal;
				double coneHalfAngle;
				lg::ComputeConeNormalAngle(lines, coneNormal, coneHalfAngle);

				cv::Vec3d optConeNormal = coneNormal;
				double optConeHalfAngle = coneHalfAngle;
				//lg::Optimization::Optimize(lines, optConeNormal, optConeHalfAngle);

				// 计算镜面偏转角度
				std::map<std::string, double> deltaAngles; // rad
				cv::Vec3d originGeneratrix{ lines[origin][0],
					lines[origin][1], lines[origin][2] };
				// deltaAngle 全是正值
				lg::ComputeDeltaAngle(lines, optConeNormal, optConeHalfAngle, originGeneratrix, deltaAngles);

				// y <--> deltaAngle
				std::vector<cv::Point2d> signalAngle;
				for (auto& angle : deltaAngles) {
					std::string key = angle.first;
					double y = signals[key].y / 100.;
					// rad --> degree
					double degree = angle.second * 180 / CV_PI;
					signalAngle.emplace_back(y, degree);
				}

				for (auto& sa : signalAngle) {
					// when x > 0, angle > 0
					if (sa.x > 0.) {
						sa.y = beta0 + sa.y;
					}
					else {
						sa.y = beta0 - sa.y;
					}
				}
				cv::Vec4d line;
				FitLine2d(signalAngle, line);
				double k = line[1] / line[0];
				ks.insert({ X, k });
				allDeltaAngles.insert({ X, signalAngle });
			}
			double averageK = 0.;
			for (const auto k : ks) {
				averageK += k.second;
			}
			averageK /= ks.size();

			kb1 = averageK;
			lg::Optimization::OptimizeGalvoY(allDeltaAngles, beta0, kb1, kb2, kb3, kb4, kb5, kb6, kb7, kb8);
		}

		{
			cv::FileStorage fs("../phy_model/galvo_params_order3.yaml", cv::FileStorage::WRITE);
			// 检查文件是否成功打开
			if (!fs.isOpened()) {
				std::cerr << "无法创建文件!" << std::endl;
			}

			// 写入数据到 YAML 文件
			fs << "galvo" << "{";
			fs << "alpha" << alpha0;
			fs << "beta" << beta0;
			fs << "e" << e;

			fs << "ka1" << ka1;
			fs << "ka2" << ka2;
			fs << "ka3" << ka3;
			fs << "ka4" << ka4;
			fs << "ka5" << ka5;
			fs << "ka6" << ka6;
			fs << "ka7" << ka7;
			fs << "ka8" << ka8;

			fs << "kb1" << kb1;
			fs << "kb2" << kb2;
			fs << "kb3" << kb3;
			fs << "kb4" << kb4;
			fs << "kb5" << kb5;
			fs << "kb6" << kb6;
			fs << "kb7" << kb7;
			fs << "kb8" << kb8;
			fs << "}";

			// 关闭文件
			fs.release();
		}
	}

}