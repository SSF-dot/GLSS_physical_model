#include "Optimization.h"

namespace lg {
	void Optimization::Optimize(
		const std::map<std::string, cv::Vec6d>& lines,
		cv::Vec3d& normal, double& angle)
	{
		std::vector<cv::Vec3d> directions;
		for (const auto& line : lines) {
			directions.emplace_back(line.second[0],
				line.second[1], line.second[2]);
		}

		ceres::Problem problem;
		for (int i = 0; i < lines.size() - 1; i++) {
			ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<
				VectorAngleEqual, 1, 3>(
					new VectorAngleEqual(directions[i], directions[i + 1]));
			problem.AddResidualBlock(costFunction, NULL,
				reinterpret_cast<double*>(normal.val));
		}

		ceres::Solver::Options options;
		options.max_num_iterations = 30;
		options.num_threads = 4;
		options.linear_solver_type = ceres::SPARSE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		if (!(summary.termination_type == ceres::CONVERGENCE)) {
			std::cout << "Bundle Adjustment failed.\n";
			return;
		}
		std::cout << " Bundle Adjustment statistics (approximated RMSE):\n"
			<< " residuals:   " << summary.num_residuals << "\n"
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost * 2 / summary.num_residuals) << "\n"
			<< " Final RMSE:   " << std::sqrt(summary.final_cost * 2 / summary.num_residuals) << "\n"
			<< " Time (s):     " << summary.total_time_in_seconds << "\n";

		std::vector<double> thetas;
		for (const auto& dir : directions) {
			double cosTheta = normal.dot(dir);
			cosTheta = std::clamp(cosTheta, -1.0, 1.0);
			thetas.emplace_back(acos(cosTheta));
		}
		angle = std::accumulate(thetas.begin(), thetas.end(), 0.) / thetas.size();
	}

	void Optimization::OptimizeGalvoX(
		const std::map<int, std::vector<cv::Point2d>>& allDeltaAngles,
		double& alpha0, double& k1, double& k2, double& k3,
		double& k4, double& k5, double& k6, double& k7, double& k8)
	{
		cv::Mat G = (cv::Mat_<double>(9, 1) << alpha0, k1, k2, k3, k4, k5, k6, k7, k8);

		std::vector<double> initErrors, optErrors;

		ceres::Problem problem;
		for (auto& deltaAngles : allDeltaAngles) {
			int Y = deltaAngles.first;
			double galvoY = Y / 100.;

			auto& angles = deltaAngles.second;
			for (auto& angle : angles) {
				double galvoX = angle.x;
				double alpha = angle.y;

				ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<
					GalvoDeflectionXOrder3, 1, 9>(
						new GalvoDeflectionXOrder3(galvoX, galvoY, alpha));
				problem.AddResidualBlock(costFunction, NULL,
					reinterpret_cast<double*>(G.data));

				GalvoDeflectionXOrder3 galvoDeflection(galvoX, galvoY, alpha);
				double error[1] = { 0. };
				galvoDeflection(reinterpret_cast<double*>(G.data), error);
				initErrors.emplace_back(std::fabs(error[0]));
			}						
		}

		ceres::Solver::Options options;
		options.max_num_iterations = 30;
		options.num_threads = 4;
		options.linear_solver_type = ceres::SPARSE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		if (!(summary.termination_type == ceres::CONVERGENCE)) {
			std::cout << "Bundle Adjustment failed.\n";
			return;
		}
		std::cout << " Bundle Adjustment statistics (approximated RMSE):\n"
			<< " residuals:   " << summary.num_residuals << "\n"
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost * 2 / summary.num_residuals) << "\n"
			<< " Final RMSE:   " << std::sqrt(summary.final_cost * 2 / summary.num_residuals) << "\n"
			<< " Time (s):     " << summary.total_time_in_seconds << "\n";

		alpha0 = G.at<double>(0, 0);
		k1 = G.at<double>(1, 0);
		k2 = G.at<double>(2, 0);
		k3 = G.at<double>(3, 0);
		k4 = G.at<double>(4, 0);
		k5 = G.at<double>(5, 0);
		k6 = G.at<double>(6, 0);
		k7 = G.at<double>(7, 0);
		k8 = G.at<double>(8, 0);

		for (auto& deltaAngles : allDeltaAngles) {
			int Y = deltaAngles.first;
			double galvoY = Y / 100.;

			auto& angles = deltaAngles.second;
			for (auto& angle : angles) {
				double galvoX = angle.x;
				double alpha = angle.y;
				GalvoDeflectionXOrder3 galvoDeflection(galvoX, galvoY, alpha);
				double error[1] = { 0. };
				galvoDeflection(reinterpret_cast<double*>(G.data), error);
				optErrors.emplace_back(std::fabs(error[0]));
			}
		}

		std::cout << std::endl << std::endl << "-----------------" << std::endl <<
			"GALVO DEFLECTION" << std::endl << "init error: " <<
			std::accumulate(initErrors.begin(), initErrors.end(), 0.) / initErrors.size() << std::endl <<
			"opt error: " <<
			std::accumulate(optErrors.begin(), optErrors.end(), 0.) / optErrors.size() << std::endl <<
			"-----------------" << std::endl;
	}

	void Optimization::OptimizeGalvoY(
		const std::map<int, std::vector<cv::Point2d>>& allDeltaAngles,
		double& beta0, double& k1, double& k2, double& k3,
		double& k4, double& k5, double& k6, double& k7, double& k8)
	{
		cv::Mat G = (cv::Mat_<double>(9, 1) << beta0, k1, k2, k3, k4, k5, k6, k7, k8);

		std::vector<double> initErrors, optErrors;

		ceres::Problem problem;
		for (auto& deltaAngles : allDeltaAngles) {
			int X = deltaAngles.first;
			double galvoX = X / 100.;

			auto& angles = deltaAngles.second;
			for (auto& angle : angles) {
				double galvoY = angle.x;
				double beta = angle.y;

				ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<
					GalvoDeflectionYOrder3, 1, 9>(
						new GalvoDeflectionYOrder3(galvoX, galvoY, beta));
				problem.AddResidualBlock(costFunction, NULL,
					reinterpret_cast<double*>(G.data));

				GalvoDeflectionYOrder3 galvoDeflection(galvoX, galvoY, beta);
				double error[1] = { 0. };
				galvoDeflection(reinterpret_cast<double*>(G.data), error);
				initErrors.emplace_back(std::fabs(error[0]));
			}
		}

		ceres::Solver::Options options;
		options.max_num_iterations = 30;
		options.num_threads = 4;
		options.linear_solver_type = ceres::SPARSE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		if (!(summary.termination_type == ceres::CONVERGENCE)) {
			std::cout << "Bundle Adjustment failed.\n";
			return;
		}
		std::cout << " Bundle Adjustment statistics (approximated RMSE):\n"
			<< " residuals:   " << summary.num_residuals << "\n"
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost * 2 / summary.num_residuals) << "\n"
			<< " Final RMSE:   " << std::sqrt(summary.final_cost * 2 / summary.num_residuals) << "\n"
			<< " Time (s):     " << summary.total_time_in_seconds << "\n";

		beta0 = G.at<double>(0, 0);
		k1 = G.at<double>(1, 0);
		k2 = G.at<double>(2, 0);
		k3 = G.at<double>(3, 0);
		k4 = G.at<double>(4, 0);
		k5 = G.at<double>(5, 0);
		k6 = G.at<double>(6, 0);
		k7 = G.at<double>(7, 0);
		k8 = G.at<double>(8, 0);

		for (auto& deltaAngles : allDeltaAngles) {
			int X = deltaAngles.first;
			double galvoX = X / 100.;

			auto& angles = deltaAngles.second;
			for (auto& angle : angles) {
				double galvoY = angle.x;
				double beta = angle.y;
				GalvoDeflectionYOrder3 galvoDeflection(galvoX, galvoY, beta);
				double error[1] = { 0. };
				galvoDeflection(reinterpret_cast<double*>(G.data), error);
				optErrors.emplace_back(std::fabs(error[0]));
			}
		}

		std::cout << std::endl << std::endl << "-----------------" << std::endl <<
			"GALVO DEFLECTION" << std::endl << "init error: " <<
			std::accumulate(initErrors.begin(), initErrors.end(), 0.) / initErrors.size() << std::endl <<
			"opt error: " <<
			std::accumulate(optErrors.begin(), optErrors.end(), 0.) / optErrors.size() << std::endl <<
			"-----------------" << std::endl;
	}

	void Optimization::OptiGalvoRotPhyModel(
		std::vector<lg::SignalLineMat>& slms,
		cv::Mat& G, cv::Mat& R_g2v,
		cv::Vec3d& laserDirection)
	{
		int n = static_cast<int>(slms.size());

		std::vector<cv::Vec3d> directions;
		for (int i = 0; i < n; i++) {
			cv::Vec6d line = slms[i].line_;
			directions.emplace_back(line[0], line[1], line[2]);
		}

		cv::Mat newG(G.rows - 1, G.cols, G.type());
		for (int i = 0, j = 0; i < G.rows; ++i) {
			if (i != 2) {
				newG.at<double>(j++, 0) = G.at<double>(i, 0);
			}
		}

		cv::Mat rvec_g2v;
		cv::Rodrigues(R_g2v, rvec_g2v);

		ceres::Problem problem;
		for (int i = 0; i < n; i++) {
			double galvoX = slms[i].signal_.x;
			double galvoY = slms[i].signal_.y;
			ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<
				RotationGalvoAdjustmentO3, 4, 14, 3, 3>(new RotationGalvoAdjustmentO3(
					directions[i], galvoX, galvoY));
			problem.AddResidualBlock(costFunction, NULL,
				reinterpret_cast<double*>(newG.data),
				reinterpret_cast<double*>(rvec_g2v.data),
				laserDirection.val);
		}
		//problem.SetParameterBlockConstant(reinterpret_cast<double*>(newG.data));

		ceres::Solver::Options options;
		options.max_num_iterations = 500;
		options.num_threads = 4;
		options.linear_solver_type = ceres::SPARSE_SCHUR;
		options.minimizer_progress_to_stdout = false;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		if (!(summary.termination_type == ceres::CONVERGENCE)) {
			std::cout << "Bundle Adjustment failed.\n";
			return;
		}
		//std::cout << " Bundle Adjustment statistics (approximated RMSE):\n"
		//	<< " residuals:   " << summary.num_residuals << "\n"
		//	<< " Initial RMSE: " << std::sqrt(summary.initial_cost * 2 / summary.num_residuals) << "\n"
		//	<< " Final RMSE:   " << std::sqrt(summary.final_cost * 2 / summary.num_residuals) << "\n"
		//	<< " Time (s):     " << summary.total_time_in_seconds << "\n";

		cv::Rodrigues(rvec_g2v, R_g2v);

		for (int i = 0, j = 0; i < G.rows; ++i) {
			if (i == 2) {
				continue;
			}
			G.at<double>(i, 0) = newG.at<double>(j++, 0);
		}
	}

	void Optimization::OptiGalvoTranPhyModel(
		std::vector<lg::SignalLineMat>& slms,
		cv::Mat& G, cv::Mat& R_g2v, cv::Mat& t_g2v,
		cv::Vec3d& laserDirection, cv::Point3d& laserPoint)
	{
		cv::Mat rvec_g2v;
		cv::Rodrigues(R_g2v, rvec_g2v);
		cv::Mat tvec_g2v = t_g2v.clone();
		cv::Vec3d lp(laserPoint.x, laserPoint.y, laserPoint.z);

		cv::Mat newG(G.rows - 1, G.cols, G.type());
		for (int i = 0, j = 0; i < G.rows; ++i) {
			if (i != 2) {
				newG.at<double>(j++, 0) = G.at<double>(i, 0);
			}
		}
		double E[1] = { G.at<double>(2, 0) };

		ceres::Problem problem;
		int n = static_cast<int>(slms.size());
		for (int i = 0; i < n; i++) {
			cv::Vec6d line = slms[i].line_;
			double galvoX = slms[i].signal_.x;
			double galvoY = slms[i].signal_.y;

			ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<
				TranslationGalvoAdjustmentO3, 3, 14, 1, 3, 3, 3, 3>(
					new TranslationGalvoAdjustmentO3(line, galvoX, galvoY));
			problem.AddResidualBlock(costFunction, NULL,
				reinterpret_cast<double*>(newG.data), E,
				reinterpret_cast<double*>(rvec_g2v.data),
				reinterpret_cast<double*>(tvec_g2v.data),
				laserDirection.val, lp.val);
		}
		problem.SetParameterBlockConstant(reinterpret_cast<double*>(newG.data));
		problem.SetParameterBlockConstant(reinterpret_cast<double*>(rvec_g2v.data));
		problem.SetParameterBlockConstant(laserDirection.val);

		ceres::Solver::Options options;
		options.max_num_iterations = 500;
		options.num_threads = 4;
		options.linear_solver_type = ceres::SPARSE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		if (!(summary.termination_type == ceres::CONVERGENCE)) {
			std::cout << "Bundle Adjustment failed.\n";
			return;
		}
		std::cout << " Bundle Adjustment statistics (approximated RMSE):\n"
			<< " residuals:   " << summary.num_residuals << "\n"
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost * 2 / summary.num_residuals) << "\n"
			<< " Final RMSE:   " << std::sqrt(summary.final_cost * 2 / summary.num_residuals) << "\n"
			<< " Time (s):     " << summary.total_time_in_seconds << "\n";

		cv::Rodrigues(rvec_g2v, R_g2v);
		t_g2v = tvec_g2v.clone();
		laserPoint.x = lp[0];
		laserPoint.y = lp[1];
		laserPoint.z = lp[2];

		for (int i = 0, j = 0; i < G.rows; ++i) {
			if (i == 2) {
				continue;
			}
			G.at<double>(i, 0) = newG.at<double>(j++, 0);
		}
		G.at<double>(2, 0) = E[0];
	}

	void Optimization::OptiGalvoRotTranPhyModel(
		std::vector<lg::SignalLineMat>& slms,
		cv::Mat& G, cv::Mat& R_g2v, cv::Mat& t_g2v,
		cv::Vec3d& laserDirection, cv::Point3d& laserPoint)
	{
		int n = static_cast<int>(slms.size());

		cv::Mat rvec_g2v;
		cv::Rodrigues(R_g2v, rvec_g2v);
		cv::Mat tvec_g2v = t_g2v.clone();
		cv::Vec3d lp(laserPoint.x, laserPoint.y, laserPoint.z);

		cv::Mat newG(G.rows - 1, G.cols, G.type());
		for (int i = 0, j = 0; i < G.rows; ++i) {
			if (i != 2) {
				newG.at<double>(j++, 0) = G.at<double>(i, 0);
			}
		}
		double E[1] = { G.at<double>(2,0) };

		ceres::Problem problem;
		for (int i = 0; i < n; i++) {
			cv::Vec6d line = slms[i].line_;
			double galvoX = slms[i].signal_.x;
			double galvoY = slms[i].signal_.y;
			ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<
				RotTranGalvoAdjustment03, 7, 14, 1, 3, 3, 3, 3>(new RotTranGalvoAdjustment03(
					line, galvoX, galvoY));
			problem.AddResidualBlock(costFunction, NULL,
				reinterpret_cast<double*>(newG.data), E,
				reinterpret_cast<double*>(rvec_g2v.data),
				reinterpret_cast<double*>(tvec_g2v.data),
				laserDirection.val, lp.val);
		}
		//problem.SetParameterBlockConstant(reinterpret_cast<double*>(newG.data));
		//problem.SetParameterBlockConstant(reinterpret_cast<double*>(E));

		ceres::Solver::Options options;
		options.max_num_iterations = 500;
		options.num_threads = 4;
		options.linear_solver_type = ceres::SPARSE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		if (!(summary.termination_type == ceres::CONVERGENCE)) {
			std::cout << "Bundle Adjustment failed.\n";
			return;
		}
		//std::cout << " Bundle Adjustment statistics (approximated RMSE):\n"
		//	<< " residuals:   " << summary.num_residuals << "\n"
		//	<< " Initial RMSE: " << std::sqrt(summary.initial_cost * 2 / summary.num_residuals) << "\n"
		//	<< " Final RMSE:   " << std::sqrt(summary.final_cost * 2 / summary.num_residuals) << "\n"
		//	<< " Time (s):     " << summary.total_time_in_seconds << "\n";

		cv::Rodrigues(rvec_g2v, R_g2v);
		t_g2v = tvec_g2v.clone();
		laserPoint.x = lp[0];
		laserPoint.y = lp[1];
		laserPoint.z = lp[2];
		for (int i = 0, j = 0; i < G.rows; ++i) {
			if (i == 2) {
				continue;
			}
			G.at<double>(i, 0) = newG.at<double>(j++, 0);
		}
		G.at<double>(2, 0) = E[0];
	}

	void Optimization::OptiImprovedIdealModel(
		std::vector<lg::SignalLineMat>& slms,
		cv::Mat& G, cv::Mat& R_g2v, cv::Mat& t_g2v,
		cv::Vec3d& laserDirection, cv::Point3d& laserPoint)
	{
		int n = static_cast<int>(slms.size());

		cv::Mat rvec_g2v;
		cv::Rodrigues(R_g2v, rvec_g2v);
		cv::Mat tvec_g2v = t_g2v.clone();
		cv::Vec3d lp(laserPoint.x, laserPoint.y, laserPoint.z);

		cv::Mat newG(G.rows - 1, G.cols, G.type());
		for (int i = 0, j = 0; i < G.rows; ++i) {
			if (i != 2) {
				newG.at<double>(j++, 0) = G.at<double>(i, 0);
			}
		}
		double E[1] = { G.at<double>(2,0) };

		ceres::Problem problem;
		for (int i = 0; i < n; i++) {
			cv::Vec6d line = slms[i].line_;
			double galvoX = slms[i].signal_.x;
			double galvoY = slms[i].signal_.y;
			ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<
				RotTranGalvoAdjustment01, 7, 14, 1, 3, 3, 3, 3>(new RotTranGalvoAdjustment01(
					line, galvoX, galvoY));
			problem.AddResidualBlock(costFunction, NULL,
				reinterpret_cast<double*>(newG.data), E,
				reinterpret_cast<double*>(rvec_g2v.data),
				reinterpret_cast<double*>(tvec_g2v.data),
				laserDirection.val, lp.val);
		}
		//problem.SetParameterBlockConstant(reinterpret_cast<double*>(newG.data));

		ceres::Solver::Options options;
		options.max_num_iterations = 200;
		options.num_threads = 4;
		options.linear_solver_type = ceres::SPARSE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		if (!(summary.termination_type == ceres::CONVERGENCE)) {
			std::cout << "Bundle Adjustment failed.\n";
			return;
		}
		std::cout << " Bundle Adjustment statistics (approximated RMSE):\n"
			<< " residuals:   " << summary.num_residuals << "\n"
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost * 2 / summary.num_residuals) << "\n"
			<< " Final RMSE:   " << std::sqrt(summary.final_cost * 2 / summary.num_residuals) << "\n"
			<< " Time (s):     " << summary.total_time_in_seconds << "\n";

		cv::Rodrigues(rvec_g2v, R_g2v);
		t_g2v = tvec_g2v.clone();
		laserPoint.x = lp[0];
		laserPoint.y = lp[1];
		laserPoint.z = lp[2];
		for (int i = 0, j = 0; i < G.rows; ++i) {
			if (i == 2) {
				continue;
			}
			G.at<double>(i, 0) = newG.at<double>(j++, 0);
		}
		G.at<double>(2, 0) = E[0];
	}
}
