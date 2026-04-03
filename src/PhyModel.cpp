#include "PhyModel.h"

namespace glss
{
	double FitLine3d(const std::vector<cv::Point3d>& points, cv::Vec6d& line)
	{
		cv::fitLine(points, line, cv::DIST_L2, 0., 0.01, 0.01);
		if (line[2] < 0) {
			line[0] *= (-1);
			line[1] *= (-1);
			line[2] *= (-1);
		}

		double z = 0.;
		double t = (z - line[5]) / line[2];
		line[3] += (t * line[0]);
		line[4] += (t * line[1]);
		line[5] = z;

		cv::Vec3d lineDir(line[0], line[1], line[2]);
		cv::Vec3d linePoint(line[3], line[4], line[5]);
		double maxError = DBL_MIN;
		for (const auto& point : points) {
			cv::Vec3d v = cv::Vec3d(point.x, point.y, point.z) - linePoint;
			cv::Vec3d projection = (v.dot(lineDir) / (cv::norm(lineDir) * cv::norm(lineDir))) * lineDir;
			cv::Vec3d h = v - projection;
			double e = cv::norm(h);
			if (e > maxError) {
				maxError = e;
			}
		}
		return maxError;
	}


	void ComputeRv2g(const std::vector<lg::SignalLineMat>& slms,
		cv::Mat& R_v2g, cv::Vec3d& direction)
	{
		int n = static_cast<int>(slms.size());
		cv::Mat A(3 * (n - 1), 9, CV_64FC1);
		for (int i = 0; i < n - 1; i++) {
			cv::Vec6d line1 = slms[i].line_;
			cv::Mat H1 = slms[i].H_;

			cv::Vec3d n1 = cv::Vec3d(line1[0], line1[1], line1[2]);
			cv::Mat R1 = H1(cv::Rect(0, 0, 3, 3)).clone();

			cv::Vec6d line2 = slms[i + 1].line_;
			cv::Mat H2 = slms[i + 1].H_;

			cv::Vec3d n2 = cv::Vec3d(line2[0], line2[1], line2[2]);
			cv::Mat R2 = H2(cv::Rect(0, 0, 3, 3)).clone();

			cv::Mat M0 = n1[0] * R1 - n2[0] * R2;
			cv::Mat M1 = n1[1] * R1 - n2[1] * R2;
			cv::Mat M2 = n1[2] * R1 - n2[2] * R2;
			M0.copyTo(A(cv::Rect(0, 3 * i, 3, 3)));
			M1.copyTo(A(cv::Rect(3, 3 * i, 3, 3)));
			M2.copyTo(A(cv::Rect(6, 3 * i, 3, 3)));
		}

		cv::Mat w, u, vt;
		cv::SVD::compute(A, w, u, vt); // vt.t()的最后一列
		cv::Mat vec = vt.row(vt.rows - 1).t();

		cv::Mat matrix = (cv::Mat_<double>(3, 3) <<
			vec.at<double>(0, 0), vec.at<double>(1, 0), vec.at<double>(2, 0),
			vec.at<double>(3, 0), vec.at<double>(4, 0), vec.at<double>(5, 0),
			vec.at<double>(6, 0), vec.at<double>(7, 0), vec.at<double>(8, 0));
		// SVD分解 找到最接近原始矩阵的旋转矩阵
		cv::Mat U, S, VT;
		cv::SVD::compute(matrix, S, U, VT);
		R_v2g = VT.t() * U.t();

		// 确保 R_gv 的行列式为正
		if (cv::determinant(R_v2g) < 0) {
			VT.col(2) *= -1; // 反转最后一列
			R_v2g = VT.t() * U.t();
		}

		std::vector<cv::Vec3d> nxyz;
		for (int i = 0; i < n; i++) {
			cv::Mat dir = (cv::Mat_<double>(3, 1) <<
				slms[i].line_[0], slms[i].line_[1], slms[i].line_[2]);
			cv::Mat R = slms[i].H_(cv::Rect(0, 0, 3, 3));
			cv::Mat ret = R * R_v2g * dir;
			nxyz.emplace_back(ret.at<double>(0, 0),
				ret.at<double>(1, 0), ret.at<double>(2, 0));
		}
		cv::Vec3d sum = std::accumulate(nxyz.begin(), nxyz.end(), cv::Vec3d(0., 0., 0.));
		sum = sum / cv::norm(sum);

		direction = sum;
	}

	// ideal model
	void ComputeRv2g(const std::vector<lg::SignalLineMat>& slms,
		const cv::Vec3d& direction, cv::Mat& R_v2g)
	{
		int n = static_cast<int>(slms.size());
		cv::Mat A(3 * n, 9, CV_64FC1);
		cv::Mat B(3 * n, 1, CV_64FC1);
		for (int i = 0; i < n; i++) {
			cv::Vec6d line1 = slms[i].line_;
			cv::Mat H1 = slms[i].H_;

			cv::Vec3d n1 = cv::Vec3d(line1[0], line1[1], line1[2]);
			cv::Mat R1 = H1(cv::Rect(0, 0, 3, 3)).clone();

			cv::Mat M0 = n1[0] * R1;
			cv::Mat M1 = n1[1] * R1;
			cv::Mat M2 = n1[2] * R1;
			M0.copyTo(A(cv::Rect(0, 3 * i, 3, 3)));
			M1.copyTo(A(cv::Rect(3, 3 * i, 3, 3)));
			M2.copyTo(A(cv::Rect(6, 3 * i, 3, 3)));
			
			B.at<double>(3 * i, 0) = direction[0];
			B.at<double>(3 * i + 1, 0) = direction[1];
			B.at<double>(3 * i + 2, 0) = direction[2];
		}
		cv::Mat vec;
		cv::solve(A, B, vec, cv::DECOMP_SVD);

		cv::Mat matrix = (cv::Mat_<double>(3, 3) <<
			vec.at<double>(0, 0), vec.at<double>(1, 0), vec.at<double>(2, 0),
			vec.at<double>(3, 0), vec.at<double>(4, 0), vec.at<double>(5, 0),
			vec.at<double>(6, 0), vec.at<double>(7, 0), vec.at<double>(8, 0));
		// SVD分解 找到最接近原始矩阵的旋转矩阵
		cv::Mat U, S, VT;
		cv::SVD::compute(matrix, S, U, VT);
		R_v2g = VT.t() * U.t();

		// 确保 R_gv 的行列式为正
		if (cv::determinant(R_v2g) < 0) {
			VT.col(2) *= -1; // 反转最后一列
			R_v2g = VT.t() * U.t();
		}
	}

	void ComputeTranslationV2g(const std::vector<lg::SignalLineMat>& slms,
		const cv::Mat& R_v2g, const cv::Vec3d& direction,
		cv::Mat& translation_v2g, cv::Point3d& linePoint)
	{
		int n = static_cast<int>(slms.size());

		std::vector<cv::Mat> Ri_Rv2g, Ri, tveci;
		for (int i = 0; i < n; i++) {
			cv::Mat R = slms[i].H_(cv::Rect(0, 0, 3, 3)).clone();
			Ri_Rv2g.push_back(R * R_v2g);
			Ri.push_back(R);
			tveci.push_back(slms[i].H_(cv::Rect(3, 0, 1, 3)));
		}

		double n1 = direction[0];
		double n2 = direction[1];
		double n3 = direction[2];

		cv::Mat A(2 * (n - 1), 3, CV_64FC1);
		cv::Mat B(2 * (n - 1), 1, CV_64FC1);

		for (int i = 0; i < n - 1; i++) {
			cv::Mat deltaR = Ri[i] - Ri[i + 1];
			double a11 = deltaR.at<double>(0, 0);
			double a12 = deltaR.at<double>(0, 1);
			double a13 = deltaR.at<double>(0, 2);

			double a21 = deltaR.at<double>(1, 0);
			double a22 = deltaR.at<double>(1, 1);
			double a23 = deltaR.at<double>(1, 2);

			double a31 = deltaR.at<double>(2, 0);
			double a32 = deltaR.at<double>(2, 1);
			double a33 = deltaR.at<double>(2, 2);

			cv::Mat p1 = (cv::Mat_<double>(3, 1) <<
				slms[i].line_[3], slms[i].line_[4], slms[i].line_[5]);
			cv::Mat p2 = (cv::Mat_<double>(3, 1) <<
				slms[i + 1].line_[3], slms[i + 1].line_[4], slms[i + 1].line_[5]);

			cv::Mat b = (Ri_Rv2g[i] * p1 + tveci[i]) -
				(Ri_Rv2g[i + 1] * p2 + tveci[i + 1]);
			double b1 = b.at<double>(0, 0);
			double b2 = b.at<double>(1, 0);
			double b3 = b.at<double>(2, 0);

			double m11 = a11 * n2 - a21 * n1;
			double m12 = a12 * n2 - a22 * n1;
			double m13 = a13 * n2 - a23 * n1;

			double m21 = a11 * n3 - a31 * n1;
			double m22 = a12 * n3 - a32 * n1;
			double m23 = a13 * n3 - a33 * n1;

			//double m31 = a21 * n3 - a31 * n2;
			//double m32 = a22 * n3 - a32 * n2;
			//double m33 = a23 * n3 - a33 * n2;

			double b21 = b2 * n1 - b1 * n2;
			double b31 = b3 * n1 - b1 * n3;
			//double b32 = b3 * n2 - b2 * n3;

			cv::Mat M = (cv::Mat_<double>(2, 3) <<
				m11, m12, m13,
				m21, m22, m23);

			M.copyTo(A(cv::Rect(0, 2 * i, 3, 2)));

			B.at<double>(2 * i, 0) = b21;
			B.at<double>(2 * i + 1, 0) = b31;
			//B.at<double>(3 * i + 2, 0) = b32;
		}

		cv::Mat X;
		cv::solve(A, B, X, cv::DECOMP_SVD);

		translation_v2g = (cv::Mat_<double>(3, 1) <<
			X.at<double>(0, 0), X.at<double>(1, 0), X.at<double>(2, 0));

		double cx = 0.;
		double cy = 0.;
		double cz = 0.;
		for (int i = 0; i < n; i++) {
			cv::Mat p = (cv::Mat_<double>(3, 1) <<
				slms[i].line_[3], slms[i].line_[4], slms[i].line_[5]);

			cv::Mat Q = Ri_Rv2g[i] * p + Ri[i] * translation_v2g + tveci[i];
			cx += Q.at<double>(0, 0);
			cy += Q.at<double>(1, 0);
			cz += Q.at<double>(2, 0);
		}
		cx /= n;
		cy /= n;
		cz /= n;
		linePoint = { cx,cy,cz };
	}

	// ideal model
	void ComputeTransV2g(const std::vector<lg::SignalLineMat>& slms,
		const cv::Mat& R_v2g, const cv::Vec3d& laserDirection,
		const cv::Point3d& laserPoint, cv::Mat& translation_v2g)
	{
		cv::Mat P = (cv::Mat_<double>(3, 1) <<
			laserPoint.x, laserPoint.y, laserPoint.z);
		
		int n = static_cast<int>(slms.size());
		cv::Mat A = cv::Mat::zeros(3 * n, 3, CV_64FC1);
		cv::Mat B = cv::Mat::zeros(3 * n, 1, CV_64FC1);
		for (int i = 0; i < n; i++) {
			cv::Mat H = slms[i].H_;
			cv::Mat R, t;
			H(cv::Rect(0, 0, 3, 3)).copyTo(R);
			H(cv::Rect(3, 0, 1, 3)).copyTo(t);

			double x = slms[i].line_[3];
			double y = slms[i].line_[4];
			double z = slms[i].line_[5];
			cv::Mat Q = (cv::Mat_<double>(3, 1) << x, y, z);

			R.copyTo(A(cv::Rect(0, 3 * i, 3, 3)));

			cv::Mat M = P - R * R_v2g * Q - t;
			M.copyTo(B(cv::Rect(0, 3 * i, 1, 3)));			
		}
		cv::Mat X;
		cv::solve(A, B, X, cv::DECOMP_SVD);

		translation_v2g = (cv::Mat_<double>(3, 1) <<
			X.at<double>(0, 0), X.at<double>(1, 0), X.at<double>(2, 0));
	}


#if 0
	void ComputeTransV2g(const std::vector<lg::SignalLineMat>& slms,
		const cv::Mat& R_v2g, const cv::Vec3d& laserDirection,
		const cv::Point3d& laserPoint, cv::Mat& translation_v2g)
	{
		cv::Mat p = (cv::Mat_<double>(3, 1) <<
			laserPoint.x, laserPoint.y, laserPoint.z);
		cv::Mat R_g2v = R_v2g.t();

		int n = static_cast<int>(slms.size());
		cv::Mat A = cv::Mat::zeros(2 * n, 3, CV_64FC1);
		cv::Mat B = cv::Mat::zeros(2 * n, 1, CV_64FC1);
		for (int i = 0; i < n; i++) {
			cv::Mat antiH = slms[i].antiH_;
			cv::Mat R, t;
			antiH(cv::Rect(0, 0, 3, 3)).copyTo(R);
			antiH(cv::Rect(3, 0, 1, 3)).copyTo(t);

			cv::Mat tmp = R_g2v * (R * p + t);
			double a = tmp.at<double>(0, 0);
			double b = tmp.at<double>(1, 0);
			double c = tmp.at<double>(2, 0);

			double nx = slms[i].line_[0];
			double ny = slms[i].line_[1];
			double nz = slms[i].line_[2];

			double x0 = slms[i].line_[3];
			double y0 = slms[i].line_[4];
			double z0 = slms[i].line_[5];

			cv::Mat M = (cv::Mat_<double>(2, 3) <<
				ny, -nx, 0.,
				nz, 0., -nx);
			M.copyTo(A(cv::Rect(0, 2 * i, 3, 2)));

			double B1 = (b - y0) * nx + (x0 - a) * ny;
			double B2 = (c - z0) * nx + (x0 - a) * nz;
			B.at<double>(2 * i, 0) = B1;
			B.at<double>(2 * i + 1, 0) = B2;
		}

		cv::Mat X;
		cv::solve(A, B, X, cv::DECOMP_SVD);

		cv::Mat trans_g2v = (cv::Mat_<double>(3, 1) <<
			X.at<double>(0, 0), X.at<double>(1, 0), X.at<double>(2, 0) * (-1));

		translation_v2g = (-R_g2v.t()) * trans_g2v;
	}
#endif


	void Update(const cv::Mat& G,
		std::vector<lg::SignalLineMat>& trainSlms,
		std::vector<lg::SignalLineMat>& testSlms)
	{
		double alpha0 = G.at<double>(0, 0);
		double beta0 = G.at<double>(1, 0);
		double e = G.at<double>(2, 0);

		double a1 = G.at<double>(3, 0);
		double a2 = G.at<double>(4, 0);
		double a3 = G.at<double>(5, 0);
		double a4 = G.at<double>(6, 0);
		double a5 = G.at<double>(7, 0);
		double a6 = G.at<double>(8, 0);

		double b1 = G.at<double>(9, 0);
		double b2 = G.at<double>(10, 0);
		double b3 = G.at<double>(11, 0);
		double b4 = G.at<double>(12, 0);
		double b5 = G.at<double>(13, 0);
		double b6 = G.at<double>(14, 0);

		cg::Householder holder;
		holder.SetData(alpha0, beta0, e,
			a1, a2, a3, a4, a5, a6,
			b1, b2, b3, b4, b5, b6);
		for (int i = 0; i < trainSlms.size(); i++) {
			double galvoX = trainSlms[i].signal_.x;
			double galvoY = trainSlms[i].signal_.y;

			cv::Mat H1 = holder.CalculateH1(galvoX, galvoY);
			cv::Mat H2 = holder.CalculateH2(galvoX, galvoY);
			cv::Mat H = H1 * H2;
			cv::Mat invH = H.inv();
			cv::Mat antiH = H2 * H1;

			trainSlms[i].H_ = H.clone();
			trainSlms[i].invH_ = invH.clone();
			trainSlms[i].antiH_ = antiH.clone();
		}
		for (int i = 0; i < testSlms.size(); i++) {
			double galvoX = testSlms[i].signal_.x;
			double galvoY = testSlms[i].signal_.y;

			cv::Mat H1 = holder.CalculateH1(galvoX, galvoY);
			cv::Mat H2 = holder.CalculateH2(galvoX, galvoY);
			cv::Mat H = H1 * H2;
			cv::Mat invH = H.inv();
			cv::Mat antiH = H2 * H1;

			testSlms[i].H_ = H.clone();
			testSlms[i].invH_ = invH.clone();
			testSlms[i].antiH_ = antiH.clone();
		}
	}

	void SolvePhyModel()
	{
		std::map<std::string, cv::Point2d> table;
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
				table[str] = { x / 100., y / 100. };
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

		double e, alpha0, beta0;
		double a1, a2, a3, a4, a5, a6;
		double b1, b2, b3, b4, b5, b6;
		{
			cv::FileStorage fs("../phy_model/galvo_params_order3.yaml", cv::FileStorage::READ);
			// 检查文件是否成功打开
			if (!fs.isOpened()) {
				std::cerr << "无法打开文件!" << std::endl;
			}
			// 读取数据
			fs["galvo"]["alpha"] >> alpha0;
			fs["galvo"]["beta"] >> beta0;
			fs["galvo"]["e"] >> e;
			fs["galvo"]["ka1"] >> a1;
			fs["galvo"]["ka2"] >> a2;
			fs["galvo"]["ka3"] >> a3;
			fs["galvo"]["ka5"] >> a4;
			fs["galvo"]["ka6"] >> a5;
			fs["galvo"]["ka7"] >> a6;

			fs["galvo"]["kb1"] >> b1;
			fs["galvo"]["kb2"] >> b2;
			fs["galvo"]["kb3"] >> b3;
			fs["galvo"]["kb5"] >> b4;
			fs["galvo"]["kb6"] >> b5;
			fs["galvo"]["kb7"] >> b6;

			fs.release();
		}
		cv::Mat G = (cv::Mat_<double>(15, 1) << alpha0, beta0, e,
			a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6);
		cg::Householder holder;
		holder.SetData(alpha0, beta0, e,
			a1, a2, a3, a4, a5, a6,
			b1, b2, b3, b4, b5, b6);

		std::vector<lg::SignalLineMat> slms;
		for (const auto& signal : table) {
			std::string name = signal.first;
			auto iter = allLines.find(name);
			if (iter == allLines.end()) {
				continue;
			}

			double galvoX = signal.second.x;
			double galvoY = signal.second.y;
			cv::Mat H1 = holder.CalculateH1(galvoX, galvoY);
			cv::Mat H2 = holder.CalculateH2(galvoX, galvoY);
			cv::Mat H = H1 * H2;
			cv::Mat invH = H.inv();
			cv::Mat antiH = H2 * H1;

			lg::SignalLineMat slm;
			slm.signal_ = signal.second;
			slm.line_ = iter->second;
			slm.H_ = H.clone();
			slm.invH_ = invH.clone();
			slm.antiH_ = antiH.clone();
			slms.emplace_back(slm);
		}

		std::vector<lg::SignalLineMat> trainSlms, testSlms;
		for (int i = 0; i < slms.size(); i++) {
			if (i % 10 == 0) {
				testSlms.emplace_back(slms[i]);
			}
			else {
				trainSlms.emplace_back(slms[i]);
			}
		}
		std::cout << "train set number: " << trainSlms.size() << std::endl <<
			"test set number: " << testSlms.size() << std::endl;

		cv::Vec3d laserDirection;
		cv::Mat R_v2g;
		ComputeRv2g(trainSlms, R_v2g, laserDirection);
		cv::Mat optimizedR_g2v = R_v2g.t();
		cv::Vec3d optimizedDir = laserDirection;
		lg::Optimization::OptiGalvoRotPhyModel(trainSlms, G, optimizedR_g2v, optimizedDir);
		cv::Mat optimizedR_v2g = optimizedR_g2v.t();
		std::cout << std::endl << std::endl <<
			"优化后的方向: " << optimizedDir[0] << ", " <<
			optimizedDir[1] << ", " << optimizedDir[2] << std::endl;
		Update(G, trainSlms, testSlms);
		ComputeAngleError(testSlms, optimizedDir, optimizedR_v2g);

		cv::Point3d laserPoint(-43.34, 13.888, 0.368);
		cv::Mat trans_v2g;
		ComputeTranslationV2g(trainSlms, optimizedR_v2g, optimizedDir, trans_v2g, laserPoint);
		cv::Mat trans_g2v = (-optimizedR_v2g.t()) * trans_v2g;
		lg::Optimization::OptiGalvoRotTranPhyModel(trainSlms, G, optimizedR_g2v, trans_g2v, optimizedDir, laserPoint);
		//lg::Optimization::OptiGalvoTranPhyModel(trainSlms, G, optimizedR_g2v, trans_g2v, optimizedDir, laserPoint);
		{
			double constantX = -43.34;
			double t = (constantX - laserPoint.x) / optimizedDir[0];
			double Y = laserPoint.y + t * optimizedDir[1];
			double Z = laserPoint.z + t * optimizedDir[2];
			std::cout << std::endl << "LP: " << constantX << "\t" << Y << "\t" << Z;
		}
		Update(G, trainSlms, testSlms);
		ComputeP2L(testSlms, optimizedR_g2v, optimizedDir, trans_g2v, laserPoint);
		ComputeLinesApproximation(testSlms, optimizedR_g2v, optimizedDir, trans_g2v, laserPoint);
		ComputeLinesApproximation2(testSlms, optimizedR_g2v, optimizedDir, trans_g2v, laserPoint);

		Write(G, optimizedR_g2v, optimizedDir, trans_g2v, laserPoint);
	}

	void SolveNoisePhyModel()
	{
		std::map<std::string, cv::Point2d> table;
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
				table[str] = { x / 100., y / 100. };
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

		double e, alpha0, beta0;
		double a1, a2, a3, a4, a5, a6;
		double b1, b2, b3, b4, b5, b6;
		{
			cv::FileStorage fs("../phy_model/galvo_params_order3.yaml", cv::FileStorage::READ);
			// 检查文件是否成功打开
			if (!fs.isOpened()) {
				std::cerr << "无法打开文件!" << std::endl;
			}
			// 读取数据
			fs["galvo"]["alpha"] >> alpha0;
			fs["galvo"]["beta"] >> beta0;
			fs["galvo"]["e"] >> e;
			fs["galvo"]["ka1"] >> a1;
			fs["galvo"]["ka2"] >> a2;
			fs["galvo"]["ka3"] >> a3;
			fs["galvo"]["ka5"] >> a4;
			fs["galvo"]["ka6"] >> a5;
			fs["galvo"]["ka7"] >> a6;

			fs["galvo"]["kb1"] >> b1;
			fs["galvo"]["kb2"] >> b2;
			fs["galvo"]["kb3"] >> b3;
			fs["galvo"]["kb5"] >> b4;
			fs["galvo"]["kb6"] >> b5;
			fs["galvo"]["kb7"] >> b6;

			fs.release();
		}
		cv::Mat G = (cv::Mat_<double>(15, 1) << alpha0, beta0, e,
			a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6);
		cg::Householder holder;
		holder.SetData(alpha0, beta0, e,
			a1, a2, a3, a4, a5, a6,
			b1, b2, b3, b4, b5, b6);

		std::vector<lg::SignalLineMat> slms;
		for (const auto& signal : table) {
			std::string name = signal.first;
			auto iter = allLines.find(name);
			if (iter == allLines.end()) {
				continue;
			}

			double galvoX = signal.second.x;
			double galvoY = signal.second.y;
			cv::Mat H1 = holder.CalculateH1(galvoX, galvoY);
			cv::Mat H2 = holder.CalculateH2(galvoX, galvoY);
			cv::Mat H = H1 * H2;
			cv::Mat invH = H.inv();
			cv::Mat antiH = H2 * H1;

			lg::SignalLineMat slm;
			slm.signal_ = signal.second;
			slm.line_ = iter->second;
			slm.H_ = H.clone();
			slm.invH_ = invH.clone();
			slm.antiH_ = antiH.clone();
			slms.emplace_back(slm);
		}

		std::vector<lg::SignalLineMat> trainSlms, testSlms;
		for (int i = 0; i < slms.size(); i++) {
			if (i % 10 == 0) {
				testSlms.emplace_back(slms[i]);
			}
			else {
				trainSlms.emplace_back(slms[i]);
			}
		}
		std::cout << "train set number: " << trainSlms.size() << std::endl <<
			"test set number: " << testSlms.size() << std::endl;

		cv::Vec3d laserDirection;
		cv::Mat R_v2g;
		ComputeRv2g(trainSlms, R_v2g, laserDirection);
		cv::Mat optimizedR_g2v = R_v2g.t();
		cv::Vec3d optimizedDir = laserDirection;
		lg::Optimization::OptiGalvoRotPhyModel(trainSlms, G, optimizedR_g2v, optimizedDir);
		cv::Mat optimizedR_v2g = optimizedR_g2v.t();
		std::cout << std::endl << std::endl <<
			"优化后的方向: " << optimizedDir[0] << ", " <<
			optimizedDir[1] << ", " << optimizedDir[2] << std::endl;
		Update(G, trainSlms, testSlms);
		ComputeAngleError(testSlms, optimizedDir, optimizedR_v2g);

		{
			double deg = 0.4;
			cv::Mat copyR = optimizedR_v2g.clone();
			optimizedR_v2g = PerturbRotation(copyR, deg);
			optimizedR_g2v = optimizedR_v2g.inv();

			cv::Mat copyDir;
			cv::Rodrigues(optimizedDir, copyDir);
			copyDir = PerturbRotation(copyDir, deg);
			cv::Rodrigues(copyDir, optimizedDir);
		}


		cv::Point3d laserPoint(-43.34, 13.888, 0.368);
		cv::Mat trans_v2g;
		ComputeTranslationV2g(trainSlms, optimizedR_v2g, optimizedDir, trans_v2g, laserPoint);
		cv::Mat trans_g2v = (-optimizedR_v2g.t()) * trans_v2g;
		lg::Optimization::OptiGalvoRotTranPhyModel(trainSlms, G, optimizedR_g2v, trans_g2v, optimizedDir, laserPoint);
		//lg::Optimization::OptiGalvoTranPhyModel(trainSlms, G, optimizedR_g2v, trans_g2v, optimizedDir, laserPoint);
		{
			double constantX = -43.34;
			double t = (constantX - laserPoint.x) / optimizedDir[0];
			double Y = laserPoint.y + t * optimizedDir[1];
			double Z = laserPoint.z + t * optimizedDir[2];
			std::cout << std::endl << "LP: " << constantX << "\t" << Y << "\t" << Z;
		}
		Update(G, trainSlms, testSlms);
		ComputeP2L(testSlms, optimizedR_g2v, optimizedDir, trans_g2v, laserPoint);
		ComputeLinesApproximation(testSlms, optimizedR_g2v, optimizedDir, trans_g2v, laserPoint);
	}

	void ComputeAngleError(const std::vector<lg::SignalLineMat>& slms,
		const cv::Vec3d& optimizedDir, const cv::Mat& optimizedR_v2g)
	{
		std::vector<double> deltaAngle;
		cv::Mat ld = (cv::Mat_<double>(3, 1) <<
			optimizedDir[0], optimizedDir[1], optimizedDir[2]);
		for (int i = 0; i < slms.size(); i++) {
			cv::Mat dir = (cv::Mat_<double>(3, 1) <<
				slms[i].line_[0], slms[i].line_[1], slms[i].line_[2]);

			cv::Mat R = slms[i].H_(cv::Rect(0, 0, 3, 3));
			cv::Mat ret = R * optimizedR_v2g * dir;
			cv::Vec3d v(ret.at<double>(0, 0),
				ret.at<double>(1, 0), ret.at<double>(2, 0));

			double cosTheta = ld.dot(v);
			cosTheta = std::clamp(cosTheta, -1.0, 1.0);
			double rad = std::acos(cosTheta);
			double deg = rad * 180 / CV_PI;
			deltaAngle.emplace_back(deg);
		}
		std::cout << std::endl << std::endl <<
			"优化后的角度误差: " <<
			std::accumulate(deltaAngle.begin(),
				deltaAngle.end(), 0.) / deltaAngle.size() << std::endl;
	}

	void ComputeP2L(const std::vector<lg::SignalLineMat>& slms,
		const cv::Mat& R_g2v, const cv::Vec3d& direction,
		cv::Mat& t_g2v, cv::Point3d& linePoint)
	{
		cv::Mat p = (cv::Mat_<double>(3, 1) << linePoint.x, linePoint.y, linePoint.z);
		cv::Mat dir = (cv::Mat_<double>(3, 1) << direction[0], direction[1], direction[2]);

		std::vector<double> distances, deltaAngle;
		for (int i = 0; i < slms.size(); i++) {
			cv::Mat antiH = slms[i].antiH_;
			cv::Mat R, t;
			antiH(cv::Rect(0, 0, 3, 3)).copyTo(R);
			antiH(cv::Rect(3, 0, 1, 3)).copyTo(t);

			cv::Mat calDir = R_g2v * R * dir;
			cv::Mat tmp = R_g2v * (R * p + t) + t_g2v;

			cv::Vec3d Q(tmp.at<double>(0, 0),
				tmp.at<double>(1, 0), tmp.at<double>(2, 0));
			cv::Vec6d line = slms[i].line_;
			cv::Vec3d direction(line[0], line[1], line[2]);
			cv::Vec3d O(line[3], line[4], line[5]);
			cv::Vec3d v = Q - O;
			cv::Vec3d projection = (v.dot(direction) / (cv::norm(direction) * cv::norm(direction))) * direction;
			cv::Vec3d h = v - projection;
			distances.emplace_back(cv::norm(h));

			cv::Vec3d vec(calDir.at<double>(0, 0),
				calDir.at<double>(1, 0), calDir.at<double>(2, 0));
			double cosTheta = direction.dot(vec);
			cosTheta = std::clamp(cosTheta, -1.0, 1.0);
			double rad = std::acos(cosTheta);
			double deg = rad * 180 / CV_PI;
			deltaAngle.emplace_back(deg);
		}
		std::cout << std::endl << std::endl <<
			"优化后的点线距离: " <<
			std::accumulate(distances.begin(),
				distances.end(), 0.) / distances.size() << std::endl;
		std::cout << std::endl << std::endl <<
			"优化后的角度误差: " <<
			std::accumulate(deltaAngle.begin(),
				deltaAngle.end(), 0.) / deltaAngle.size() << std::endl;
	}

	void ComputeLinesApproximation(const std::vector<lg::SignalLineMat>& slms,
		const cv::Mat& R_g2v, const cv::Vec3d& direction,
		cv::Mat& t_g2v, cv::Point3d& linePoint)
	{
		cv::Mat N = (cv::Mat_<double>(3, 1) <<
			direction[0], direction[1], direction[2]);
		cv::Mat O = (cv::Mat_<double>(3, 1) <<
			linePoint.x, linePoint.y, linePoint.z);

		std::vector<double> distance1, deltaAngle1;
		std::vector<double> distance2, deltaAngle2;
		for (int i = 0; i < slms.size(); i++) {
			cv::Mat antiH = slms[i].antiH_;
			cv::Mat R, t;
			antiH(cv::Rect(0, 0, 3, 3)).copyTo(R);
			antiH(cv::Rect(3, 0, 1, 3)).copyTo(t);

			cv::Mat V = R_g2v * R * N;
			cv::Mat P = R_g2v * (R * O + t) + t_g2v;

			// estimation
			double nx = V.at<double>(0, 0);
			double ny = V.at<double>(1, 0);
			double nz = V.at<double>(2, 0);

			double x = P.at<double>(0, 0);
			double y = P.at<double>(1, 0);
			double z = P.at<double>(2, 0);

			// measurement
			cv::Vec6d line = slms[i].line_;
			double NX = line[0];
			double NY = line[1];
			double NZ = line[2];

			double X = line[3];
			double Y = line[4];
			double Z = line[5];

			cv::Vec3d Pp(X - x, Y - y, Z - z);

			cv::Vec3d a(nx, ny, nz);
			cv::Vec3d b(NX, NY, NZ);

			cv::Vec3d crossProduct = a.cross(b);
			double crossProductMagnitude = cv::norm(crossProduct);
			double dist = std::fabs(Pp.dot(crossProduct)) / crossProductMagnitude;

			double cosTheta = a.dot(b);
			cosTheta = std::clamp(cosTheta, -1.0, 1.0);
			double rad = std::acos(cosTheta);
			double deg = rad * 180 / CV_PI;

			if (std::fabs(slms[i].signal_.x) < 30 &&
				std::fabs(slms[i].signal_.y) < 30) {
				distance1.emplace_back(dist);
				deltaAngle1.emplace_back(deg);
			}
			else {
				distance2.emplace_back(dist);
				deltaAngle2.emplace_back(deg);
			}	
		}

		double d1 = std::accumulate(distance1.begin(),
			distance1.end(), 0.) / distance1.size();
		double d2 = std::accumulate(distance2.begin(),
			distance2.end(), 0.) / distance2.size();
		double alpha1 = std::accumulate(deltaAngle1.begin(),
			deltaAngle1.end(), 0.) / deltaAngle1.size();
		double alpha2 = std::accumulate(deltaAngle2.begin(),
			deltaAngle2.end(), 0.) / deltaAngle2.size();

		std::cout << std::endl << std::endl <<
			"评价两条直线的接近程度D-A: " << std::endl <<
			distance1.size() << "\t" << d1 << "\t" << alpha1 << std::endl <<
			distance2.size() << "\t" << d2 << "\t" << alpha2 << std::endl <<
			0.5 * (d1 + d2) << "\t" << 0.5 * (alpha1 + alpha2) << std::endl;


		{
			std::ofstream out("../phy_model/angle_dist_error_center.yaml");
			out << std::fixed << std::setprecision(7);
			for (int i = 0; i < deltaAngle1.size(); i++) {
				out << deltaAngle1[i] << "\t" << distance1[i] << "\n";
			}
			out.close();
		}
		{
			std::ofstream out("../phy_model/angle_dist_error_edge.yaml");
			out << std::fixed << std::setprecision(7);
			for (int i = 0; i < deltaAngle2.size(); i++) {
				out << deltaAngle2[i] << "\t" << distance2[i] << "\n";
			}
			out.close();
		}
	}

	void ComputeLinesApproximation2(const std::vector<lg::SignalLineMat>& slms,
		const cv::Mat& R_g2v, const cv::Vec3d& direction,
		cv::Mat& t_g2v, cv::Point3d& linePoint)
	{
		cv::Mat N = (cv::Mat_<double>(3, 1) <<
			direction[0], direction[1], direction[2]);
		cv::Mat O = (cv::Mat_<double>(3, 1) <<
			linePoint.x, linePoint.y, linePoint.z);

		std::map<int, std::vector<double>> dist;
		for (int i = 0; i < slms.size(); i++) {
			cv::Mat antiH = slms[i].antiH_;
			cv::Mat R, t;
			antiH(cv::Rect(0, 0, 3, 3)).copyTo(R);
			antiH(cv::Rect(3, 0, 1, 3)).copyTo(t);

			cv::Mat V = R_g2v * R * N;
			cv::Mat P = R_g2v * (R * O + t) + t_g2v;

			// estimation
			double nx = V.at<double>(0, 0);
			double ny = V.at<double>(1, 0);
			double nz = V.at<double>(2, 0);

			double x = P.at<double>(0, 0);
			double y = P.at<double>(1, 0);
			double z = P.at<double>(2, 0);

			// measurement
			cv::Vec6d line = slms[i].line_;
			double NX = line[0];
			double NY = line[1];
			double NZ = line[2];

			double X = line[3];
			double Y = line[4];
			double Z = line[5];

			for (int constantZ = -500; constantZ <= 700; constantZ += 100) {			
				double t1 = (constantZ - z) / nz;
				double calX = x + t1 * nx;
				double calY = y + t1 * ny;

				double t2 = (constantZ - Z) / NZ;
				double refX = X + t2 * NX;
				double refY = Y + t2 * NY;

				double delta = std::sqrt((calX - refX) * (calX - refX) + (calY - refY) * (calY - refY));
				dist[constantZ].emplace_back(delta);
			}
		}

		std::cout << std::endl << std::endl <<
			"评价两条直线的接近程度: " << std::endl;
		for (auto& it : dist) {
			double ave = std::accumulate(it.second.begin(), it.second.end(), 0.) / it.second.size();
			auto[min_it, max_it] = std::minmax_element(it.second.begin(), it.second.end());
			double min_val = *min_it;
			double max_val = *max_it;
			std::cout << it.first << "\t" << ave << "\t" << min_val << "\t" << max_val << "\n";
		}
	}

	void ComputeLinesApproximation(
		const std::vector<cv::Vec6d>& line1,
		const std::vector<cv::Vec6d>& line2)
	{
		std::vector<double> distance, deltaAngle;
		for (int i = 0; i < line1.size(); i++) {
			// estimation
			double nx = line1[i][0];
			double ny = line1[i][1];
			double nz = line1[i][2];

			double x = line1[i][3];
			double y = line1[i][4];
			double z = line1[i][5];

			// measurement
			double NX = line2[i][0];
			double NY = line2[i][1];
			double NZ = line2[i][2];

			double X = line2[i][3];
			double Y = line2[i][4];
			double Z = line2[i][5];

			cv::Vec3d Pp(X - x, Y - y, Z - z);

			cv::Vec3d a(nx, ny, nz);
			cv::Vec3d b(NX, NY, NZ);

			cv::Vec3d crossProduct = a.cross(b);
			double crossProductMagnitude = cv::norm(crossProduct);
			double dist = std::fabs(Pp.dot(crossProduct)) / crossProductMagnitude;
			distance.push_back(dist);

			double cosTheta = a.dot(b);
			cosTheta = std::clamp(cosTheta, -1.0, 1.0);
			double rad = std::acos(cosTheta);
			double deg = rad * 180 / CV_PI;
			deltaAngle.push_back(deg);
		}
		std::cout << std::endl << std::endl <<
			"评价两条直线的接近程度D-A: " <<
			std::accumulate(distance.begin(),
				distance.end(), 0.) / distance.size()
			<< std::endl <<
			std::accumulate(deltaAngle.begin(),
				deltaAngle.end(), 0.) / deltaAngle.size() << std::endl;

		{
			std::ofstream out("../phy_model/angle_dist_error.yaml");
			out << std::fixed << std::setprecision(7);
			for (int i = 0; i < deltaAngle.size(); i++) {
				out << deltaAngle[i] << "\t" << distance[i] << "\n";
			}
			out.close();
		}
	}

	void ComputeLinesApproximation2(
		const std::vector<cv::Vec6d>& line1,
		const std::vector<cv::Vec6d>& line2)
	{
		std::map<int, std::vector<double>> dist;
		for (int i = 0; i < line1.size(); i++) {
			// estimation
			double nx = line1[i][0];
			double ny = line1[i][1];
			double nz = line1[i][2];

			double x = line1[i][3];
			double y = line1[i][4];
			double z = line1[i][5];

			// measurement
			double NX = line2[i][0];
			double NY = line2[i][1];
			double NZ = line2[i][2];

			double X = line2[i][3];
			double Y = line2[i][4];
			double Z = line2[i][5];

			for (int constantZ = -1000; constantZ <= 5000; constantZ += 500) {
				double t1 = (constantZ - z) / nz;
				double calX = x + t1 * nx;
				double calY = y + t1 * ny;

				double t2 = (constantZ - Z) / NZ;
				double refX = X + t2 * NX;
				double refY = Y + t2 * NY;

				double delta = std::sqrt((calX - refX) * (calX - refX) + (calY - refY) * (calY - refY));
				dist[constantZ].emplace_back(delta);
			}
		}
		std::cout << std::endl << std::endl <<
			"评价两条直线的接近程度: " << std::endl;
		for (auto& it : dist) {
			double ave = std::accumulate(it.second.begin(), it.second.end(), 0.) / it.second.size();
			auto[min_it, max_it] = std::minmax_element(it.second.begin(), it.second.end());
			double min_val = *min_it;
			double max_val = *max_it;
			std::cout << it.first << "\t" << ave << "\t" << min_val << "\t" << max_val << "\n";
		}
	}

	void ComputeLinesApproForCamera(
		const std::vector<cv::Point2d>& signals,
		const std::vector<cv::Vec6d>& line1,
		const std::vector<cv::Vec6d>& line2)
	{
		std::vector<double> distance1, deltaAngle1;
		std::vector<double> distance2, deltaAngle2;
		for (int i = 0; i < line1.size(); i++) {
			// estimation
			double nx = line1[i][0];
			double ny = line1[i][1];
			double nz = line1[i][2];

			double x = line1[i][3];
			double y = line1[i][4];
			double z = line1[i][5];

			// measurement
			double NX = line2[i][0];
			double NY = line2[i][1];
			double NZ = line2[i][2];

			double X = line2[i][3];
			double Y = line2[i][4];
			double Z = line2[i][5];

			cv::Vec3d Pp(X - x, Y - y, Z - z);

			cv::Vec3d a(nx, ny, nz);
			cv::Vec3d b(NX, NY, NZ);

			cv::Vec3d crossProduct = a.cross(b);
			double crossProductMagnitude = cv::norm(crossProduct);
			double dist = std::fabs(Pp.dot(crossProduct)) / crossProductMagnitude;

			double cosTheta = a.dot(b);
			cosTheta = std::clamp(cosTheta, -1.0, 1.0);
			double rad = std::acos(cosTheta);
			double deg = rad * 180 / CV_PI;

			if (std::fabs(signals[i].x) < 30 &&
				std::fabs(signals[i].y) < 30) {
				distance1.emplace_back(dist);
				deltaAngle1.emplace_back(deg);
			}
			else {
				distance2.emplace_back(dist);
				deltaAngle2.emplace_back(deg);
			}
		}
		std::cout << std::endl << std::endl <<
			"评价两条直线的接近程度D-A: " << std::endl <<
			distance1.size() << "\t" <<
			std::accumulate(distance1.begin(),
				distance1.end(), 0.) / distance1.size()
			<< "\t" <<
			std::accumulate(deltaAngle1.begin(),
				deltaAngle1.end(), 0.) / deltaAngle1.size()
			<< std::endl <<
			distance2.size() << "\t" <<
			std::accumulate(distance2.begin(),
				distance2.end(), 0.) / distance2.size()
			<< "\t" <<
			std::accumulate(deltaAngle2.begin(),
				deltaAngle2.end(), 0.) / deltaAngle2.size()
			<< std::endl;

		{
			std::ofstream out("../phy_model/angle_dist_error_center.yaml");
			out << std::fixed << std::setprecision(7);
			for (int i = 0; i < deltaAngle1.size(); i++) {
				out << deltaAngle1[i] << "\t" << distance1[i] << "\n";
			}
			out.close();
		}
		{
			std::ofstream out("../phy_model/angle_dist_error_edge.yaml");
			out << std::fixed << std::setprecision(7);
			for (int i = 0; i < deltaAngle2.size(); i++) {
				out << deltaAngle2[i] << "\t" << distance2[i] << "\n";
			}
			out.close();
		}
	}

	void ComputeLinesApproForCamera2(
		const std::vector<cv::Point2d>& signals,
		const std::vector<cv::Vec6d>& line1,
		const std::vector<cv::Vec6d>& line2)
	{
		std::map<int, std::vector<double>> dist;
		for (int i = 0; i < line1.size(); i++) {
			// estimation
			double nx = line1[i][0];
			double ny = line1[i][1];
			double nz = line1[i][2];

			double x = line1[i][3];
			double y = line1[i][4];
			double z = line1[i][5];

			// measurement
			double NX = line2[i][0];
			double NY = line2[i][1];
			double NZ = line2[i][2];

			double X = line2[i][3];
			double Y = line2[i][4];
			double Z = line2[i][5];

			for (int constantZ = -1000; constantZ <= 5000; constantZ += 500) {
				double t1 = (constantZ - z) / nz;
				double calX = x + t1 * nx;
				double calY = y + t1 * ny;

				double t2 = (constantZ - Z) / NZ;
				double refX = X + t2 * NX;
				double refY = Y + t2 * NY;

				double delta = std::sqrt((calX - refX) * (calX - refX) + (calY - refY) * (calY - refY));
				dist[constantZ].emplace_back(delta);
			}
		}
		std::cout << std::endl << std::endl <<
			"评价两条直线的接近程度: " << std::endl;
		for (auto& it : dist) {
			double ave = std::accumulate(it.second.begin(), it.second.end(), 0.) / it.second.size();
			auto[min_it, max_it] = std::minmax_element(it.second.begin(), it.second.end());
			double min_val = *min_it;
			double max_val = *max_it;
			std::cout << it.first << "\t" << ave << "\t" << min_val << "\t" << max_val << "\n";
		}
	}

	cv::Vec3d AddNoiseToUnitVector(const cv::Vec3d& v, double sigma) {
		// 1. 生成两个正交于 v 的基向量 (构建切平面)
		cv::Vec3d e1, e2;
		if (std::abs(v[2]) < 0.99) {
			e1 = v.cross(cv::Vec3d(0, 0, 1));
		}
		else {
			e1 = v.cross(cv::Vec3d(1, 0, 0));
		}
		e1 = e1 / cv::norm(e1);
		e2 = v.cross(e1); // 自动单位化且正交

		// 2. 在切平面上加高斯噪声
		//std::random_device rd;
		std::mt19937 gen(12345);
		std::normal_distribution<double> noise(0.0, sigma);

		double n1 = noise(gen);
		double n2 = noise(gen);
		cv::Vec3d delta = n1 * e1 + n2 * e2;

		// 3. 指数映射（近似）：v' = normalize(v + delta)
		cv::Vec3d noisyVec = v + delta;
		noisyVec = noisyVec / cv::norm(noisyVec); // 投影回单位球面

		double deltaAngle = std::acos(std::max(-1.0, std::min(1.0, v.dot(noisyVec))));
		deltaAngle = deltaAngle * 180 / CV_PI;

		return noisyVec;
	}

	cv::Vec3d PerturbUnitVector(const cv::Vec3d& v, double angle) {
		double rad = angle * CV_PI / 180.0;
		cv::Mat deltaR = (cv::Mat_<double>(3, 3) <<
			cos(rad), -sin(rad), 0,
			sin(rad), cos(rad), 0,
			0, 0, 1);
		cv::Mat R = deltaR * v;

		cv::Vec3d perturbedV(R.at<double>(0, 0), R.at<double>(1, 0), R.at<double>(2, 0));
		cv::normalize(perturbedV, perturbedV);

		double deltaAngle = std::acos(std::max(-1.0, std::min(1.0, v.dot(perturbedV))));
		deltaAngle = deltaAngle * 180 / CV_PI;
		std::cout << "Perturbation: " << deltaAngle << "\n";
		return perturbedV;
	}

	cv::Mat AddNoiseToRotationMatrix(const cv::Mat& R, double sigma) {
		// 1. 生成小幅度的旋转向量（李代数 so(3)）
		//std::random_device rd;
		std::mt19937 gen(12345);
		std::normal_distribution<double> noise(0., sigma);
		double nx = noise(gen);
		double ny = noise(gen);
		double nz = std::sqrt(sigma * sigma - nx * nx - ny * ny);
		cv::Vec3d omega(nx, ny, nz); // 小角度旋转向量

		// 2. 将旋转向量转换为旋转矩阵（Rodrigues）
		cv::Mat R_noise;
		cv::Rodrigues(omega, R_noise);

		// 3. 左乘（或右乘）扰动：R' = R_noise * R
		cv::Mat R_noisy = R_noise * R;

		// （可选）确保 det(R_noisy) == 1（数值误差修正）
		cv::SVD svd(R_noisy);
		cv::Mat R_corrected = svd.u * svd.vt;
		if (cv::determinant(R_corrected) < 0) {
			R_corrected.at<double>(2, 2) *= -1; // 修正反射
		}

		cv::Mat R_diff = R_corrected * R.t(); // 相对旋转
		cv::Vec3d rvec;
		cv::Rodrigues(R_diff, rvec);
		double deltaAngle = cv::norm(rvec);
		deltaAngle = deltaAngle * 180 / CV_PI;
		std::cout << "Perturbation: " << deltaAngle << "\n";

		return R_corrected;
	}

	cv::Mat PerturbRotation(const cv::Mat& R, double angle)
	{
		double rad = angle * CV_PI / 180.0;

		std::mt19937 gen(12345);
		double nx = static_cast<double>(gen()) / RAND_MAX * 2 - 1;
		double ny = static_cast<double>(gen()) / RAND_MAX * 2 - 1;
		double nz = static_cast<double>(gen()) / RAND_MAX * 2 - 1;
		cv::Mat n = (cv::Mat_<double>(3, 1) << nx, ny, nz);
		cv::normalize(n, n);

		nx = n.at<double>(0, 0);
		ny = n.at<double>(1, 0);
		nz = n.at<double>(2, 0);
		cv::Mat I = cv::Mat::eye(3, 3, CV_64FC1);
		cv::Mat K = (cv::Mat_<double>(3, 3) <<
			0, -nz, ny,
			nz, 0, -nx,
			-ny, nx, 0);
		cv::Mat deltaR = cos(rad) * I + (1 - cos(rad)) * n * n.t() + sin(rad) * K;
		cv::Mat R2 = deltaR * R;

		cv::Mat delta = R2 * R.t();
		cv::Vec3d rvec;
		cv::Rodrigues(delta, rvec);
		double deltaAngle = cv::norm(rvec);
		deltaAngle = deltaAngle * 180 / CV_PI;
		std::cout << "Perturbation: " << deltaAngle << "\n";

		return R2;
	}

	void SolveIdealModel()
	{
		std::map<std::string, cv::Point2d> table;	
		{
			std::ifstream file;
			file.open("../phy_model/table-221A.txt");
			std::string str;
			int x, y;
			while (file >> str >> x >> y) {
				if (str.empty()) {
					continue;
				}
				table[str] = { x / 100., y / 100. };
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

		double e, alpha0, beta0;
		double a1, a2, a3, a4, a5, a6;
		double b1, b2, b3, b4, b5, b6;
		{
			cv::FileStorage fs("../phy_model/galvo_params_order1.yaml", cv::FileStorage::READ);
			// 检查文件是否成功打开
			if (!fs.isOpened()) {
				std::cerr << "无法打开文件!" << std::endl;
			}
			// 读取数据
			fs["galvo"]["alpha"] >> alpha0;
			fs["galvo"]["beta"] >> beta0;
			fs["galvo"]["e"] >> e;
			fs["galvo"]["ka1"] >> a1;
			fs["galvo"]["ka2"] >> a2;
			fs["galvo"]["ka3"] >> a3;
			fs["galvo"]["ka5"] >> a4;
			fs["galvo"]["ka6"] >> a5;
			fs["galvo"]["ka7"] >> a6;

			fs["galvo"]["kb1"] >> b1;
			fs["galvo"]["kb2"] >> b2;
			fs["galvo"]["kb3"] >> b3;
			fs["galvo"]["kb5"] >> b4;
			fs["galvo"]["kb6"] >> b5;
			fs["galvo"]["kb7"] >> b6;

			fs.release();
		}
		cv::Mat G = (cv::Mat_<double>(15, 1) << alpha0, beta0, e,
			a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6);
		cg::Householder holder;
		holder.SetData(alpha0, beta0, e,
			a1, a2, a3, a4, a5, a6,
			b1, b2, b3, b4, b5, b6);

		std::vector<lg::SignalLineMat> slms;
		for (const auto& signal : table) {
			std::string name = signal.first;
			auto iter = allLines.find(name);
			if (iter == allLines.end()) {
				continue;
			}

			double galvoX = signal.second.x;
			double galvoY = signal.second.y;
			cv::Mat H1 = holder.CalculateH1(galvoX, galvoY);
			cv::Mat H2 = holder.CalculateH2(galvoX, galvoY);
			cv::Mat H = H1 * H2;
			cv::Mat invH = H.inv();
			cv::Mat antiH = H2 * H1;

			lg::SignalLineMat slm;
			slm.signal_ = signal.second;
			slm.line_ = iter->second;
			slm.H_ = H.clone();
			slm.invH_ = invH.clone();
			slm.antiH_ = antiH.clone();
			slms.emplace_back(slm);
		}
		std::vector<lg::SignalLineMat> trainSlms, testSlms;
		for (int i = 0; i < slms.size(); i++) {
			if (i % 10 == 0) {
				testSlms.emplace_back(slms[i]);
			}
			else {
				trainSlms.emplace_back(slms[i]);
			}
		}
		std::cout << "train set number: " << trainSlms.size() << std::endl <<
			"test set number: " << testSlms.size() << std::endl;

		cv::Vec3d laserDirection = { 1., 0., 0. };
		cv::Mat R_v2g;
		ComputeRv2g(trainSlms, laserDirection, R_v2g);	
		ComputeAngleError(testSlms, laserDirection, R_v2g);

		//cv::Point3d laserPoint(-43.34, 13.888, 0.368);
		cv::Point3d laserPoint(-100., 0., 0.);
		cv::Mat trans_v2g;
		ComputeTransV2g(trainSlms, R_v2g, laserDirection, laserPoint, trans_v2g);
	
		cv::Mat R_g2v = R_v2g.t();
		cv::Mat trans_g2v = (-R_v2g.t()) * trans_v2g;
		ComputeP2L(testSlms, R_g2v, laserDirection, trans_g2v, laserPoint);
		ComputeLinesApproximation(testSlms, R_g2v, laserDirection, trans_g2v, laserPoint);
		ComputeLinesApproximation2(testSlms, R_g2v, laserDirection, trans_g2v, laserPoint);

		lg::Optimization::OptiImprovedIdealModel(trainSlms, G, R_g2v, trans_g2v, laserDirection, laserPoint);
		Update(G, trainSlms, testSlms);
		ComputeP2L(testSlms, R_g2v, laserDirection, trans_g2v, laserPoint);
		ComputeLinesApproximation(testSlms, R_g2v, laserDirection, trans_g2v, laserPoint);
		ComputeLinesApproximation2(testSlms, R_g2v, laserDirection, trans_g2v, laserPoint);
	}

	void SolveCameraModel()
	{
		std::map<std::string, cv::Point2d> table;
		{
			std::ifstream file;
			file.open("../phy_model/table-221A.txt");
			std::string str;
			int x, y;
			while (file >> str >> x >> y) {
				if (str.empty()) {
					continue;
				}
				table[str] = { x / 100., y / 100. };
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
		std::vector<lg::SignalLineMat> slms;
		for (const auto& signal : table) {
			std::string name = signal.first;
			auto iter = allLines.find(name);
			if (iter == allLines.end()) {
				continue;
			}

			lg::SignalLineMat slm;
			slm.signal_ = signal.second;
			slm.line_ = iter->second;
			slms.emplace_back(slm);
		}
		std::vector<lg::SignalLineMat> trainSlms, testSlms;
		for (int i = 0; i < slms.size(); i++) {
			if (i % 10 == 0) {
				testSlms.emplace_back(slms[i]);
			}
			else {
				trainSlms.emplace_back(slms[i]);
			}
		}
		std::cout << "train set number: " << trainSlms.size() << std::endl <<
			"test set number: " << testSlms.size() << std::endl;

		std::map<int, cv::Mat> homographys;
		for (int Z = 3500; Z <= 4500; Z += 100) {
			std::vector<cv::Point2f> objPois;
			std::vector<cv::Point2f> imgPois;
			for (const auto& slm : trainSlms) {			
				cv::Vec6d line = slm.line_;
				double nx = line[0];
				double ny = line[1];
				double nz = line[2];
				double x = line[3];
				double y = line[4];
				double z = line[5];
				double t = (Z - z) / nz;
				double X = x + t * nx;
				double Y = y + t * ny;

				objPois.emplace_back(X, Y);
				imgPois.emplace_back(slm.signal_);
			}
			cv::Mat H = cv::findHomography(imgPois, objPois);
			homographys[Z] = H;
		}

		std::vector<std::vector<cv::Point3d>> poisInline;
		poisInline.resize(testSlms.size());
		for (const auto& homo : homographys) {
			int Z = homo.first;
			cv::Mat H = homo.second;
			std::vector<cv::Point2f> imgPois;
			for (const auto& slm : testSlms) {
				imgPois.emplace_back(slm.signal_);
			}
			std::vector<cv::Point2f> projectedPoints;
			cv::perspectiveTransform(imgPois, projectedPoints, H);
			for (int i = 0; i < projectedPoints.size(); i++) {
				poisInline[i].emplace_back(
					projectedPoints[i].x, projectedPoints[i].y, Z);
			}
		}
		
		std::vector<cv::Vec6d> lines1, lines2;
		for (int i = 0; i < poisInline.size(); i++) {
			cv::Vec6d estimatedLine;
			double maxError = FitLine3d(poisInline[i], estimatedLine);
			lines1.emplace_back(estimatedLine);
			lines2.emplace_back(testSlms[i].line_);
		}

		std::vector<cv::Point2d> signals;
		for (const auto& slm : testSlms) {
			signals.emplace_back(slm.signal_);
		}
		ComputeLinesApproForCamera(signals, lines1, lines2);
	}

	void Write(const cv::Mat& G,
		const cv::Mat& R_g2v,
		const cv::Vec3d& direction,
		const cv::Mat& translation_g2v,
		const cv::Point3d& linePoint)
	{
		cv::FileStorage fs("../phy_model/glss_phy_param.yaml", cv::FileStorage::WRITE);
		if (!fs.isOpened()) {
			std::cerr << "无法打开文件!" << std::endl;
			return;
		}

		fs << "G" << G;
		fs << "R_g2v" << R_g2v;
		fs << "direction" << direction;
		fs << "translation_g2v" << translation_g2v;
		fs << "linePoint" << linePoint;

		fs.release();
	}
}