#pragma once

#include "DS.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace lg {
	struct VectorAngleEqual
	{
		cv::Vec3d n1_, n2_;

		VectorAngleEqual(const cv::Vec3d& n1, const cv::Vec3d& n2)
			:n1_(n1), n2_(n2) {}

		template <typename T>
		bool operator()(const T* coneNormal, T* residuals) const
		{
			T n1[3] = { T(n1_[0]),T(n1_[1]), T(n1_[2]) };
			T n2[3] = { T(n2_[0]),T(n2_[1]), T(n2_[2]) };

			T ref[2] = { T(coneNormal[0]), T(coneNormal[1]) };
			T z = ceres::sqrt(T(1.) - ref[0] * ref[0] - ref[1] * ref[1]);

			T cosTheta1 = n1[0] * ref[0] + n1[1] * ref[1] + n1[2] * z;
			T cosTheta2 = n2[0] * ref[0] + n2[1] * ref[1] + n2[2] * z;

			residuals[0] = cosTheta1 - cosTheta2;
			return true;
		}
	};

	struct GalvoDeflectionXOrder1
	{
		double deltaAlpha_;
		double galvoX_, galvoY_;

		GalvoDeflectionXOrder1(double galvoX, double galvoY, double deltaAlpha)
			:galvoX_(galvoX), galvoY_(galvoY), deltaAlpha_(deltaAlpha) {}

		template <typename T>
		bool operator()(const T* G, T* residuals) const
		{
			const T& alpha0 = T(G[0]);
			const T& k1 = T(G[1]);
			const T& k2 = T(G[2]);
			const T& k3 = T(G[3]);
			const T& k4 = T(G[4]);
			const T& k5 = T(G[5]);
			const T& k6 = T(G[6]);
			const T& k7 = T(G[7]);
			const T& k8 = T(G[8]);

			T a = T(galvoX_);
			T b = T(galvoY_);

			T delta_alpha =
				alpha0 +
				k1 * a /*+
				k5 * b*/;

			residuals[0] = delta_alpha - T(deltaAlpha_);
			return true;
		}
	};

	struct GalvoDeflectionXOrder2
	{
		double deltaAlpha_;
		double galvoX_, galvoY_;

		GalvoDeflectionXOrder2(double galvoX, double galvoY, double deltaAlpha)
			:galvoX_(galvoX), galvoY_(galvoY), deltaAlpha_(deltaAlpha) {}

		template <typename T>
		bool operator()(const T* G, T* residuals) const
		{
			const T& alpha0 = T(G[0]);
			const T& k1 = T(G[1]);
			const T& k2 = T(G[2]);
			const T& k3 = T(G[3]);
			const T& k4 = T(G[4]);
			const T& k5 = T(G[5]);
			const T& k6 = T(G[6]);
			const T& k7 = T(G[7]);
			const T& k8 = T(G[8]);

			T a = T(galvoX_);
			T b = T(galvoY_);

			T delta_alpha =
				alpha0 +
				k1 * a +
				k2 * a * a /*+
				k5 * b +
				k6 * b * b*/;

			residuals[0] = delta_alpha - T(deltaAlpha_);
			return true;
		}
	};

	struct GalvoDeflectionXOrder3
	{
		double deltaAlpha_;
		double galvoX_, galvoY_;

		GalvoDeflectionXOrder3(double galvoX, double galvoY, double deltaAlpha)
			:galvoX_(galvoX), galvoY_(galvoY), deltaAlpha_(deltaAlpha) {}

		template <typename T>
		bool operator()(const T* G, T* residuals) const
		{
			const T& alpha0 = T(G[0]);
			const T& k1 = T(G[1]);
			const T& k2 = T(G[2]);
			const T& k3 = T(G[3]);
			const T& k4 = T(G[4]);
			const T& k5 = T(G[5]);
			const T& k6 = T(G[6]);
			const T& k7 = T(G[7]);
			const T& k8 = T(G[8]);

			T a = T(galvoX_);
			T b = T(galvoY_);

			T delta_alpha =
				alpha0 +
				k1 * a +
				k2 * a * a +
				k3 * a * a * a +
				k5 * b +
				k6 * b * b +
				k7 * b * b * b;

			residuals[0] = delta_alpha - T(deltaAlpha_);
			return true;
		}
	};

	struct GalvoDeflectionXOrder4
	{
		double deltaAlpha_;
		double galvoX_, galvoY_;

		GalvoDeflectionXOrder4(double galvoX, double galvoY, double deltaAlpha)
			:galvoX_(galvoX), galvoY_(galvoY), deltaAlpha_(deltaAlpha) {}

		template <typename T>
		bool operator()(const T* G, T* residuals) const
		{
			const T& alpha0 = T(G[0]);
			const T& k1 = T(G[1]);
			const T& k2 = T(G[2]);
			const T& k3 = T(G[3]);
			const T& k4 = T(G[4]);
			const T& k5 = T(G[5]);
			const T& k6 = T(G[6]);
			const T& k7 = T(G[7]);
			const T& k8 = T(G[8]);

			T a = T(galvoX_);
			T b = T(galvoY_);

			T delta_alpha =
				alpha0 +
				k1 * a +
				k2 * a * a +
				k3 * a * a * a +
				k4 * a * a * a * a /*+
				k5 * b +
				k6 * b * b +
				k7 * b * b * b +
				k8 * b * b * b * b*/;

			residuals[0] = delta_alpha - T(deltaAlpha_);
			return true;
		}
	};

	struct GalvoDeflectionYOrder1
	{
		double deltaBeta_;
		double galvoX_, galvoY_;

		GalvoDeflectionYOrder1(double galvoX, double galvoY, double deltaBeta)
			:galvoX_(galvoX), galvoY_(galvoY), deltaBeta_(deltaBeta) {}

		template <typename T>
		bool operator()(const T* G, T* residuals) const
		{
			const T& beta0 = T(G[0]);
			const T& k1 = T(G[1]);
			const T& k2 = T(G[2]);
			const T& k3 = T(G[3]);
			const T& k4 = T(G[4]);
			const T& k5 = T(G[5]);
			const T& k6 = T(G[6]);
			const T& k7 = T(G[7]);
			const T& k8 = T(G[8]);

			T a = T(galvoX_);
			T b = T(galvoY_);

			T delta_beta =
				beta0 +
				k1 * b /*+
				k5 * a*/;

			residuals[0] = delta_beta - T(deltaBeta_);
			return true;
		}
	};

	struct GalvoDeflectionYOrder2
	{
		double deltaBeta_;
		double galvoX_, galvoY_;

		GalvoDeflectionYOrder2(double galvoX, double galvoY, double deltaBeta)
			:galvoX_(galvoX), galvoY_(galvoY), deltaBeta_(deltaBeta) {}

		template <typename T>
		bool operator()(const T* G, T* residuals) const
		{
			const T& beta0 = T(G[0]);
			const T& k1 = T(G[1]);
			const T& k2 = T(G[2]);
			const T& k3 = T(G[3]);
			const T& k4 = T(G[4]);
			const T& k5 = T(G[5]);
			const T& k6 = T(G[6]);
			const T& k7 = T(G[7]);
			const T& k8 = T(G[8]);

			T a = T(galvoX_);
			T b = T(galvoY_);

			T delta_beta =
				beta0 +
				k1 * b +
				k2 * b * b /*+			
				k5 * a +
				k6 * a * a*/;

			residuals[0] = delta_beta - T(deltaBeta_);
			return true;
		}
	};

	struct GalvoDeflectionYOrder3
	{
		double deltaBeta_;
		double galvoX_, galvoY_;

		GalvoDeflectionYOrder3(double galvoX, double galvoY, double deltaBeta)
			:galvoX_(galvoX), galvoY_(galvoY), deltaBeta_(deltaBeta) {}

		template <typename T>
		bool operator()(const T* G, T* residuals) const
		{
			const T& beta0 = T(G[0]);
			const T& k1 = T(G[1]);
			const T& k2 = T(G[2]);
			const T& k3 = T(G[3]);
			const T& k4 = T(G[4]);
			const T& k5 = T(G[5]);
			const T& k6 = T(G[6]);
			const T& k7 = T(G[7]);
			const T& k8 = T(G[8]);

			T a = T(galvoX_);
			T b = T(galvoY_);

			T delta_beta =
				beta0 +
				k1 * b +
				k2 * b * b +
				k3 * b * b * b +				
				k5 * a +
				k6 * a * a +
				k7 * a * a * a;

			residuals[0] = delta_beta - T(deltaBeta_);
			return true;
		}
	};

	struct GalvoDeflectionYOrder4
	{
		double deltaBeta_;
		double galvoX_, galvoY_;

		GalvoDeflectionYOrder4(double galvoX, double galvoY, double deltaBeta)
			:galvoX_(galvoX), galvoY_(galvoY), deltaBeta_(deltaBeta) {}

		template <typename T>
		bool operator()(const T* G, T* residuals) const
		{
			const T& beta0 = T(G[0]);
			const T& k1 = T(G[1]);
			const T& k2 = T(G[2]);
			const T& k3 = T(G[3]);
			const T& k4 = T(G[4]);
			const T& k5 = T(G[5]);
			const T& k6 = T(G[6]);
			const T& k7 = T(G[7]);
			const T& k8 = T(G[8]);

			T a = T(galvoX_);
			T b = T(galvoY_);

			T delta_beta =
				beta0 +
				k1 * b +
				k2 * b * b +
				k3 * b * b * b +
				k4 * b * b * b * b /*+
				k5 * a +
				k6 * a * a +
				k7 * a * a * a +
				k8 * a * a * a * a*/;

			residuals[0] = delta_beta - T(deltaBeta_);
			return true;
		}
	};

	struct RotationGalvoAdjustmentO3
	{
		cv::Vec3d n_;
		double galvoX_, galvoY_;

		RotationGalvoAdjustmentO3(
			const cv::Vec3d& n, double galvoX, double galvoY)
			:n_(n), galvoX_(galvoX), galvoY_(galvoY) {}

		template <typename T>
		bool operator()(const T* G, const T* rvec_g2v,
			const T* laserDirection, T* residuals) const
		{
			T ld[3] = { T(laserDirection[0]), T(laserDirection[1]), T(laserDirection[2]) };

			const T& alpha0 = T(G[0]);
			const T& beta0 = T(G[1]);

			const T& ka1 = T(G[2]);
			const T& ka2 = T(G[3]);
			const T& ka3 = T(G[4]);
			const T& ka4 = T(G[5]);
			const T& ka5 = T(G[6]);
			const T& ka6 = T(G[7]);

			const T& kb1 = T(G[8]);
			const T& kb2 = T(G[9]);
			const T& kb3 = T(G[10]);
			const T& kb4 = T(G[11]);
			const T& kb5 = T(G[12]);
			const T& kb6 = T(G[13]);

			T a1 = T(galvoX_);
			T b1 = T(galvoY_);

			T alpha1 =
				alpha0 +
				ka1 * a1 +
				ka2 * a1 * a1 +
				ka3 * a1 * a1 * a1 +
				ka4 * b1 +
				ka5 * b1 * b1 +
				ka6 * b1 * b1 * b1;
			alpha1 = alpha1 / T(180.0) * T(CV_PI);

			T beta1 =
				beta0 +
				kb1 * b1 +
				kb2 * b1 * b1 +
				kb3 * b1 * b1 * b1 +
				kb4 * a1 +
				kb5 * a1 * a1 +
				kb6 * a1 * a1 * a1;
			beta1 = beta1 / T(180.0) * T(CV_PI);

			//mirror-1
			T sinAlpha1 = ceres::sin(alpha1);
			T cosAlpha1 = ceres::cos(alpha1);

			T mirrorA_ld[3];
			mirrorA_ld[0] = ld[0] * (T(1.0) - T(2.0) * cosAlpha1 * cosAlpha1) +
				ld[1] * (T(-2.0) * cosAlpha1 * sinAlpha1);

			mirrorA_ld[1] = ld[0] * (T(-2.0) * cosAlpha1 * sinAlpha1) +
				ld[1] * (T(1.0) - T(2.0) * sinAlpha1 * sinAlpha1);

			mirrorA_ld[2] = ld[2];

			//mirror-2
			T sinBeta1 = ceres::sin(beta1);
			T cosBeta1 = ceres::cos(beta1);

			T mirrorB_ld[3];
			mirrorB_ld[0] = mirrorA_ld[0];

			mirrorB_ld[1] = mirrorA_ld[1] * (T(1.0) - T(2.0) * sinBeta1 * sinBeta1) +
				mirrorA_ld[2] * (T(-2.0) * cosBeta1 * sinBeta1);

			mirrorB_ld[2] = mirrorA_ld[1] * (T(-2.0) * cosBeta1 * sinBeta1) +
				mirrorA_ld[2] * (T(1.0) - T(2.0) * cosBeta1 * cosBeta1);

			T ldv[3];
			ceres::AngleAxisRotatePoint(rvec_g2v, mirrorB_ld, ldv);

			T n[3] = { T(n_[0]), T(n_[1]), T(n_[2]) };

			residuals[0] = ldv[0] - n[0];
			residuals[1] = ldv[1] - n[1];
			residuals[2] = ldv[2] - n[2];
			residuals[3] = ld[0] * ld[0] + ld[1] * ld[1] + ld[2] * ld[2] - T(1.);
			return true;
		}
	};

	struct TranslationGalvoAdjustmentO3
	{
		cv::Vec6d line_;
		double galvoX_, galvoY_;

		TranslationGalvoAdjustmentO3(
			const cv::Vec6d& line,
			double galvoX, double galvoY)
			:line_(line), galvoX_(galvoX), galvoY_(galvoY) {}

		template <typename T>
		bool operator()(const T* G, const T* E, const T* rvec_g2v, const T* tvec_g2v,
			const T* laserDirection, const T* laserPoint, T* residuals) const
		{
			T pg[3] = { T(laserPoint[0]), T(laserPoint[1]), T(laserPoint[2]) };			

			const T& alpha0 = T(G[0]);
			const T& beta0 = T(G[1]);

			const T& ka1 = T(G[2]);
			const T& ka2 = T(G[3]);
			const T& ka3 = T(G[4]);
			const T& ka4 = T(G[5]);
			const T& ka5 = T(G[6]);
			const T& ka6 = T(G[7]);

			const T& kb1 = T(G[8]);
			const T& kb2 = T(G[9]);
			const T& kb3 = T(G[10]);
			const T& kb4 = T(G[11]);
			const T& kb5 = T(G[12]);
			const T& kb6 = T(G[13]);

			const T& e = T(E[0]);

			T a1 = T(galvoX_);
			T b1 = T(galvoY_);

			T alpha1 =
				alpha0 +
				ka1 * a1 +
				ka2 * a1 * a1 +
				ka3 * a1 * a1 * a1 +
				ka4 * b1 +
				ka5 * b1 * b1 +
				ka6 * b1 * b1 * b1;
			alpha1 = alpha1 / T(180.0) * T(CV_PI);

			T beta1 =
				beta0 +
				kb1 * b1 +
				kb2 * b1 * b1 +
				kb3 * b1 * b1 * b1 +
				kb4 * a1 +
				kb5 * a1 * a1 +
				kb6 * a1 * a1 * a1;
			beta1 = beta1 / T(180.0) * T(CV_PI);

			//mirror-1
			T sinAlpha1 = ceres::sin(alpha1);
			T cosAlpha1 = ceres::cos(alpha1);

			T mirrorA_p1[3];
			mirrorA_p1[0] = pg[0] * (T(1.0) - T(2.0) * cosAlpha1 * cosAlpha1) +
				pg[1] * (T(-2.0) * cosAlpha1 * sinAlpha1) +
				T(2.0) * e * cosAlpha1 * sinAlpha1;

			mirrorA_p1[1] = pg[0] * (T(-2.0) * cosAlpha1 * sinAlpha1) +
				pg[1] * (T(1.0) - T(2.0) * sinAlpha1 * sinAlpha1) +
				T(2.0) * e * sinAlpha1 * sinAlpha1;

			mirrorA_p1[2] = pg[2];

			//mirror-2
			T sinBeta1 = ceres::sin(beta1);
			T cosBeta1 = ceres::cos(beta1);

			T mirrorB_p1[3];
			mirrorB_p1[0] = mirrorA_p1[0];

			mirrorB_p1[1] = mirrorA_p1[1] * (T(1.0) - T(2.0) * sinBeta1 * sinBeta1) +
				mirrorA_p1[2] * (T(-2.0) * cosBeta1 * sinBeta1);

			mirrorB_p1[2] = mirrorA_p1[1] * (T(-2.0) * cosBeta1 * sinBeta1) +
				mirrorA_p1[2] * (T(1.0) - T(2.0) * cosBeta1 * cosBeta1);

			T pv[3];
			ceres::AngleAxisRotatePoint(rvec_g2v, mirrorB_p1, pv);
			pv[0] += tvec_g2v[0];
			pv[1] += tvec_g2v[1];
			pv[2] += tvec_g2v[2];

			T m[3] = { pv[0] - T(line_[3]), pv[1] - T(line_[4]), pv[2] - T(line_[5]) };
			T n[3] = { T(line_[0]), T(line_[1]), T(line_[2]) };

			residuals[0] = m[0] * n[1] - m[1] * n[0];
			residuals[1] = m[1] * n[2] - m[2] * n[1];
			residuals[2] = m[2] * n[0] - m[0] * n[2];			
			return true;
		}
	};

	struct RotTranGalvoAdjustment03
	{
		cv::Vec6d line_;
		double galvoX_, galvoY_;

		RotTranGalvoAdjustment03(
			const cv::Vec6d& line,
			double galvoX, double galvoY)
			:line_(line), galvoX_(galvoX), galvoY_(galvoY) {}

		template <typename T>
		bool operator()(const T* G, const T* E, const T* rvec_g2v, const T* tvec_g2v,
			const T* laserDirection, const T* laserPoint, T* residuals) const
		{
			T pg[3] = { T(laserPoint[0]), T(laserPoint[1]), T(laserPoint[2]) };
			T ld[3] = { T(laserDirection[0]), T(laserDirection[1]), T(laserDirection[2]) };

			const T& alpha0 = T(G[0]);
			const T& beta0 = T(G[1]);

			const T& ka1 = T(G[2]);
			const T& ka2 = T(G[3]);
			const T& ka3 = T(G[4]);
			const T& ka4 = T(G[5]);
			const T& ka5 = T(G[6]);
			const T& ka6 = T(G[7]);

			const T& kb1 = T(G[8]);
			const T& kb2 = T(G[9]);
			const T& kb3 = T(G[10]);
			const T& kb4 = T(G[11]);
			const T& kb5 = T(G[12]);
			const T& kb6 = T(G[13]);

			const T& e = T(E[0]);

			T a1 = T(galvoX_);
			T b1 = T(galvoY_);

			T alpha1 =
				alpha0 +
				ka1 * a1 +
				ka2 * a1 * a1 +
				ka3 * a1 * a1 * a1 +
				ka4 * b1 +
				ka5 * b1 * b1 +
				ka6 * b1 * b1 * b1;
			alpha1 = alpha1 / T(180.0) * T(CV_PI);

			T beta1 =
				beta0 +
				kb1 * b1 +
				kb2 * b1 * b1 +
				kb3 * b1 * b1 * b1 +
				kb4 * a1 +
				kb5 * a1 * a1 +
				kb6 * a1 * a1 * a1;
			beta1 = beta1 / T(180.0) * T(CV_PI);

			//mirror-1
			T sinAlpha1 = ceres::sin(alpha1);
			T cosAlpha1 = ceres::cos(alpha1);

			T mirrorA_p1[3];
			mirrorA_p1[0] = pg[0] * (T(1.0) - T(2.0) * cosAlpha1 * cosAlpha1) +
				pg[1] * (T(-2.0) * cosAlpha1 * sinAlpha1) +
				T(2.0) * e * cosAlpha1 * sinAlpha1;

			mirrorA_p1[1] = pg[0] * (T(-2.0) * cosAlpha1 * sinAlpha1) +
				pg[1] * (T(1.0) - T(2.0) * sinAlpha1 * sinAlpha1) +
				T(2.0) * e * sinAlpha1 * sinAlpha1;

			mirrorA_p1[2] = pg[2];


			T mirrorA_ld[3];
			mirrorA_ld[0] = ld[0] * (T(1.0) - T(2.0) * cosAlpha1 * cosAlpha1) +
				ld[1] * (T(-2.0) * cosAlpha1 * sinAlpha1);

			mirrorA_ld[1] = ld[0] * (T(-2.0) * cosAlpha1 * sinAlpha1) +
				ld[1] * (T(1.0) - T(2.0) * sinAlpha1 * sinAlpha1);

			mirrorA_ld[2] = ld[2];

			//mirror-2
			T sinBeta1 = ceres::sin(beta1);
			T cosBeta1 = ceres::cos(beta1);

			T mirrorB_p1[3];
			mirrorB_p1[0] = mirrorA_p1[0];

			mirrorB_p1[1] = mirrorA_p1[1] * (T(1.0) - T(2.0) * sinBeta1 * sinBeta1) +
				mirrorA_p1[2] * (T(-2.0) * cosBeta1 * sinBeta1);

			mirrorB_p1[2] = mirrorA_p1[1] * (T(-2.0) * cosBeta1 * sinBeta1) +
				mirrorA_p1[2] * (T(1.0) - T(2.0) * cosBeta1 * cosBeta1);

			T mirrorB_ld[3];
			mirrorB_ld[0] = mirrorA_ld[0];

			mirrorB_ld[1] = mirrorA_ld[1] * (T(1.0) - T(2.0) * sinBeta1 * sinBeta1) +
				mirrorA_ld[2] * (T(-2.0) * cosBeta1 * sinBeta1);

			mirrorB_ld[2] = mirrorA_ld[1] * (T(-2.0) * cosBeta1 * sinBeta1) +
				mirrorA_ld[2] * (T(1.0) - T(2.0) * cosBeta1 * cosBeta1);

			T pv[3];
			ceres::AngleAxisRotatePoint(rvec_g2v, mirrorB_p1, pv);
			pv[0] += tvec_g2v[0];
			pv[1] += tvec_g2v[1];
			pv[2] += tvec_g2v[2];

			T ldv[3];
			ceres::AngleAxisRotatePoint(rvec_g2v, mirrorB_ld, ldv);

			T m[3] = { pv[0] - T(line_[3]), pv[1] - T(line_[4]), pv[2] - T(line_[5]) };
			T n[3] = { T(line_[0]), T(line_[1]), T(line_[2]) };

			residuals[0] = m[0] * n[1] - m[1] * n[0];
			residuals[1] = m[1] * n[2] - m[2] * n[1];
			residuals[2] = m[2] * n[0] - m[0] * n[2];

			residuals[3] = (ldv[0] - n[0]) * T(100.);
			residuals[4] = (ldv[1] - n[1]) * T(100.);
			residuals[5] = (ldv[2] - n[2]) * T(100.);

			//residuals[3] = (ldv[0] - n[0]);
			//residuals[4] = (ldv[1] - n[1]);
			//residuals[5] = (ldv[2] - n[2]);

			residuals[6] = ld[0] * ld[0] + ld[1] * ld[1] + ld[2] * ld[2] - T(1.);
			return true;
		}
	};

	struct RotTranGalvoAdjustment01
	{
		cv::Vec6d line_;
		double galvoX_, galvoY_;

		RotTranGalvoAdjustment01(
			const cv::Vec6d& line,
			double galvoX, double galvoY)
			:line_(line), galvoX_(galvoX), galvoY_(galvoY) {}

		template <typename T>
		bool operator()(const T* G, const T* E, const T* rvec_g2v, const T* tvec_g2v,
			const T* laserDirection, const T* laserPoint, T* residuals) const
		{
			T pg[3] = { T(laserPoint[0]), T(laserPoint[1]), T(laserPoint[2]) };
			T ld[3] = { T(laserDirection[0]), T(laserDirection[1]), T(laserDirection[2]) };

			const T& alpha0 = T(G[0]);
			const T& beta0 = T(G[1]);

			const T& ka1 = T(G[2]);
			const T& ka2 = T(G[3]);
			const T& ka3 = T(G[4]);
			const T& ka4 = T(G[5]);
			const T& ka5 = T(G[6]);
			const T& ka6 = T(G[7]);

			const T& kb1 = T(G[8]);
			const T& kb2 = T(G[9]);
			const T& kb3 = T(G[10]);
			const T& kb4 = T(G[11]);
			const T& kb5 = T(G[12]);
			const T& kb6 = T(G[13]);

			const T& e = T(E[0]);

			T a1 = T(galvoX_);
			T b1 = T(galvoY_);

			T alpha1 =
				alpha0 +
				ka1 * a1;
			alpha1 = alpha1 / T(180.0) * T(CV_PI);

			T beta1 =
				beta0 +
				kb1 * b1;
			beta1 = beta1 / T(180.0) * T(CV_PI);

			//mirror-1
			T sinAlpha1 = ceres::sin(alpha1);
			T cosAlpha1 = ceres::cos(alpha1);

			T mirrorA_p1[3];
			mirrorA_p1[0] = pg[0] * (T(1.0) - T(2.0) * cosAlpha1 * cosAlpha1) +
				pg[1] * (T(-2.0) * cosAlpha1 * sinAlpha1) +
				T(2.0) * e * cosAlpha1 * sinAlpha1;

			mirrorA_p1[1] = pg[0] * (T(-2.0) * cosAlpha1 * sinAlpha1) +
				pg[1] * (T(1.0) - T(2.0) * sinAlpha1 * sinAlpha1) +
				T(2.0) * e * sinAlpha1 * sinAlpha1;

			mirrorA_p1[2] = pg[2];


			T mirrorA_ld[3];
			mirrorA_ld[0] = ld[0] * (T(1.0) - T(2.0) * cosAlpha1 * cosAlpha1) +
				ld[1] * (T(-2.0) * cosAlpha1 * sinAlpha1);

			mirrorA_ld[1] = ld[0] * (T(-2.0) * cosAlpha1 * sinAlpha1) +
				ld[1] * (T(1.0) - T(2.0) * sinAlpha1 * sinAlpha1);

			mirrorA_ld[2] = ld[2];

			//mirror-2
			T sinBeta1 = ceres::sin(beta1);
			T cosBeta1 = ceres::cos(beta1);

			T mirrorB_p1[3];
			mirrorB_p1[0] = mirrorA_p1[0];

			mirrorB_p1[1] = mirrorA_p1[1] * (T(1.0) - T(2.0) * sinBeta1 * sinBeta1) +
				mirrorA_p1[2] * (T(-2.0) * cosBeta1 * sinBeta1);

			mirrorB_p1[2] = mirrorA_p1[1] * (T(-2.0) * cosBeta1 * sinBeta1) +
				mirrorA_p1[2] * (T(1.0) - T(2.0) * cosBeta1 * cosBeta1);

			T mirrorB_ld[3];
			mirrorB_ld[0] = mirrorA_ld[0];

			mirrorB_ld[1] = mirrorA_ld[1] * (T(1.0) - T(2.0) * sinBeta1 * sinBeta1) +
				mirrorA_ld[2] * (T(-2.0) * cosBeta1 * sinBeta1);

			mirrorB_ld[2] = mirrorA_ld[1] * (T(-2.0) * cosBeta1 * sinBeta1) +
				mirrorA_ld[2] * (T(1.0) - T(2.0) * cosBeta1 * cosBeta1);

			T pv[3];
			ceres::AngleAxisRotatePoint(rvec_g2v, mirrorB_p1, pv);
			pv[0] += tvec_g2v[0];
			pv[1] += tvec_g2v[1];
			pv[2] += tvec_g2v[2];

			T ldv[3];
			ceres::AngleAxisRotatePoint(rvec_g2v, mirrorB_ld, ldv);

			T m[3] = { pv[0] - T(line_[3]), pv[1] - T(line_[4]), pv[2] - T(line_[5]) };
			T n[3] = { T(line_[0]), T(line_[1]), T(line_[2]) };

			residuals[0] = m[0] * n[1] - m[1] * n[0];
			residuals[1] = m[1] * n[2] - m[2] * n[1];
			residuals[2] = m[2] * n[0] - m[0] * n[2];

			residuals[3] = (ldv[0] - n[0]) * T(100.);
			residuals[4] = (ldv[1] - n[1]) * T(100.);
			residuals[5] = (ldv[2] - n[2]) * T(100.);

			//residuals[3] = (ldv[0] - n[0]);
			//residuals[4] = (ldv[1] - n[1]);
			//residuals[5] = (ldv[2] - n[2]);

			residuals[6] = ld[0] * ld[0] + ld[1] * ld[1] + ld[2] * ld[2] - T(1.);
			return true;
		}
	};

	class Optimization
	{
	public:
		Optimization() = default;
		~Optimization() = default;

		static void Optimize(
			const std::map<std::string, cv::Vec6d>& lines,
			cv::Vec3d& normal, double& angle);

		static void OptimizeGalvoX(
			const std::map<int, std::vector<cv::Point2d>>& allDeltaAngles,
			double& alpha0, double& k1, double& k2, double& k3,
			double& k4, double& k5, double& k6, double& k7, double& k8);

		static void OptimizeGalvoY(
			const std::map<int, std::vector<cv::Point2d>>& allDeltaAngles,
			double& beta0, double& k1, double& k2, double& k3,
			double& k4, double& k5, double& k6, double& k7, double& k8);

		static void OptiGalvoRotPhyModel(
			std::vector<lg::SignalLineMat>& slms,
			cv::Mat& G, cv::Mat& R_g2v,
			cv::Vec3d& laserDirection);

		static void OptiGalvoTranPhyModel(
			std::vector<lg::SignalLineMat>& slms,
			cv::Mat& G, cv::Mat& R_g2v, cv::Mat& t_g2v,
			cv::Vec3d& laserDirection, cv::Point3d& laserPoint);

		static void OptiGalvoRotTranPhyModel(
			std::vector<lg::SignalLineMat>& slms,
			cv::Mat& G, cv::Mat& R_g2v, cv::Mat& t_g2v,
			cv::Vec3d& laserDirection, cv::Point3d& laserPoint);

		static void OptiImprovedIdealModel(
			std::vector<lg::SignalLineMat>& slms,
			cv::Mat& G, cv::Mat& R_g2v, cv::Mat& t_g2v,
			cv::Vec3d& laserDirection, cv::Point3d& laserPoint);
	};
}