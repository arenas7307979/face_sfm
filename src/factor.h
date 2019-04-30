#pragma once
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <ceres/autodiff_cost_function.h>

class ProjectionFactor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ProjectionFactor(const Eigen::Vector3d& pt_, double focal_length);

    template<class T>
    bool operator()(const T* Twc_raw, const T* x3Dw_raw, T* residuals_raw) const{
        Eigen::Map<const Sophus::SE3<T>> Twc(Twc_raw);
        Eigen::Map<const Sophus::Vector3<T>> x3Dw(x3Dw_raw);
        Eigen::Map<Sophus::Vector2<T>> residuals(residuals_raw);

        Sophus::Vector3<T> x3Dc = Twc.inverse() * x3Dw;
        T inv_z = T(1.0f) / x3Dc(2);
        residuals = sqrt_info.cast<T>() * (x3Dc.template head<2>() * inv_z - pt.head<2>().cast<T>());
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& pt, double focal_length);

    Eigen::Matrix2d sqrt_info;
    Eigen::Vector3d pt;
};
