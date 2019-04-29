#include "factor.h"

ProjectionFactor::ProjectionFactor(const Eigen::Vector3d& pt_, double focal_length)
    : pt(pt_)
{
    sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 1.5;
}

ceres::CostFunction* ProjectionFactor::Create(const Eigen::Vector3d& pt, double focal_length) {
    return new ceres::AutoDiffCostFunction<ProjectionFactor, 2, 7, 3>(
                new ProjectionFactor(pt, focal_length));
}
