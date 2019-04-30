#include "local_parameterization.h"
#include <sophus/se3.hpp>

LocalParameterizationSE3::~LocalParameterizationSE3() {}

// Generalization of the addition operation,
//
//   x_plus_delta = Plus(x, delta)
//
// with the condition that Plus(x, 0) = x.
bool LocalParameterizationSE3::Plus(const double* x, const double* delta, double* x_plus_delta) const {
    Eigen::Map<const Sophus::SE3d> T(x);
    Eigen::Map<const Sophus::Vector6d> delta_(delta);
    Eigen::Map<Sophus::SE3d> T_plus_delta(x_plus_delta);
    T_plus_delta = T * Sophus::SE3d::exp(delta_);
    return true;
}

// The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
//
// jacobian is a row-major GlobalSize() x LocalSize() matrix.
bool LocalParameterizationSE3::ComputeJacobian(const double* x, double* jacobian) const {
    Eigen::Map<const Sophus::SE3d> T(x);
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
    J = T.Dx_this_mul_exp_x_at_0();
    return true;
}

// Size of x.
int LocalParameterizationSE3::GlobalSize() const { return Sophus::SE3d::num_parameters; }

// Size of delta.
int LocalParameterizationSE3::LocalSize() const { return Sophus::SE3d::DoF; }
