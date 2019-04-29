#include "camera.h"

CameraBase::CameraBase(int width_, int height_) : width(width_), height(height_) {}
CameraBase::~CameraBase() {}

IdealPinhole::IdealPinhole(int width, int height, double fx_, double fy_, double cx_, double cy_)
    : CameraBase(width, height), fx(fx_), fy(fy_), cx(cx_), cy(cy_), inv_fx(1.0f/fx_), inv_fy(1.0f/fy_) {}
IdealPinhole::~IdealPinhole() {}

Eigen::Vector2d IdealPinhole::Project(const Eigen::Vector3d& P) const {
    Eigen::Vector2d p;
    const double inv_z = 1.0f / P(2);
    p << fx * P(0) * inv_z + cx,
         fy * P(1) * inv_z + cy;
    return p;
}

Eigen::Vector3d IdealPinhole::BackProject(const Eigen::Vector2d& p) const {
    Eigen::Vector3d P;
    P << (p(0) - cx) * inv_fx, (p(1) - cy) * inv_fy, 1;
    return P;
}

double IdealPinhole::f() const { return fx; }

Pinhole::Pinhole(int width, int height, double fx, double fy, double cx, double cy,
                 double k1_, double k2_, double p1_, double p2_)
    : IdealPinhole(width, height, fx, fy, cx, cy), k1(k1_), k2(k2_), p1(p1_), p2(p2_) {}

Pinhole::~Pinhole() {}

Eigen::Vector2d Pinhole::Project(const Eigen::Vector3d& P) const {
    Eigen::Vector2d p, p_u, p_d;
    double inv_z = 1.0f / P(2);
    p_u << P(0) * inv_z, P(1) * inv_z;

    Eigen::Vector2d d_u = Distortion(p_u);
    p_d = p_u + d_u;

    // Apply generalised projection matrix
    p << fx * p_d(0) + cx,
         fy * p_d(1) + cy;

    return p;
}

Eigen::Vector3d Pinhole::BackProject(const Eigen::Vector2d& p) const {
    Eigen::Vector3d P;
    // Lift points to normalised plane
    double mx_d, my_d, mx_u, my_u;

    mx_d = (p(0) - cx) * inv_fx;
    my_d = (p(1) - cy) * inv_fy;
    int n = 8;
    Eigen::Vector2d d_u;
    d_u = Distortion(Eigen::Vector2d(mx_d, my_d));
    // Approximate value
    mx_u = mx_d - d_u(0);
    my_u = my_d - d_u(1);

    for(int i = 0; i < n; ++i) {
        d_u = Distortion(Eigen::Vector2d(mx_u, my_u));
        mx_u = mx_d - d_u(0);
        my_u = my_d - d_u(1);
    }

    P << mx_u, my_u, 1;
    return P;
}

Eigen::Vector2d Pinhole::Distortion(const Eigen::Vector2d& p_u) const {
    Eigen::Vector2d d_u;
    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;
    mx2_u = p_u(0) * p_u(0);
    my2_u = p_u(1) * p_u(1);
    mxy_u = p_u(0) * p_u(1);
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
           p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
    return d_u;
}
