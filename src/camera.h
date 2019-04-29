#pragma once
#include <Eigen/Dense>

class CameraBase {
public:
    CameraBase(int width_, int height_);
    virtual ~CameraBase();
    virtual Eigen::Vector2d Project(const Eigen::Vector3d& P) const = 0;
    virtual Eigen::Vector3d BackProject(const Eigen::Vector2d& p) const = 0;
    virtual double f() const = 0;

    const int width;
    const int height;
};

class IdealPinhole : public CameraBase {
public:
    IdealPinhole(int width, int height, double fx_, double fy_, double cx_, double cy_);
    virtual ~IdealPinhole();
    virtual Eigen::Vector2d Project(const Eigen::Vector3d& P) const;
    virtual Eigen::Vector3d BackProject(const Eigen::Vector2d& p) const;
    virtual double f() const;

    const double fx;
    const double fy;
    const double cx;
    const double cy;
    const double inv_fx;
    const double inv_fy;
};

class Pinhole : public IdealPinhole {
public:
    Pinhole(int width, int height, double fx, double fy, double cx, double cy,
            double k1_, double k2_, double p1_, double p2_);
    virtual ~Pinhole();
    virtual Eigen::Vector2d Project(const Eigen::Vector3d& P) const;
    virtual Eigen::Vector3d BackProject(const Eigen::Vector2d& p) const;
    Eigen::Vector2d Distortion(const Eigen::Vector2d& p_u) const;

    const double k1;
    const double k2;
    const double p1;
    const double p2;
};
