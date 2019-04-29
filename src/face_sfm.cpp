#include "face_sfm.h"
#include <opencv2/opencv.hpp>
#include <rosbag/view.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <cv_bridge/cv_bridge.h>
#include <boost/foreach.hpp>
#include <ceres/ceres.h>
#include "ceres/local_parameterization_se3.h"
#include <sophus/se3.hpp>
#include <sensor_msgs/PointCloud.h>
#include <thread>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "ceres/factor.h"
#include "camera.h"
#define foreach BOOST_FOREACH


class DataMgr{
public:

    void ReadParam(ros::NodeHandle& nh) {
        nh.getParam("fx", fx);
        nh.getParam("fy", fy);
        nh.getParam("cx", cx);
        nh.getParam("cy", cy);
        nh.getParam("width", width);
        nh.getParam("height", height);
        nh.getParam("k1", k1);
        nh.getParam("k2", k2);
        nh.getParam("p1", p1);
        nh.getParam("p2", p2);
        pub_face = nh.advertise<sensor_msgs::PointCloud>("/face", 1000);
    }


    void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                          Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
    {

        Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
        design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
        design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
        design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
        design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);



        Eigen::Vector4d triangulated_point;
        triangulated_point =
                design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

        //get depth from cam of C0 ( C0 = WORLD)
        point_3d(0) = triangulated_point(0) / triangulated_point(3);
        point_3d(1) = triangulated_point(1) / triangulated_point(3);
        point_3d(2) = triangulated_point(2) / triangulated_point(3);
    }

    bool solveRelativeRT(std::vector<cv::Point2f> &ll, std::vector<cv::Point2f> rr, Eigen::Matrix3d &Rotation, Eigen::Vector3d &Translation)
    {
#if 1
        cv::Mat mask;
        cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat rot, trans;
        int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);
        //cout << "inlier_cnt " << inlier_cnt << endl;

        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        for (int i = 0; i < 3; i++)
        {
            T(i) = trans.at<double>(i, 0);
            for (int j = 0; j < 3; j++)
                R(i, j) = rot.at<double>(i, j);
        }

        Rotation = R;
        Translation = T; //Tcw
        if(inlier_cnt > 12)
            return true;
        else
            return false;
#endif
    }


    void callback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::PointCloud2ConstPtr & points)
    {

        cv_bridge::CvImagePtr cv_ptr;
        std::vector<cv::Point2f> landmarks(points->width);
        cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
        cv::Mat tmp = cv_ptr->image;

        for(int i = 0; i < landmarks.size(); ++i){
            //8 is cycle for points: x1y1 , x2y2
            memcpy(&landmarks[i].x, points->data.data() + 8*i, sizeof(float));
            memcpy(&landmarks[i].y, points->data.data() + 4 + 8*i, sizeof(float));
            //cv::circle(tmp, landmarks[i], 3, cv::Scalar(255), 3);
        }
        images.emplace_back(tmp);
        face_pts.emplace_back(landmarks);
        //cv::imshow("tmp", tmp);
        //cv::waitKey(2);

        // Solve all of perception here...
    };

    int findInitFrame(){
        int id = -1;
        int max_distance = -1;
        for(int i = 0; i < face_pts.size(); i++){
            int sum = 0;
            for(int j = 0; j < face_pts[i].size(); j++){
                sum += cv::norm(face_pts[0][j] - face_pts[i][j]);
                if(sum > max_distance)
                {
                    max_distance = sum;
                    id = i;
                }
            }
        }
        return id;
    };

    void process(){
        //std::cout << "images / facePts sz = " << images.size() << " / " << face_pts.size() <<std::endl;

        //Instrinc / distoration parameter
        cv::Mat K =(cv::Mat_<double>(3,3)<<fx, 0, cx, 0, fy, cy, 0, 0, 1);
        cv::Mat_<double> Distort(1,4); Distort <<k1, k2, p1 , p2;
        un_pts_im.resize(face_pts.size());
        un_pts_n.resize(face_pts.size());
        frame_pose.resize(face_pts.size());
        un_pt2f_n.resize(face_pts.size());
        un_pts_n_3dim.resize(face_pts.size());
        //set Twc0
        Eigen::Matrix3d I;
        I.setIdentity();
        Sophus::SE3d Twc0(I, Eigen::Vector3d(0,0,0));
        frame_pose[0] = Twc0;

        //get undistorted point
        for(int i = 0; i < images.size(); i++){
            cv::undistortPoints(face_pts[i], un_pts_im[i], K, Distort);
            for(int j=0; j < un_pts_im[i].size(); j++){
                un_pts_n[i].push_back(Eigen::Vector2d(un_pts_im[i][j].x, un_pts_im[i][j].y));
                un_pt2f_n[i].push_back(cv::Point2f(un_pts_im[i][j].x, un_pts_im[i][j].y));
                un_pts_n_3dim[i].push_back(Eigen::Vector3d(un_pts_im[i][j].x, un_pts_im[i][j].y, 1));
                un_pts_im[i][j].x = un_pts_im[i][j].x * fx + cx;
                un_pts_im[i][j].y = un_pts_im[i][j].y * fy + cy;
            }
            //std::cout << face_pts[i][0] << " / " << un_pts_n[i][0] <<std::endl;
        }

        int f_id = findInitFrame();

        //Estimate init Pose
        Eigen::Matrix3d Rotation; Eigen::Vector3d Translation;
        bool result = solveRelativeRT(un_pt2f_n[0], un_pt2f_n[f_id], Rotation, Translation);
        Sophus::SE3d Tci_c0(Rotation, Translation);
        //Twci
        frame_pose[f_id]  = Twc0 * Tci_c0.inverse();

        //std::cout << "f_id=" << f_id <<std::endl;
        //std::cout << "Twc0 = " << frame_pose[f_id].matrix() <<std::endl;
        int fail = 0;

        //Triangulate 2 frame to estimate 3d points
        {
            Eigen::Matrix<double, 3, 4> P;
            Eigen::Matrix<double, 3, 4> P_I;
            P_I << frame_pose[0].inverse().rotationMatrix(), frame_pose[0].inverse().translation();
            P << frame_pose[f_id].inverse().rotationMatrix(), frame_pose[f_id].inverse().translation();
            for(int j=0; j < un_pts_n[0].size(); j++){
                Eigen::Vector3d Xw, Xc0;
                triangulatePoint(P_I, P, un_pts_n[0][j], un_pts_n[f_id][j], Xw); //get depth in camera 0 (WORLD)

                if(1)
                    Xw.z() = 2;
                //Xc0 = un_pts_n_3dim[0][j] * 5.0;
                //Xw = frame_pose[0] * Xc0;

                x3Dw.push_back(cv::Point3d(Xw.x(), Xw.y(), Xw.z()));
                x3Dw_eigen.push_back(Xw);
            }
        }
        //slove pnp
        cv::Mat_<double> Dis_0(1,4);
        cv::Mat K0 =(cv::Mat_<double>(3,3)<<1, 0, 0, 0, 1, 0, 0, 0, 1);
        Dis_0 << 0, 0, 0, 0;

        for(int k = 0; k < images.size(); k++){
            cv::Mat rvec, tvec;

            //Tcw
            cv::solvePnPRansac(x3Dw, un_pt2f_n[k], K0, Dis_0, rvec, tvec, false, 100, 8.0 / fx, 0.99);
            cv::Mat R;
            cv::Rodrigues(rvec, R);

            Eigen::Matrix3d Rcw;
            Eigen::Vector3d tcw;
            cv::cv2eigen(R, Rcw);
            cv::cv2eigen(tvec, tcw);
            Sophus::SE3d Tckw(Rcw, tcw);
            frame_pose[k] = Tckw.inverse();
        }

        //global BA
        ceres::Problem problem;
        ceres::LossFunction* loss_function = new::ceres::CauchyLoss(1);
        ceres::LocalParameterization* local_para_se3 = new LocalParameterizationSE3();
        double *para_pose = new double[frame_pose.size() * 7];
        double *para_x3Dw = new double[x3Dw_eigen.size() * 3];

        for(int id = 0; id < frame_pose.size(); id++){
            std::memcpy(para_pose + 7 * id, frame_pose[id].data(), sizeof(double) *7);
            problem.AddParameterBlock(para_pose + 7 * id, 7, local_para_se3);
            if(id == 0){
                problem.SetParameterBlockConstant(para_pose); //para_pose + 7 * id = para_pose ;(id==0)
            }
        }



        for(int i = 0; i < x3Dw_eigen.size(); i++){
            std::memcpy(para_x3Dw + 3 * i, x3Dw_eigen[i].data(), sizeof(double) * 3);
        }

        for(int i = 0; i < x3Dw.size(); i++){
            for(int obs = 0; obs < frame_pose.size(); obs++){
                auto factor = ProjectionFactor::Create(un_pts_n_3dim[obs][i], fx);
                problem.AddResidualBlock(factor, loss_function, para_pose + 7 * obs, para_x3Dw + 3 * i);
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.max_num_iterations = 50;
        options.num_threads = 6;
        ceres::Solver::Summary summary;
        LOG(WARNING) << "Solve...";
        ceres::Solve(options, &problem, &summary);

        LOG(INFO) << summary.FullReport() << std::endl;

        for(int i = 0, n = frame_pose.size(); i < n; ++i)
            std::memcpy(frame_pose[i].data(), para_pose + 7 * i, sizeof(double) * 7);

        for(int i = 0, n = x3Dw_eigen.size(); i < n; ++i){
            std::cout << "x3Dw_eigen[i] bf = " << x3Dw_eigen[i] <<std::endl;
            std::memcpy(x3Dw_eigen[i].data(), para_x3Dw + 3 * i, sizeof(double) * 3);
            std::cout << "x3Dw_eigen[i] af = " << x3Dw_eigen[i] <<std::endl;
        }
        // release
        delete [] para_pose;
        delete [] para_x3Dw;

        //show 3d point
        while(1) {
            sensor_msgs::PointCloud face_msg;
            face_msg.header.frame_id = "world";

            for(auto& lm : x3Dw_eigen) {
                geometry_msgs::Point32 pt;
                pt.x = lm.x();
                pt.y = lm.y();
                pt.z = lm.z();
                face_msg.points.emplace_back(pt);
            }

            pub_face.publish(face_msg);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    std::vector<cv::Mat> images;
    std::vector<std::vector<cv::Point2f>> face_pts;
    //undistorted point in image
    std::vector<std::vector<cv::Point2f>> un_pts_im;
    //undistorted point in normal plane
    std::vector<std::vector<Eigen::Vector2d>> un_pts_n;
    std::vector<std::vector<cv::Point2f>> un_pt2f_n;
    std::vector<std::vector<Eigen::Vector3d>> un_pts_n_3dim;
    //pose in slide window Twc
    std::vector<Sophus::SE3d> frame_pose;
    std::vector<cv::Point3d> x3Dw;
    std::vector<Eigen::Vector3d> x3Dw_eigen;
    double fx, fy, cx, cy, k1, k2, p1, p2;
    int width, height;
    ros::Publisher pub_face;
    std::shared_ptr<CameraBase> camera;
};

template <class M>
class BagSubscriber : public message_filters::SimpleFilter<M>
{
public:
    void newMessage(const boost::shared_ptr<M const> &msg)
    {
        this->signalMessage(msg);
    }
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "face_landmark_sfm_node");
    ros::NodeHandle nh("~");
    rosbag::Bag bag;
    DataMgr data_mgr;
    bag.open(argv[1], rosbag::bagmode::Read);
    data_mgr.ReadParam(nh);
    std::vector<std::string> topics;
    topics.push_back(std::string("/image_raw"));
    topics.push_back(std::string("/landmark"));
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    BagSubscriber<sensor_msgs::Image> image_sub;
    BagSubscriber<sensor_msgs::PointCloud2> pl_sub;
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::PointCloud2>
            sync(image_sub, pl_sub, 10);
    sync.registerCallback(boost::bind(&DataMgr::callback, &data_mgr, _1, _2));

    foreach(rosbag::MessageInstance const m, view)
    {
        sensor_msgs::Image::ConstPtr s = m.instantiate<sensor_msgs::Image>();
        if (s != NULL){
            image_sub.newMessage(s);
        }

        sensor_msgs::PointCloud2ConstPtr i = m.instantiate<sensor_msgs::PointCloud2>();
        if (i != NULL)
        {
            pl_sub.newMessage(i);
        }
    }

    bag.close();
    data_mgr.process();
    return  0;
}