#pragma once
#ifndef TRACKDLO_H
#define TRACKDLO_H

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/rgbd.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/float64.h>

#include <ctime>
#include <chrono>
#include <thread>
#include <algorithm>
#include <string>

#include <unistd.h>
#include <cstdlib>
#include <signal.h>

using Eigen::MatrixXd;
using cv::Mat;

namespace trackdlo {

template <typename C> struct is_vector : std::false_type {};
template <typename T,typename A> struct is_vector< std::vector<T,A> > : std::true_type {};
template <typename C> inline constexpr bool is_vector_v = is_vector<C>::value;


/**
 * @brief TrackDLO class
 */
class TrackDLO
{
  public:
    TrackDLO();
    TrackDLO(int num_of_nodes);
    // fancy constructor
    TrackDLO(int num_of_nodes,
            double visibility_threshold,
            double beta,
            double lambda,
            double alpha,
            double k_vis,
            double mu,
            int max_iter,
            double tol,
            double beta_pre_proc,
            double lambda_pre_proc,
            double lle_weight);

    double get_sigma2();
    MatrixXd get_tracking_result();
    MatrixXd get_guide_nodes();
    std::vector<MatrixXd> get_correspondence_pairs();
    void initialize_geodesic_coord(std::vector<double> geodesic_coord);
    void initialize_nodes(MatrixXd Y_init);
    void set_sigma2 (double sigma2);

    bool cpd_lle(MatrixXd X_orig,
                  MatrixXd& Y,
                  double& sigma2,
                  double beta,
                  double lambda,
                  double lle_weight,
                  double mu,
                  int max_iter = 30,
                  double tol = 0.0001,
                  bool include_lle = true,
                  std::vector<MatrixXd> correspondence_priors = {},
                  double alpha = 0,
                  std::vector<int> visible_nodes = {},
                  double k_vis = 0,
                  double visibility_threshold = 0.01);

    void tracking_step(MatrixXd X_orig,
                      std::vector<int> visible_nodes,
                      std::vector<int> visible_nodes_extended,
                      MatrixXd proj_matrix,
                      int img_rows,
                      int img_cols);

  private:
    MatrixXd Y_;
    MatrixXd guide_nodes_;
    double sigma2_;
    double beta_;
    double beta_pre_proc_;
    double lambda_;
    double lambda_pre_proc_;
    double alpha_;
    double k_vis_;
    double mu_;
    int max_iter_;
    double tol_;
    double lle_weight_;

    std::vector<double> geodesic_coord_;
    std::vector<MatrixXd> correspondence_priors_;
    double visibility_threshold_;

    std::vector<int> get_nearest_indices(int k, int M, int idx);
    MatrixXd calc_LLE_weights(int k, MatrixXd X);
    std::vector<MatrixXd> traverse_geodesic(std::vector<double> geodesic_coord, const MatrixXd guide_nodes,
                                              const std::vector<int> visible_nodes, int alignment);
    std::vector<MatrixXd> traverse_euclidean (std::vector<double> geodesic_coord, const MatrixXd guide_nodes,
                                              const std::vector<int> visible_nodes, int alignment, int alignment_node_idx = -1);
};

/**
 * @brief TrackDLONode class
 */
class TrackDLONode : public rclcpp::Node
{
  public:
    TrackDLONode(std::shared_ptr<TrackDLO> trackdlo);

    /**
     * @brief Declares and loads a ROS parameter
     *
     * @param name name
     * @param param parameter variable to load into
     * @param description description
     * @param is_required whether failure to load parameter will stop node
     * @param read_only set parameter to read-only
     * @param from_value parameter range minimum
     * @param to_value parameter range maximum
     * @param step_value parameter range step
     * @param additional_constraints additional constraints description
     */
    template <typename T>
    void declare_and_load_parameter(const std::string &name,
                                T &param,
                                const std::string &description,
                                const bool is_required = false,
                                const bool read_only = false,
                                const std::optional<double> &from_value = std::nullopt,
                                const std::optional<double> &to_value = std::nullopt,
                                const std::optional<double> &step_value = std::nullopt,
                                const std::string &additional_constraints = "");

    sensor_msgs::msg::Image::Ptr Callback(const sensor_msgs::msg::Image::ConstPtr& image_msg, const sensor_msgs::msg::Image::ConstPtr& depth_msg);

  private:
    /**
     * @brief Sets up subscribers, publishers, etc. to configure the node
     */
    void setup();

    void update_opencv_mask(const sensor_msgs::msg::Image::ConstPtr& opencv_mask_msg);

    void update_init_nodes(const sensor_msgs::msg::PointCloud2::ConstPtr& pc_msg);

    void update_camera_info(const sensor_msgs::msg::CameraInfo::ConstPtr& cam_msg);

    Mat color_thresholding(Mat cur_image_hsv);

  private:
    std::shared_ptr<TrackDLO> tracker_;
    std::string camera_info_topic_;
    std::string rgb_topic_;
    std::string depth_topic_;
    std::string result_frame_id_;
    std::string hsv_threshold_lower_limit_;
    std::string hsv_threshold_upper_limit_;

    std::vector<int> upper_;
    std::vector<int> lower_;

    MatrixXd Y_;
    double sigma2_;
    bool initialized_;
    bool received_init_nodes_;
    bool received_proj_matrix_;
    MatrixXd init_nodes_;
    std::vector<double> converted_node_coord_;
    Mat occlusion_mask_;
    bool updated_opencv_mask_;
    MatrixXd proj_matrix_;
    bool multi_color_dlo_;

    double beta_;
    double lambda_;
    double alpha_;
    double mu_;
    int max_iter_;
    double tol_;
    double k_vis_;
    double d_vis_;
    double visibility_threshold_;
    int dlo_pixel_width_;
    double beta_pre_proc_;
    double lambda_pre_proc_;
    double lle_weight_;
    double downsample_leaf_size_;

    double pre_proc_total_;
    double algo_total_;
    double pub_data_total_;
    int frames_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr results_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr guide_nodes_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr corr_priors_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr self_occluded_pc_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr result_pc_pub_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr init_nodes_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
};

} // namespace trackdlo

#endif
