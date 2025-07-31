#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <trackdlo/trackdlo.h>
#include <trackdlo/utils.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using cv::Mat;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using std::placeholders::_1;
using std::placeholders::_2;

namespace trackdlo
{
  TrackDLONode::TrackDLONode(std::shared_ptr<TrackDLO> trackdlo) : Node("trackdlo_node"), tracker_(trackdlo)
  {
    // --- Load parameters
    // Camera related parameters
    this->declare_and_load_parameter("camera_info_topic", camera_info_topic_, "TODO", true);
    this->declare_and_load_parameter("rgb_topic", rgb_topic_, "TODO", true);
    this->declare_and_load_parameter("depth_topic", depth_topic_, "TODO", true);
    this->declare_and_load_parameter("result_frame_id", result_frame_id_, "TODO", true);
    // HSV color segmentation parameters
    this->declare_and_load_parameter("hsv_threshold_lower_limit", hsv_threshold_lower_limit_, "TODO", true);
    this->declare_and_load_parameter("hsv_threshold_upper_limit", hsv_threshold_upper_limit_, "TODO", true);
    // TrackDLO parameters
    this->declare_and_load_parameter("beta", beta_, "beta: MCT weight. the larger it is, the more rigid the object becomes", true);
    this->declare_and_load_parameter("lambda", lambda_, "lambda: MCT weight. the larger it is, the more rigid the object becomes", true);
    this->declare_and_load_parameter("alpha", alpha_, "alpha: the alignment strength", true);
    this->declare_and_load_parameter("mu", mu_, "mu: ranges from 0 to 1, large mu indicates the point cloud is noisy", true);
    this->declare_and_load_parameter("max_iter", max_iter_, "max_iter: the maximum number of iterations the EM loop undergoes before termination", true);
    this->declare_and_load_parameter("tol", tol_, "tol: EM optimization convergence tolerance", true);
    this->declare_and_load_parameter("k_vis", k_vis_, "k_vis: the strength of visibility information's effect on membership probability computation", true);
    this->declare_and_load_parameter("d_vis", d_vis_, "d_vis: the max geodesic distance between two adjacent visible nodes for the nodes between them to be considered visible", true);
    this->declare_and_load_parameter("visibility_threshold", visibility_threshold_, "visibility_threshold (tau_vis): the max distance a node can be away from the current point cloud to be considered visible", true);
    this->declare_and_load_parameter("dlo_pixel_width", dlo_pixel_width_, "dlo_pixel_width (w): the approximate dlo width when projected onto 2D", true);
    this->declare_and_load_parameter("beta_pre_proc", beta_pre_proc_, "parameter for the GLTP registration during pre-processing", true);
    this->declare_and_load_parameter("lambda_pre_proc", lambda_pre_proc_, "parameter for the GLTP registration during pre-processing", true);
    this->declare_and_load_parameter("lle_weight", lle_weight_, "parameter for the GLTP registration during pre-processing", true);
    this->declare_and_load_parameter("downsample_leaf_size", downsample_leaf_size_, "parameter for the GLTP registration during pre-processing", true);
  }

  void TrackDLONode::setup()
  {
    // Init some variables
    initialized_ = false;
    received_init_nodes_ = false;
    received_proj_matrix_ = false;
    converted_node_coord_ = { 0.0 };
    updated_opencv_mask_ = false;
    proj_matrix_ = Eigen::MatrixXd(3, 4);
    multi_color_dlo_ = false;
    pre_proc_total_ = 0;
    algo_total_ = 0;
    pub_data_total_ = 0;
    frames_ = 0;

    // update color thresholding lower bound
    std::string rgb_val_lower = "";
    for (int i = 0; i < hsv_threshold_lower_limit_.length(); i++)
    {
      if (hsv_threshold_lower_limit_.substr(i, 1) != " ")
      {
        rgb_val_lower += hsv_threshold_lower_limit_.substr(i, 1);
      }
      else
      {
        lower_.push_back(std::stoi(rgb_val_lower));
        rgb_val_lower = "";
      }

      if (i == hsv_threshold_lower_limit_.length() - 1)
      {
        lower_.push_back(std::stoi(rgb_val_lower));
      }
    }

    // update color thresholding upper bound
    std::string rgb_val_upper = "";
    for (int i = 0; i < hsv_threshold_upper_limit_.length(); i++)
    {
      if (hsv_threshold_upper_limit_.substr(i, 1) != " ")
      {
        rgb_val_upper += hsv_threshold_upper_limit_.substr(i, 1);
      }
      else
      {
        upper_.push_back(std::stoi(rgb_val_upper));
        rgb_val_upper = "";
      }

      if (i == hsv_threshold_upper_limit_.length() - 1)
      {
        upper_.push_back(std::stoi(rgb_val_upper));
      }
    }

    // Subcriptions
    it_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
    opencv_mask_sub_ = it_->subscribe("/mask_with_occlusion", 10, std::bind(&TrackDLONode::update_opencv_mask, this, _1));
    init_nodes_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/trackdlo/init_nodes", 1, std::bind(&TrackDLONode::update_init_nodes, this, _1));
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(camera_info_topic_, 1, std::bind(&TrackDLONode::update_camera_info, this, _1));

    image_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(shared_from_this(), rgb_topic_);
    depth_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(shared_from_this(), depth_topic_);

    // Initialize synchronizer
    sync_ = std::make_shared<message_filters::Synchronizer<sync_policy_>>(sync_policy_(10), *image_sub_, *depth_sub_);

    // Publishers
    int pub_queue_size = 30;
    mask_pub_ = it_->advertise("/trackdlo/mask", pub_queue_size);
    RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", mask_pub_.getTopic().c_str());
    tracking_img_pub_ = it_->advertise("/trackdlo/results_img", pub_queue_size);
    RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", tracking_img_pub_.getTopic().c_str());
    pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/trackdlo/filtered_pointcloud", pub_queue_size);
    RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", pc_pub_->get_topic_name());
    results_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/trackdlo/results_marker", pub_queue_size);
    RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", results_pub_->get_topic_name());
    guide_nodes_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/trackdlo/guide_nodes", pub_queue_size);
    RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", guide_nodes_pub_->get_topic_name());
    corr_priors_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/trackdlo/corr_priors", pub_queue_size);
    RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", corr_priors_pub_->get_topic_name());
    // trackdlo point cloud topic
    result_pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/trackdlo/results_pc", pub_queue_size);
    RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", result_pc_pub_->get_topic_name());
    self_occluded_pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/trackdlo/self_occluded_pc", pub_queue_size);
    RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", self_occluded_pc_pub_->get_topic_name());

    // Callback function for aligned messages
    sync_.get()->registerCallback<std::function<void(
        const sensor_msgs::msg::Image::ConstPtr&,
        const sensor_msgs::msg::Image::ConstPtr&,
        const std::shared_ptr<const message_filters::NullType>,
        const std::shared_ptr<const message_filters::NullType>,
        const std::shared_ptr<const message_filters::NullType>,
        const std::shared_ptr<const message_filters::NullType>,
        const std::shared_ptr<const message_filters::NullType>,
        const std::shared_ptr<const message_filters::NullType>,
        const std::shared_ptr<const message_filters::NullType>)>>(
        [&](const sensor_msgs::msg::Image::ConstPtr& img_msg,
            const sensor_msgs::msg::Image::ConstPtr& depth_msg,
            const std::shared_ptr<const message_filters::NullType> var1,
            const std::shared_ptr<const message_filters::NullType> var2,
            const std::shared_ptr<const message_filters::NullType> var3,
            const std::shared_ptr<const message_filters::NullType> var4,
            const std::shared_ptr<const message_filters::NullType> var5,
            const std::shared_ptr<const message_filters::NullType> var6,
            const std::shared_ptr<const message_filters::NullType> var7)
        {
          sensor_msgs::msg::Image::Ptr tracking_img = Callback(img_msg, depth_msg);
          tracking_img_pub_.publish(tracking_img);
        });
  }

  void TrackDLONode::update_opencv_mask(const sensor_msgs::msg::Image::ConstPtr& opencv_mask_msg)
  {
    occlusion_mask_ = cv_bridge::toCvShare(opencv_mask_msg, "bgr8")->image;  // SAME
    if (!occlusion_mask_.empty())
    {
      updated_opencv_mask_ = true;
    }
  }

  void TrackDLONode::update_init_nodes(const sensor_msgs::msg::PointCloud2::ConstPtr& pc_msg)
  {
    pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;
    pcl_conversions::toPCL(*pc_msg, *cloud);
    pcl::PointCloud<pcl::PointXYZRGB> cloud_xyz;
    pcl::fromPCLPointCloud2(*cloud, cloud_xyz);

    init_nodes_ = cloud_xyz.getMatrixXfMap().topRows(3).transpose().cast<double>();
    received_init_nodes_ = true;
    // init_nodes_sub.shutdown(); TODO
  }

  void TrackDLONode::update_camera_info(const sensor_msgs::msg::CameraInfo::ConstPtr& cam_msg)
  {
    auto P = cam_msg->p;
    for (int i = 0; i < P.size(); i++)
    {
      proj_matrix_(i / 4, i % 4) = P[i];
    }
    received_proj_matrix_ = true;
    // camera_info_sub.shutdown(); TODO
  }

  Mat TrackDLONode::color_thresholding(Mat cur_image_hsv)
  {
    std::vector<int> lower_blue = { 90, 90, 60 };
    std::vector<int> upper_blue = { 130, 255, 255 };

    std::vector<int> lower_red_1 = { 130, 60, 50 };
    std::vector<int> upper_red_1 = { 255, 255, 255 };

    std::vector<int> lower_red_2 = { 0, 60, 50 };
    std::vector<int> upper_red_2 = { 10, 255, 255 };

    std::vector<int> lower_yellow = { 15, 100, 80 };
    std::vector<int> upper_yellow = { 40, 255, 255 };

    Mat mask_blue, mask_red_1, mask_red_2, mask_red, mask_yellow, mask;
    // filter blue
    cv::inRange(cur_image_hsv, cv::Scalar(lower_blue[0], lower_blue[1], lower_blue[2]), cv::Scalar(upper_blue[0], upper_blue[1], upper_blue[2]), mask_blue);

    // filter red
    cv::inRange(cur_image_hsv, cv::Scalar(lower_red_1[0], lower_red_1[1], lower_red_1[2]), cv::Scalar(upper_red_1[0], upper_red_1[1], upper_red_1[2]), mask_red_1);
    cv::inRange(cur_image_hsv, cv::Scalar(lower_red_2[0], lower_red_2[1], lower_red_2[2]), cv::Scalar(upper_red_2[0], upper_red_2[1], upper_red_2[2]), mask_red_2);

    // filter yellow
    cv::inRange(cur_image_hsv, cv::Scalar(lower_yellow[0], lower_yellow[1], lower_yellow[2]), cv::Scalar(upper_yellow[0], upper_yellow[1], upper_yellow[2]), mask_yellow);

    // combine red mask
    cv::bitwise_or(mask_red_1, mask_red_2, mask_red);
    // combine overall mask
    cv::bitwise_or(mask_red, mask_blue, mask);
    cv::bitwise_or(mask_yellow, mask, mask);

    return mask;
  }

  sensor_msgs::msg::Image::Ptr TrackDLONode::Callback(const sensor_msgs::msg::Image::ConstPtr& image_msg, const sensor_msgs::msg::Image::ConstPtr& depth_msg)
  {
    Mat cur_image_orig = cv_bridge::toCvShare(image_msg, "bgr8")->image;
    Mat cur_depth = cv_bridge::toCvShare(depth_msg, depth_msg->encoding)->image;

    // will get overwritten later if intialized
    sensor_msgs::msg::Image::Ptr tracking_img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", cur_image_orig).toImageMsg();

    if (!initialized_)
    {
      std::cout<< "Not initialized!, received_init_nodes is: " <<
       received_init_nodes_ << " and received_proj_matrix is: "
       << received_proj_matrix_ << std::endl;
      if (received_init_nodes_ && received_proj_matrix_)
      {
        tracker_ = std::make_shared<trackdlo::TrackDLO>(init_nodes_.rows(), visibility_threshold_, beta_, lambda_, alpha_, k_vis_, mu_, max_iter_, tol_, beta_pre_proc_, lambda_pre_proc_, lle_weight_);
        sigma2_ = 0.001;

        // record geodesic coord
        double cur_sum = 0;
        for (int i = 0; i < init_nodes_.rows() - 1; i++)
        {
          cur_sum += (init_nodes_.row(i + 1) - init_nodes_.row(i)).norm();
          converted_node_coord_.push_back(cur_sum);
        }

        tracker_->initialize_nodes(init_nodes_);
        tracker_->initialize_geodesic_coord(converted_node_coord_);
        Y_ = init_nodes_.replicate(1, 1);

        initialized_ = true;
      }
    }
    else
    {
      // log time
      std::chrono::high_resolution_clock::time_point cur_time_cb = std::chrono::high_resolution_clock::now();
      double time_diff;
      std::chrono::high_resolution_clock::time_point cur_time;

      Mat mask, mask_rgb, mask_without_occlusion_block;
      Mat cur_image_hsv;

      // convert color
      cv::cvtColor(cur_image_orig, cur_image_hsv, cv::COLOR_BGR2HSV);

      if (!multi_color_dlo_)
      {
        // color_thresholding
        cv::inRange(cur_image_hsv, cv::Scalar(lower_[0], lower_[1], lower_[2]), cv::Scalar(upper_[0], upper_[1], upper_[2]), mask_without_occlusion_block);
      }
      else
      {
        mask_without_occlusion_block = color_thresholding(cur_image_hsv);
      }

      // update cur image for visualization
      Mat cur_image;
      Mat occlusion_mask_gray;
      if (updated_opencv_mask_)
      {
        cv::cvtColor(occlusion_mask_, occlusion_mask_gray, cv::COLOR_BGR2GRAY);
        cv::bitwise_and(mask_without_occlusion_block, occlusion_mask_gray, mask);
        cv::bitwise_and(cur_image_orig, occlusion_mask_, cur_image);
      }
      else
      {
        mask_without_occlusion_block.copyTo(mask);
        cur_image_orig.copyTo(cur_image);
      }

      cv::cvtColor(mask, mask_rgb, cv::COLOR_GRAY2BGR);  // SAME

      bool simulated_occlusion = false;
      int occlusion_corner_i = -1;
      int occlusion_corner_j = -1;
      int occlusion_corner_i_2 = -1;
      int occlusion_corner_j_2 = -1;

      // filter point cloud
      pcl::PointCloud<pcl::PointXYZRGB> cur_pc;
      pcl::PointCloud<pcl::PointXYZRGB> cur_pc_downsampled;

      // filter point cloud from mask
      for (int i = 0; i < mask.rows; i++)
      {
        for (int j = 0; j < mask.cols; j++)
        {
          // for text label (visualization)
          if (updated_opencv_mask_ && !simulated_occlusion && occlusion_mask_gray.at<uchar>(i, j) == 0)
          {
            occlusion_corner_i = i;
            occlusion_corner_j = j;
            simulated_occlusion = true;
          }

          // update the other corner of occlusion mask (visualization)
          if (updated_opencv_mask_ && occlusion_mask_gray.at<uchar>(i, j) == 0)
          {
            occlusion_corner_i_2 = i;
            occlusion_corner_j_2 = j;
          }

          if (mask.at<uchar>(i, j) != 0)
          {
            // point cloud from image pixel coordinates and depth value
            pcl::PointXYZRGB point;
            double pixel_x = static_cast<double>(j);
            double pixel_y = static_cast<double>(i);
            double cx = proj_matrix_(0, 2);
            double cy = proj_matrix_(1, 2);
            double fx = proj_matrix_(0, 0);
            double fy = proj_matrix_(1, 1);
            double pc_z = cur_depth.at<uint16_t>(i, j) / 1000.0;

            point.x = (pixel_x - cx) * pc_z / fx;
            point.y = (pixel_y - cy) * pc_z / fy;
            point.z = pc_z;

            // currently something so color doesn't show up in rviz
            point.r = cur_image_orig.at<cv::Vec3b>(i, j)[0];
            point.g = cur_image_orig.at<cv::Vec3b>(i, j)[1];
            point.b = cur_image_orig.at<cv::Vec3b>(i, j)[2];

            cur_pc.push_back(point);
          }
        }
      }

      // Perform downsampling
      pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloudPtr(cur_pc.makeShared());
      pcl::VoxelGrid<pcl::PointXYZRGB> sor;
      sor.setInputCloud(cloudPtr);
      sor.setLeafSize(downsample_leaf_size_, downsample_leaf_size_, downsample_leaf_size_);
      sor.filter(cur_pc_downsampled);

      MatrixXd X = cur_pc_downsampled.getMatrixXfMap().topRows(3).transpose().cast<double>();
      RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Number of points in downsampled point cloud: " + std::to_string(X.rows()));

      MatrixXd guide_nodes;
      std::vector<MatrixXd> priors;

      // log time
      time_diff = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - cur_time_cb).count() / 1000.0;
      RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Before tracking step: " + std::to_string(time_diff) + " ms");
      pre_proc_total_ += time_diff;
      cur_time = std::chrono::high_resolution_clock::now();

      // calculate node visibility
      // for each node in Y, determine its shortest distance to X
      // for each point in X, determine its shortest distance to Y
      std::map<int, double> shortest_node_pt_dists;
      std::vector<double> shortest_pt_node_dists(X.rows(), 100000.0);
      for (int m = 0; m < Y_.rows(); m++)
      {
        int closest_pt_idx = 0;
        double shortest_dist = 100000;
        // loop through all points in X
        for (int n = 0; n < X.rows(); n++)
        {
          double dist = (Y_.row(m) - X.row(n)).norm();
          // update shortest dist for Y
          if (dist < shortest_dist)
          {
            closest_pt_idx = n;
            shortest_dist = dist;
          }

          // update shortest dist for X
          if (dist < shortest_pt_node_dists[n])
          {
            shortest_pt_node_dists[n] = dist;
          }
        }
        shortest_node_pt_dists.insert(std::pair<int, double>(m, shortest_dist));
      }

      // for current nodes and edges in Y, sort them based on how far away they
      // are from the camera
      std::vector<double> averaged_node_camera_dists = {};
      std::vector<int> indices_vec = {};
      for (int i = 0; i < Y_.rows() - 1; i++)
      {
        averaged_node_camera_dists.push_back(((Y_.row(i) + Y_.row(i + 1)) / 2).norm());
        indices_vec.push_back(i);
      }
      // sort
      std::sort(indices_vec.begin(), indices_vec.end(), [&](const int& a, const int& b) { return (averaged_node_camera_dists[a] < averaged_node_camera_dists[b]); });
      Mat projected_edges = Mat::zeros(mask.rows, mask.cols, CV_8U);

      // project Y^{t-1} onto projected_edges
      MatrixXd Y_h = Y_.replicate(1, 1);
      Y_h.conservativeResize(Y_h.rows(), Y_h.cols() + 1);
      Y_h.col(Y_h.cols() - 1) = MatrixXd::Ones(Y_h.rows(), 1);
      MatrixXd image_coords_mask = (proj_matrix_ * Y_h.transpose()).transpose();

      std::vector<int> visible_nodes = {};
      std::vector<int> self_occluded_nodes = {};
      std::vector<int> not_self_occluded_nodes = {};
      std::vector<int> self_occluding_nodes = {};

      // draw edges closest to the camera first
      for (int idx : indices_vec)
      {
        int col_1 = static_cast<int>(image_coords_mask(idx, 0) / image_coords_mask(idx, 2));
        int row_1 = static_cast<int>(image_coords_mask(idx, 1) / image_coords_mask(idx, 2));

        int col_2 = static_cast<int>(image_coords_mask(idx + 1, 0) / image_coords_mask(idx + 1, 2));
        int row_2 = static_cast<int>(image_coords_mask(idx + 1, 1) / image_coords_mask(idx + 1, 2));

        // only add to visible nodes if did not overlap with existing edges
        if (projected_edges.at<uchar>(row_1, col_1) == 0)
        {
          if (shortest_node_pt_dists[idx] <= visibility_threshold_)
          {
            if (std::find(visible_nodes.begin(), visible_nodes.end(), idx) == visible_nodes.end())
            {
              visible_nodes.push_back(idx);
            }
          }
          if (std::find(not_self_occluded_nodes.begin(), not_self_occluded_nodes.end(), idx) == not_self_occluded_nodes.end())
          {
            not_self_occluded_nodes.push_back(idx);
          }
        }

        // do not consider adjacent nodes directly on top of each other
        if (projected_edges.at<uchar>(row_2, col_2) == 0)
        {
          if (shortest_node_pt_dists[idx + 1] <= visibility_threshold_)
          {
            if (std::find(visible_nodes.begin(), visible_nodes.end(), idx + 1) == visible_nodes.end())
            {
              visible_nodes.push_back(idx + 1);
            }
          }
          if (std::find(not_self_occluded_nodes.begin(), not_self_occluded_nodes.end(), idx + 1) == not_self_occluded_nodes.end())
          {
            not_self_occluded_nodes.push_back(idx + 1);
          }
        }

        // add edges for checking overlap with upcoming nodes
        double x1 = col_1;
        double y1 = row_1;
        double x2 = col_2;
        double y2 = row_2;

        cv::line(projected_edges, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 255),
                 5);  // dlo_pixel_width
      }

      // sort visible nodes to preserve the original connectivity
      std::sort(visible_nodes.begin(), visible_nodes.end());

      // minor mid-section occlusion is usually fine
      // extend visible nodes so that gaps as small as 2 to 3 nodes are filled
      std::vector<int> visible_nodes_extended = {};
      if (!visible_nodes.empty())
      {
        for (int i = 0; i < visible_nodes.size() - 1; i++)
        {
          visible_nodes_extended.push_back(visible_nodes[i]);
          // extend visible nodes
          if (fabs(converted_node_coord_[visible_nodes[i + 1]] - converted_node_coord_[visible_nodes[i]]) <= d_vis_)
          {
            for (int j = 1; j < visible_nodes[i + 1] - visible_nodes[i]; j++)
            {
              visible_nodes_extended.push_back(visible_nodes[i] + j);
            }
          }
        }
        visible_nodes_extended.push_back(visible_nodes[visible_nodes.size() - 1]);
      }

      // store Y_0 for post processing
      MatrixXd Y_0 = Y_.replicate(1, 1);

      // step tracker
      tracker_->tracking_step(X, visible_nodes, visible_nodes_extended, proj_matrix_, mask.rows, mask.cols);
      Y_ = tracker_->get_tracking_result();
      guide_nodes = tracker_->get_guide_nodes();
      priors = tracker_->get_correspondence_pairs();

      // log time
      time_diff = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - cur_time).count() / 1000.0;
      RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Tracking step: " + std::to_string(time_diff) + " ms");
      algo_total_ += time_diff;
      cur_time = std::chrono::high_resolution_clock::now();

      // projection and pub image
      averaged_node_camera_dists = {};
      indices_vec = {};
      for (int i = 0; i < Y_.rows() - 1; i++)
      {
        averaged_node_camera_dists.push_back(((Y_.row(i) + Y_.row(i + 1)) / 2).norm());
        indices_vec.push_back(i);
      }

      // sort
      std::sort(indices_vec.begin(), indices_vec.end(), [&](const int& a, const int& b) { return (averaged_node_camera_dists[a] < averaged_node_camera_dists[b]); });
      std::reverse(indices_vec.begin(), indices_vec.end());

      MatrixXd nodes_h = Y_.replicate(1, 1);
      nodes_h.conservativeResize(nodes_h.rows(), nodes_h.cols() + 1);
      nodes_h.col(nodes_h.cols() - 1) = MatrixXd::Ones(nodes_h.rows(), 1);
      MatrixXd image_coords = (proj_matrix_ * nodes_h.transpose()).transpose();

      Mat tracking_img;
      tracking_img = 0.5 * cur_image_orig + 0.5 * cur_image;

      std::vector<int> vis = visible_nodes;
      // std::vector<int> vis = not_self_occluded_nodes;

      // draw points
      for (int idx : indices_vec)
      {
        int x = static_cast<int>(image_coords(idx, 0) / image_coords(idx, 2));
        int y = static_cast<int>(image_coords(idx, 1) / image_coords(idx, 2));

        cv::Scalar point_color;
        cv::Scalar line_color;

        if (std::find(vis.begin(), vis.end(), idx) != vis.end())
        {
          point_color = cv::Scalar(0, 150, 255);
          line_color = cv::Scalar(0, 255, 0);
        }
        else
        {
          point_color = cv::Scalar(0, 0, 255);

          // line is colored red only when both bounding nodes are not visible
          if (std::find(vis.begin(), vis.end(), idx + 1) == vis.end())
          {
            line_color = cv::Scalar(0, 0, 255);
          }
          else
          {
            line_color = cv::Scalar(0, 255, 0);
          }
        }

        cv::line(tracking_img, cv::Point(x, y), cv::Point(static_cast<int>(image_coords(idx + 1, 0) / image_coords(idx + 1, 2)), static_cast<int>(image_coords(idx + 1, 1) / image_coords(idx + 1, 2))), line_color, 5);

        cv::circle(tracking_img, cv::Point(x, y), 7, point_color, -1);

        if (std::find(vis.begin(), vis.end(), idx + 1) != vis.end())
        {
          point_color = cv::Scalar(0, 150, 255);
        }
        else
        {
          point_color = cv::Scalar(0, 0, 255);
        }
        cv::circle(tracking_img, cv::Point(static_cast<int>(image_coords(idx + 1, 0) / image_coords(idx + 1, 2)), static_cast<int>(image_coords(idx + 1, 1) / image_coords(idx + 1, 2))), 7, point_color, -1);
      }

      // add text
      if (updated_opencv_mask_ && simulated_occlusion)
      {
        cv::putText(tracking_img, "occlusion", cv::Point(occlusion_corner_j, occlusion_corner_i - 10), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(0, 0, 240), 2);
      }

      // publish image
      tracking_img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", tracking_img).toImageMsg();  // SAME

      // publish the results as a marker array
      visualization_msgs::msg::MarkerArray results = MatrixXd2MarkerArray(Y_, result_frame_id_, "node_results", { 1.0, 150.0 / 255.0, 0.0, 1.0 }, { 0.0, 1.0, 0.0, 1.0 }, 0.01, 0.005, vis, { 1.0, 0.0, 0.0, 1.0 }, { 1.0, 0.0, 0.0, 1.0 });
      // visualization_msgs::MarkerArray results = MatrixXd2MarkerArray(Y,
      // result_frame_id, "node_results", {1.0, 150.0/255.0, 0.0, 1.0},
      // {0.0, 1.0, 0.0, 1.0}, 0.01, 0.005);
      visualization_msgs::msg::MarkerArray guide_nodes_results = MatrixXd2MarkerArray(guide_nodes, result_frame_id_, "guide_node_results", { 0.0, 0.0, 0.0, 0.5 }, { 0.0, 0.0, 1.0, 0.5 });
      visualization_msgs::msg::MarkerArray corr_priors_results = MatrixXd2MarkerArray(priors, result_frame_id_, "corr_prior_results", { 0.0, 0.0, 0.0, 0.5 }, { 1.0, 0.0, 0.0, 0.5 });

      // convert to pointcloud2 for eval
      pcl::PointCloud<pcl::PointXYZ> trackdlo_pc;
      for (int i = 0; i < Y_.rows(); i++)
      {
        pcl::PointXYZ temp;
        temp.x = Y_(i, 0);
        temp.y = Y_(i, 1);
        temp.z = Y_(i, 2);
        trackdlo_pc.points.push_back(temp);
      }

      // get self-occluded nodes
      pcl::PointCloud<pcl::PointXYZ> self_occluded_pc;
      for (auto i : self_occluded_nodes)
      {
        pcl::PointXYZ temp;
        temp.x = Y_(i, 0);
        temp.y = Y_(i, 1);
        temp.z = Y_(i, 2);
        self_occluded_pc.points.push_back(temp);
      }

      // publish filtered point cloud
      pcl::PCLPointCloud2 cur_pc_pointcloud2;
      pcl::PCLPointCloud2 result_pc_poincloud2;
      pcl::PCLPointCloud2 self_occluded_pc_poincloud2;
      pcl::toPCLPointCloud2(cur_pc_downsampled, cur_pc_pointcloud2);
      pcl::toPCLPointCloud2(trackdlo_pc, result_pc_poincloud2);
      pcl::toPCLPointCloud2(self_occluded_pc, self_occluded_pc_poincloud2);

      // Convert to ROS data type
      sensor_msgs::msg::PointCloud2 cur_pc_msg;
      sensor_msgs::msg::PointCloud2 result_pc_msg;
      sensor_msgs::msg::PointCloud2 self_occluded_pc_msg;
      pcl_conversions::moveFromPCL(cur_pc_pointcloud2, cur_pc_msg);
      pcl_conversions::moveFromPCL(result_pc_poincloud2, result_pc_msg);
      pcl_conversions::moveFromPCL(self_occluded_pc_poincloud2, self_occluded_pc_msg);

      // for evaluation sync
      cur_pc_msg.header.frame_id = result_frame_id_;
      result_pc_msg.header.frame_id = result_frame_id_;
      result_pc_msg.header.stamp = image_msg->header.stamp;
      self_occluded_pc_msg.header.frame_id = result_frame_id_;
      self_occluded_pc_msg.header.stamp = image_msg->header.stamp;

      results_pub_->publish(results);
      guide_nodes_pub_->publish(guide_nodes_results);
      corr_priors_pub_->publish(corr_priors_results);
      pc_pub_->publish(cur_pc_msg);
      result_pc_pub_->publish(result_pc_msg);
      self_occluded_pc_pub_->publish(self_occluded_pc_msg);

      // reset all guide nodes
      for (int i = 0; i < guide_nodes_results.markers.size(); i++)
      {
        guide_nodes_results.markers[i].action = visualization_msgs::msg::Marker::DELETEALL;
      }
      for (int i = 0; i < corr_priors_results.markers.size(); i++)
      {
        corr_priors_results.markers[i].action = visualization_msgs::msg::Marker::DELETEALL;
      }

      // log time
      time_diff = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - cur_time).count() / 1000.0;
      RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Pub data: " + std::to_string(time_diff) + " ms");
      pub_data_total_ += time_diff;

      frames_ += 1;

      RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Avg before tracking step: " + std::to_string(pre_proc_total_ / frames_) + " ms");
      RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Avg tracking step: " + std::to_string(algo_total_ / frames_) + " ms");
      RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Avg pub data: " + std::to_string(pub_data_total_ / frames_) + " ms");
      RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Avg total: " + std::to_string((pre_proc_total_ + algo_total_ + pub_data_total_) / frames_) + " ms");
    }

    return tracking_img_msg;
  }

  template<typename T>
  void TrackDLONode::declare_and_load_parameter(
      const std::string& name,
      T& param,
      const std::string& description,
      const bool is_required,
      const bool read_only,
      const std::optional<double>& from_value,
      const std::optional<double>& to_value,
      const std::optional<double>& step_value,
      const std::string& additional_constraints)
  {
    rcl_interfaces::msg::ParameterDescriptor param_desc;
    param_desc.description = description;
    param_desc.additional_constraints = additional_constraints;
    param_desc.read_only = read_only;

    auto type = rclcpp::ParameterValue(param).get_type();

    if (from_value.has_value() && to_value.has_value())
    {
      if constexpr (std::is_integral_v<T>)
      {
        rcl_interfaces::msg::IntegerRange range;
        T step = static_cast<T>(step_value.has_value() ? step_value.value() : 1);
        range.set__from_value(static_cast<T>(from_value.value())).set__to_value(static_cast<T>(to_value.value())).set__step(step);
        param_desc.integer_range = { range };
      }
      else if constexpr (std::is_floating_point_v<T>)
      {
        rcl_interfaces::msg::FloatingPointRange range;
        T step = static_cast<T>(step_value.has_value() ? step_value.value() : 1.0);
        range.set__from_value(static_cast<T>(from_value.value())).set__to_value(static_cast<T>(to_value.value())).set__step(step);
        param_desc.floating_point_range = { range };
      }
      else
      {
        RCLCPP_WARN(
            this->get_logger(),
            "Parameter type of parameter '%s' does not support specifying a "
            "range",
            name.c_str());
      }
    }

    this->declare_parameter(name, type, param_desc);

    try
    {
      param = this->get_parameter(name).get_value<T>();
      std::stringstream ss;
      ss << "Loaded parameter '" << name << "': ";
      if constexpr (is_vector_v<T>)
      {
        ss << "[";
        for (const auto& element : param)
          ss << element << (&element != &param.back() ? ", " : "");
        ss << "]";
      }
      else
      {
        ss << param;
      }
      RCLCPP_INFO_STREAM(this->get_logger(), ss.str());
    }
    catch (rclcpp::exceptions::ParameterUninitializedException&)
    {
      if (is_required)
      {
        RCLCPP_FATAL_STREAM(this->get_logger(), "Missing required parameter '" << name << "', exiting");
        exit(EXIT_FAILURE);
      }
      else
      {
        std::stringstream ss;
        ss << "Missing parameter '" << name << "', using default value: ";
        if constexpr (is_vector_v<T>)
        {
          ss << "[";
          for (const auto& element : param)
            ss << element << (&element != &param.back() ? ", " : "");
          ss << "]";
        }
        else
        {
          ss << param;
        }
        RCLCPP_WARN_STREAM(this->get_logger(), ss.str());
        this->set_parameters({ rclcpp::Parameter(name, rclcpp::ParameterValue(param)) });
      }
    }
  }

  TrackDLO::TrackDLO()
  {
  }

  TrackDLO::TrackDLO(int num_of_nodes)
  {
    // default initialize
    Y_ = MatrixXd::Zero(num_of_nodes, 3);
    guide_nodes_ = Y_.replicate(1, 1);
    sigma2_ = 0.0;
    beta_ = 5.0;
    beta_pre_proc_ = 3.0;
    lambda_ = 1.0;
    lambda_pre_proc_ = 1.0;
    alpha_ = 0.0;
    lle_weight_ = 1.0;
    k_vis_ = 0.0;
    mu_ = 0.05;
    max_iter_ = 50;
    tol_ = 0.00001;
    geodesic_coord_ = {};
    correspondence_priors_ = {};
    visibility_threshold_ = 0.02;
  }

  TrackDLO::TrackDLO(int num_of_nodes, double visibility_threshold, double beta, double lambda, double alpha, double k_vis, double mu, int max_iter, double tol, double beta_pre_proc, double lambda_pre_proc, double lle_weight)
  {
    Y_ = MatrixXd::Zero(num_of_nodes, 3);
    visibility_threshold_ = visibility_threshold;
    guide_nodes_ = Y_.replicate(1, 1);
    sigma2_ = 0.0;
    beta_ = beta;
    beta_pre_proc_ = beta_pre_proc;
    lambda_ = lambda;
    lambda_pre_proc_ = lambda_pre_proc;
    alpha_ = alpha;
    lle_weight_ = lle_weight;
    k_vis_ = k_vis;
    mu_ = mu;
    max_iter_ = max_iter;
    tol_ = tol;
    geodesic_coord_ = {};
    correspondence_priors_ = {};
  }

  double TrackDLO::get_sigma2()
  {
    return sigma2_;
  }

  MatrixXd TrackDLO::get_tracking_result()
  {
    return Y_;
  }

  MatrixXd TrackDLO::get_guide_nodes()
  {
    return guide_nodes_;
  }

  std::vector<MatrixXd> TrackDLO::get_correspondence_pairs()
  {
    return correspondence_priors_;
  }

  void TrackDLO::initialize_geodesic_coord(std::vector<double> geodesic_coord)
  {
    for (size_t i = 0; i < geodesic_coord.size(); i++)
    {
      geodesic_coord_.push_back(geodesic_coord[i]);
    }
  }

  void TrackDLO::initialize_nodes(MatrixXd Y_init)
  {
    Y_ = Y_init.replicate(1, 1);
    guide_nodes_ = Y_init.replicate(1, 1);
  }

  void TrackDLO::set_sigma2(double sigma2)
  {
    sigma2_ = sigma2;
  }

  std::vector<int> TrackDLO::get_nearest_indices(int k, int M, int idx)
  {
    std::vector<int> indices_arr;
    if (idx - k < 0)
    {
      for (int i = 0; i <= idx + k; i++)
      {
        if (i != idx)
        {
          indices_arr.push_back(i);
        }
      }
    }
    else if (idx + k >= M)
    {
      for (int i = idx - k; i <= M - 1; i++)
      {
        if (i != idx)
        {
          indices_arr.push_back(i);
        }
      }
    }
    else
    {
      for (int i = idx - k; i <= idx + k; i++)
      {
        if (i != idx)
        {
          indices_arr.push_back(i);
        }
      }
    }

    return indices_arr;
  }

  MatrixXd TrackDLO::calc_LLE_weights(int k, MatrixXd X)
  {
    MatrixXd W = MatrixXd::Zero(X.rows(), X.rows());
    for (Eigen::Index i = 0; i < X.rows(); i++)
    {
      std::vector<int> indices = get_nearest_indices(static_cast<int>(k / 2), X.rows(), i);
      MatrixXd xi = X.row(i);
      MatrixXd Xi = MatrixXd(indices.size(), X.cols());

      // fill in Xi: Xi = X[indices, :]
      for (size_t r = 0; r < indices.size(); r++)
      {
        Xi.row(r) = X.row(indices[r]);
      }

      // component = np.full((len(Xi), len(xi)), xi).T - Xi.T
      MatrixXd component = xi.replicate(Xi.rows(), 1).transpose() - Xi.transpose();
      MatrixXd Gi = component.transpose() * component;
      MatrixXd Gi_inv;

      if (Gi.determinant() != 0)
      {
        Gi_inv = Gi.inverse();
      }
      else
      {
        // std::cout << "Gi singular at entry " << i << std::endl;
        double epsilon = 0.00001;
        Gi.diagonal().array() += epsilon;
        Gi_inv = Gi.inverse();
      }

      // wi = Gi_inv * 1 / (1^T * Gi_inv * 1)
      MatrixXd ones_row_vec = MatrixXd::Constant(1, Xi.rows(), 1.0);
      MatrixXd ones_col_vec = MatrixXd::Constant(Xi.rows(), 1, 1.0);

      MatrixXd wi = (Gi_inv * ones_col_vec) / (ones_row_vec * Gi_inv * ones_col_vec).value();
      MatrixXd wi_T = wi.transpose();

      for (size_t c = 0; c < indices.size(); c++)
      {
        W(i, indices[c]) = wi_T(c);
      }
    }

    return W;
  }

  bool TrackDLO::
      cpd_lle(MatrixXd X_orig, MatrixXd& Y, double& sigma2, double beta, double lambda, double lle_weight, double mu, int max_iter, double tol, bool include_lle, std::vector<MatrixXd> correspondence_priors, double alpha, std::vector<int> visible_nodes, double k_vis, double visibility_threshold)
  {
    // prune X
    MatrixXd X_temp = MatrixXd::Zero(X_orig.rows(), 3);
    int valid_pt_counter = 0;
    for (Eigen::Index i = 0; i < X_orig.rows(); i++)
    {
      // find shortest distance between this point and any node
      double shortest_dist = 100000;
      for (int j = 0; j < Y.rows(); j++)
      {
        double dist = (Y.row(j) - X_orig.row(i)).norm();
        if (dist < shortest_dist)
        {
          shortest_dist = dist;
        }
      }
      // require a point to be sufficiently close to the node set to be valid
      if (shortest_dist < 0.1)
      {
        X_temp.row(valid_pt_counter) = X_orig.row(i);
        valid_pt_counter += 1;
      }
    }
    MatrixXd X = X_temp.topRows(valid_pt_counter);

    bool converged = true;

    int M = Y.rows();
    int N = X.rows();
    int D = 3;

    MatrixXd Y_0 = Y.replicate(1, 1);

    MatrixXd diff_yy = MatrixXd::Zero(M, M);
    MatrixXd diff_yy_sqrt = MatrixXd::Zero(M, M);
    for (Eigen::Index i = 0; i < M; i++)
    {
      for (int j = 0; j < M; j++)
      {
        diff_yy(i, j) = (Y_0.row(i) - Y_0.row(j)).squaredNorm();
        diff_yy_sqrt(i, j) = (Y_0.row(i) - Y_0.row(j)).norm();
      }
    }

    MatrixXd converted_node_dis = MatrixXd::Zero(M, M);  // this is a M*M matrix in place of diff_sqrt
    MatrixXd converted_node_dis_sq = MatrixXd::Zero(M, M);
    std::vector<double> converted_node_coord = { 0.0 };  // this is not squared

    MatrixXd G = MatrixXd::Zero(M, M);
    double cur_sum = 0;
    for (Eigen::Index i = 0; i < M - 1; i++)
    {
      cur_sum += pt2pt_dis(Y_0.row(i + 1), Y_0.row(i));
      converted_node_coord.push_back(cur_sum);
    }

    for (size_t i = 0; i < converted_node_coord.size(); i++)
    {
      for (size_t j = 0; j < converted_node_coord.size(); j++)
      {
        converted_node_dis_sq(i, j) = pow(converted_node_coord[i] - converted_node_coord[j], 2);
        converted_node_dis(i, j) = abs(converted_node_coord[i] - converted_node_coord[j]);
      }
    }

    // kernel matrix
    G = 1 / (2 * beta * 2 * beta) * (-sqrt(2) * converted_node_dis / beta).array().exp() * (2 * converted_node_dis.array() + sqrt(2) * beta);

    // get the LLE matrix
    MatrixXd L = calc_LLE_weights(6, Y_0);
    MatrixXd H = (MatrixXd::Identity(M, M) - L).transpose() * (MatrixXd::Identity(M, M) - L);

    // construct J
    MatrixXd J = MatrixXd::Zero(M, M);
    MatrixXd Y_extended = Y_0.replicate(1, 1);
    if (correspondence_priors.size() != 0)
    {
      int num_of_correspondence_priors = correspondence_priors.size();

      for (int i = 0; i < num_of_correspondence_priors; i++)
      {
        MatrixXd temp = MatrixXd::Zero(1, 3);
        int index = correspondence_priors[i](0, 0);
        temp(0, 0) = correspondence_priors[i](0, 1);
        temp(0, 1) = correspondence_priors[i](0, 2);
        temp(0, 2) = correspondence_priors[i](0, 3);

        J.row(index) = MatrixXd::Identity(M, M).row(index);
        Y_extended.row(index) = temp;

        // // enforce boundaries
        // if (i == 0 || i == num_of_correspondence_priors-1) {
        //     J.row(index) *= 5;
        // }
      }
    }

    // diff_xy should be a (M * N) matrix
    MatrixXd diff_xy = MatrixXd::Zero(M, N);
    for (Eigen::Index i = 0; i < M; i++)
    {
      for (int j = 0; j < N; j++)
      {
        diff_xy(i, j) = (Y_0.row(i) - X.row(j)).squaredNorm();
      }
    }

    // initialize sigma2
    if (sigma2 == 0)
    {
      sigma2 = diff_xy.sum() / static_cast<double>(D * M * N);
    }

    for (int it = 0; it < max_iter; it++)
    {
      // update diff_xy
      std::map<int, double> shortest_node_pt_dists;
      for (int m = 0; m < M; m++)
      {
        // for each node in Y, determine a point in X closest to it
        // for P_vis calculations
        double shortest_dist = 10000.0;
        for (int n = 0; n < N; n++)
        {
          diff_xy(m, n) = (Y.row(m) - X.row(n)).squaredNorm();
          double dist = (Y.row(m) - X.row(n)).norm();
          if (dist < shortest_dist)
          {
            shortest_dist = dist;
          }
        }
        // if close enough to X, the node is visible
        if (shortest_dist <= visibility_threshold)
        {
          shortest_dist = 0;
        }
        // push back the pair
        shortest_node_pt_dists.insert(std::pair<int, double>(m, shortest_dist));
      }

      MatrixXd P = (-0.5 * diff_xy / sigma2).array().exp();
      MatrixXd P_stored = P.replicate(1, 1);
      double c = pow((2 * M_PI * sigma2), static_cast<double>(D) / 2) * mu / (1 - mu) * static_cast<double>(M) / N;
      P = P.array().rowwise() / (P.colwise().sum().array() + c);

      // P matrix calculation based on geodesic distance
      std::vector<int> max_p_nodes(P.cols(), 0);
      MatrixXd pts_dis_sq_geodesic = MatrixXd::Zero(M, N);

      // loop through all points
      for (int i = 0; i < N; i++)
      {
        P.col(i).maxCoeff(&max_p_nodes[i]);
        int max_p_node = max_p_nodes[i];

        int potential_2nd_max_p_node_1 = max_p_node - 1;
        if (potential_2nd_max_p_node_1 == -1)
        {
          potential_2nd_max_p_node_1 = 2;
        }

        int potential_2nd_max_p_node_2 = max_p_node + 1;
        if (potential_2nd_max_p_node_2 == M)
        {
          potential_2nd_max_p_node_2 = M - 3;
        }

        int next_max_p_node;
        if (pt2pt_dis(Y.row(potential_2nd_max_p_node_1), X.row(i)) < pt2pt_dis(Y.row(potential_2nd_max_p_node_2), X.row(i)))
        {
          next_max_p_node = potential_2nd_max_p_node_1;
        }
        else
        {
          next_max_p_node = potential_2nd_max_p_node_2;
        }

        // fill the current column of pts_dis_sq_geodesic
        pts_dis_sq_geodesic(max_p_node, i) = pt2pt_dis_sq(Y.row(max_p_node), X.row(i));
        pts_dis_sq_geodesic(next_max_p_node, i) = pt2pt_dis_sq(Y.row(next_max_p_node), X.row(i));

        if (max_p_node < next_max_p_node)
        {
          for (int j = 0; j < max_p_node; j++)
          {
            pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[max_p_node]) + pt2pt_dis(Y.row(max_p_node), X.row(i)), 2);
          }
          for (int j = next_max_p_node; j < M; j++)
          {
            pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[next_max_p_node]) + pt2pt_dis(Y.row(next_max_p_node), X.row(i)), 2);
          }
        }
        else
        {
          for (int j = 0; j < next_max_p_node; j++)
          {
            pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[next_max_p_node]) + pt2pt_dis(Y.row(next_max_p_node), X.row(i)), 2);
          }
          for (int j = max_p_node; j < M; j++)
          {
            pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[max_p_node]) + pt2pt_dis(Y.row(max_p_node), X.row(i)), 2);
          }
        }
      }

      // update P
      P = (-0.5 * pts_dis_sq_geodesic / sigma2).array().exp();

      // modified membership probability (adapted from cdcpd)
      if (visible_nodes.size() != Y.rows() && !visible_nodes.empty() && k_vis != 0)
      {
        MatrixXd P_vis = MatrixXd::Ones(P.rows(), P.cols());
        double total_P_vis = 0;

        for (Eigen::Index i = 0; i < Y.rows(); i++)
        {
          double shortest_node_pt_dist = shortest_node_pt_dists[i];

          double P_vis_i = exp(-k_vis * shortest_node_pt_dist);
          total_P_vis += P_vis_i;

          P_vis.row(i) = P_vis_i * P_vis.row(i);
        }

        // normalize P_vis
        P_vis = P_vis / total_P_vis;

        // modify P
        P = P.cwiseProduct(P_vis);

        // modify c
        c = pow((2 * M_PI * sigma2), static_cast<double>(D) / 2) * mu / (1 - mu) / N;
        P = P.array().rowwise() / (P.colwise().sum().array() + c);
      }
      else
      {
        P = P.array().rowwise() / (P.colwise().sum().array() + c);
      }

      MatrixXd Pt1 = P.colwise().sum();
      MatrixXd P1 = P.rowwise().sum();
      double Np = P1.sum();
      MatrixXd PX = P * X;

      // M step
      MatrixXd A_matrix;
      MatrixXd B_matrix;
      if (include_lle)
      {
        if (correspondence_priors.size() != 0)
        {
          A_matrix = P1.asDiagonal() * G + lambda * sigma2 * MatrixXd::Identity(M, M) + sigma2 * lle_weight * H * G + alpha * J * G;
          B_matrix = PX - P1.asDiagonal() * Y_0 - sigma2 * lle_weight * H * Y_0 + alpha * (Y_extended - Y_0);
        }
        else
        {
          A_matrix = P1.asDiagonal() * G + lambda * sigma2 * MatrixXd::Identity(M, M) + sigma2 * lle_weight * H * G;
          B_matrix = PX - P1.asDiagonal() * Y_0 - sigma2 * lle_weight * H * Y_0;
        }
      }
      else
      {
        if (correspondence_priors.size() != 0)
        {
          A_matrix = P1.asDiagonal() * G + lambda * sigma2 * MatrixXd::Identity(M, M) + alpha * J * G;
          B_matrix = PX - P1.asDiagonal() * Y_0 + alpha * (Y_extended - Y_0);
        }
        else
        {
          A_matrix = P1.asDiagonal() * G + lambda * sigma2 * MatrixXd::Identity(M, M);
          B_matrix = PX - P1.asDiagonal() * Y_0;
        }
      }

      MatrixXd W = A_matrix.completeOrthogonalDecomposition().solve(B_matrix);

      MatrixXd T = Y_0 + G * W;
      double trXtdPt1X = (X.transpose() * Pt1.asDiagonal() * X).trace();
      double trPXtT = (PX.transpose() * T).trace();
      double trTtdP1T = (T.transpose() * P1.asDiagonal() * T).trace();

      sigma2 = (trXtdPt1X - 2 * trPXtT + trTtdP1T) / (Np * D);

      if (pt2pt_dis(Y, Y_0 + G * W) / Y.rows() < tol)
      {
        Y = Y_0 + G * W;
        RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Iteration until convergence: " + std::to_string(it + 1));
        break;
      }
      else
      {
        Y = Y_0 + G * W;
      }

      if (it == max_iter - 1)
      {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "optimization did not converge!");
        converged = false;
        break;
      }
    }

    return converged;
  }

  // alignment: 0 --> align with head; 1 --> align with tail
  std::vector<MatrixXd> TrackDLO::traverse_geodesic(std::vector<double> geodesic_coord, const MatrixXd guide_nodes, const std::vector<int> visible_nodes, int alignment)
  {
    std::vector<MatrixXd> node_pairs = {};

    // extreme cases: only one guide node available
    // since this function will only be called when at least one of head or tail
    // is visible, the only node will be head or tail
    if (guide_nodes.rows() == 1)
    {
      MatrixXd node_pair(1, 4);
      node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
      node_pairs.push_back(node_pair);
      return node_pairs;
    }

    double guide_nodes_total_dist = 0;
    double total_seg_dist = 0;

    if (alignment == 0)
    {
      // push back the first pair
      MatrixXd node_pair(1, 4);
      node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
      node_pairs.push_back(node_pair);

      // initialize iterators
      int guide_nodes_it = 0;
      int seg_dist_it = 0;
      int last_seg_dist_it = seg_dist_it;

      // ultimate terminating condition: run out of guide nodes to use. two
      // conditions that can trigger this:
      //   1. next visible node index - current visible node index > 1
      //   2. currenting using the last two guide nodes
      while (visible_nodes[guide_nodes_it + 1] - visible_nodes[guide_nodes_it] == 1 && guide_nodes_it + 1 <= guide_nodes.rows() - 1 && seg_dist_it + 1 <= geodesic_coord.size() - 1)
      {
        guide_nodes_total_dist += pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it + 1));
        // now keep adding segment dists until the total seg dists exceed the
        // current total guide node dists
        while (guide_nodes_total_dist > total_seg_dist)
        {
          // break condition
          if (seg_dist_it == geodesic_coord.size() - 1)
          {
            break;
          }

          total_seg_dist += fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it + 1]);
          if (total_seg_dist <= guide_nodes_total_dist)
          {
            seg_dist_it += 1;
          }
          else
          {
            total_seg_dist -= fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it + 1]);
            break;
          }
        }
        // additional break condition
        if (seg_dist_it == geodesic_coord.size() - 1)
        {
          break;
        }
        // upon exit, seg_dist_it will be at the locaiton where the total seg
        // dist is barely smaller than guide nodes total dist the node desired
        // should be in between guide_nodes[guide_nodes_it] and
        // guide_node[guide_nodes_it + 1] seg_dist_it will also be within
        // guide_nodes_it and guide_nodes_it + 1
        if (guide_nodes_it == 0 && seg_dist_it == 0)
        {
          continue;
        }
        // if one guide nodes segment is not long enough
        if (last_seg_dist_it == seg_dist_it)
        {
          guide_nodes_it += 1;
          continue;
        }
        double remaining_dist = total_seg_dist - (guide_nodes_total_dist - pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it + 1)));
        MatrixXd temp = (guide_nodes.row(guide_nodes_it + 1) - guide_nodes.row(guide_nodes_it)) * remaining_dist / pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it + 1));
        node_pair(0, 0) = seg_dist_it;
        node_pair(0, 1) = temp(0, 0) + guide_nodes(guide_nodes_it, 0);
        node_pair(0, 2) = temp(0, 1) + guide_nodes(guide_nodes_it, 1);
        node_pair(0, 3) = temp(0, 2) + guide_nodes(guide_nodes_it, 2);
        node_pairs.push_back(node_pair);

        // update guide_nodes_it at the very end
        guide_nodes_it += 1;
        last_seg_dist_it = seg_dist_it;
      }
    }
    else
    {
      // push back the first pair
      MatrixXd node_pair(1, 4);
      node_pair << visible_nodes.back(), guide_nodes(guide_nodes.rows() - 1, 0), guide_nodes(guide_nodes.rows() - 1, 1), guide_nodes(guide_nodes.rows() - 1, 2);
      node_pairs.push_back(node_pair);

      // initialize iterators
      int guide_nodes_it = guide_nodes.rows() - 1;
      int seg_dist_it = geodesic_coord.size() - 1;
      int last_seg_dist_it = seg_dist_it;

      // ultimate terminating condition: run out of guide nodes to use. two
      // conditions that can trigger this:
      //   1. next visible node index - current visible node index > 1
      //   2. currenting using the last two guide nodes
      while (visible_nodes[guide_nodes_it] - visible_nodes[guide_nodes_it - 1] == 1 && guide_nodes_it - 1 >= 0 && seg_dist_it - 1 >= 0)
      {
        guide_nodes_total_dist += pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it - 1));
        // now keep adding segment dists until the total seg dists exceed the
        // current total guide node dists
        while (guide_nodes_total_dist > total_seg_dist)
        {
          // break condition
          if (seg_dist_it == 0)
          {
            break;
          }

          total_seg_dist += fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it - 1]);
          if (total_seg_dist <= guide_nodes_total_dist)
          {
            seg_dist_it -= 1;
          }
          else
          {
            total_seg_dist -= fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it - 1]);
            break;
          }
        }
        // additional break condition
        if (seg_dist_it == 0)
        {
          break;
        }
        // upon exit, seg_dist_it will be at the locaiton where the total seg
        // dist is barely smaller than guide nodes total dist the node desired
        // should be in between guide_nodes[guide_nodes_it] and
        // guide_node[guide_nodes_it + 1] seg_dist_it will also be within
        // guide_nodes_it and guide_nodes_it + 1
        if (guide_nodes_it == 0 && seg_dist_it == 0)
        {
          continue;
        }
        // if one guide nodes segment is not long enough
        if (last_seg_dist_it == seg_dist_it)
        {
          guide_nodes_it -= 1;
          continue;
        }
        double remaining_dist = total_seg_dist - (guide_nodes_total_dist - pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it - 1)));
        MatrixXd temp = (guide_nodes.row(guide_nodes_it - 1) - guide_nodes.row(guide_nodes_it)) * remaining_dist / pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it - 1));
        node_pair(0, 0) = seg_dist_it;
        node_pair(0, 1) = temp(0, 0) + guide_nodes(guide_nodes_it, 0);
        node_pair(0, 2) = temp(0, 1) + guide_nodes(guide_nodes_it, 1);
        node_pair(0, 3) = temp(0, 2) + guide_nodes(guide_nodes_it, 2);
        node_pairs.insert(node_pairs.begin(), node_pair);

        // update guide_nodes_it at the very end
        guide_nodes_it -= 1;
        last_seg_dist_it = seg_dist_it;
      }
    }

    return node_pairs;
  }

  std::vector<MatrixXd> TrackDLO::traverse_euclidean(std::vector<double> geodesic_coord, const MatrixXd guide_nodes, const std::vector<int> visible_nodes, int alignment, int alignment_node_idx)
  {
    std::vector<MatrixXd> node_pairs = {};

    // extreme cases: only one guide node available
    // since this function will only be called when at least one of head or tail
    // is visible, the only node will be head or tail
    if (guide_nodes.rows() == 1)
    {
      MatrixXd node_pair(1, 4);
      node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
      node_pairs.push_back(node_pair);
      return node_pairs;
    }

    if (alignment == 0)
    {
      // push back the first pair
      MatrixXd node_pair(1, 4);
      node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
      node_pairs.push_back(node_pair);

      std::vector<int> consecutive_visible_nodes = {};
      for (size_t i = 0; i < visible_nodes.size(); i++)
      {
        if (i == visible_nodes[i])
        {
          consecutive_visible_nodes.push_back(i);
        }
        else
        {
          break;
        }
      }

      int last_found_index = 0;
      int seg_dist_it = 0;
      MatrixXd cur_center = guide_nodes.row(0);

      // basically pure pursuit
      while (last_found_index + 1 <= consecutive_visible_nodes.size() - 1 && seg_dist_it + 1 <= geodesic_coord.size() - 1)
      {
        double look_ahead_dist = fabs(geodesic_coord[seg_dist_it + 1] - geodesic_coord[seg_dist_it]);
        bool found_intersection = false;
        std::vector<double> intersection = {};

        for (size_t i = last_found_index; i + 1 <= consecutive_visible_nodes.size() - 1; i++)
        {
          std::vector<MatrixXd> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i + 1), cur_center, look_ahead_dist);

          // if no intersection found
          if (intersections.size() == 0)
          {
            continue;
          }
          else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i + 1)) > pt2pt_dis(cur_center, guide_nodes.row(i + 1)))
          {
            continue;
          }
          else
          {
            found_intersection = true;
            last_found_index = i;

            if (intersections.size() == 2)
            {
              if (pt2pt_dis(intersections[0], guide_nodes.row(i + 1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i + 1)))
              {
                // the first solution is closer
                intersection = { intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2) };
                cur_center = intersections[0];
              }
              else
              {
                // the second one is closer
                intersection = { intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2) };
                cur_center = intersections[1];
              }
            }
            else
            {
              intersection = { intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2) };
              cur_center = intersections[0];
            }
            break;
          }
        }

        if (!found_intersection)
        {
          break;
        }
        else
        {
          MatrixXd temp = MatrixXd::Zero(1, 4);
          temp(0, 0) = seg_dist_it + 1;
          temp(0, 1) = intersection[0];
          temp(0, 2) = intersection[1];
          temp(0, 3) = intersection[2];
          node_pairs.push_back(temp);

          seg_dist_it += 1;
        }
      }
    }
    else if (alignment == 1)
    {
      // push back the first pair
      MatrixXd node_pair(1, 4);
      node_pair << visible_nodes.back(), guide_nodes(guide_nodes.rows() - 1, 0), guide_nodes(guide_nodes.rows() - 1, 1), guide_nodes(guide_nodes.rows() - 1, 2);
      node_pairs.push_back(node_pair);

      std::vector<int> consecutive_visible_nodes = {};
      for (size_t i = 1; i <= visible_nodes.size(); i++)
      {
        if (visible_nodes[visible_nodes.size() - i] == geodesic_coord.size() - i)
        {
          consecutive_visible_nodes.push_back(geodesic_coord.size() - i);
        }
        else
        {
          break;
        }
      }

      int last_found_index = guide_nodes.rows() - 1;
      int seg_dist_it = geodesic_coord.size() - 1;
      MatrixXd cur_center = guide_nodes.row(guide_nodes.rows() - 1);

      // basically pure pursuit
      while (last_found_index - 1 >= (guide_nodes.rows() - consecutive_visible_nodes.size()) && seg_dist_it - 1 >= 0)
      {
        double look_ahead_dist = fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it - 1]);

        bool found_intersection = false;
        std::vector<double> intersection = {};

        for (int i = last_found_index; i >= (guide_nodes.rows() - consecutive_visible_nodes.size() + 1); i--)
        {
          std::vector<MatrixXd> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i - 1), cur_center, look_ahead_dist);

          // if no intersection found
          if (intersections.size() == 0)
          {
            continue;
          }
          else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i - 1)) > pt2pt_dis(cur_center, guide_nodes.row(i - 1)))
          {
            continue;
          }
          else
          {
            found_intersection = true;
            last_found_index = i;

            if (intersections.size() == 2)
            {
              if (pt2pt_dis(intersections[0], guide_nodes.row(i - 1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i - 1)))
              {
                // the first solution is closer
                intersection = { intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2) };
                cur_center = intersections[0];
              }
              else
              {
                // the second one is closer
                intersection = { intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2) };
                cur_center = intersections[1];
              }
            }
            else
            {
              intersection = { intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2) };
              cur_center = intersections[0];
            }
            break;
          }
        }

        if (!found_intersection)
        {
          break;
        }
        else
        {
          MatrixXd temp = MatrixXd::Zero(1, 4);
          temp(0, 0) = seg_dist_it - 1;
          temp(0, 1) = intersection[0];
          temp(0, 2) = intersection[1];
          temp(0, 3) = intersection[2];
          node_pairs.push_back(temp);

          seg_dist_it -= 1;
        }
      }
    }
    else
    {
      // push back the first pair
      MatrixXd node_pair(1, 4);
      node_pair << visible_nodes[alignment_node_idx], guide_nodes(alignment_node_idx, 0), guide_nodes(alignment_node_idx, 1), guide_nodes(alignment_node_idx, 2);
      node_pairs.push_back(node_pair);

      std::vector<int> consecutive_visible_nodes_2 = { visible_nodes[alignment_node_idx] };
      for (size_t i = alignment_node_idx + 1; i < visible_nodes.size(); i++)
      {
        if (visible_nodes[i] - visible_nodes[i - 1] == 1)
        {
          consecutive_visible_nodes_2.push_back(visible_nodes[i]);
        }
        else
        {
          break;
        }
      }

      // traverse from the alignment node to the tail node
      int last_found_index = alignment_node_idx;
      int seg_dist_it = visible_nodes[alignment_node_idx];
      MatrixXd cur_center = guide_nodes.row(alignment_node_idx);

      // basically pure pursuit
      while (last_found_index + 1 <= alignment_node_idx + consecutive_visible_nodes_2.size() - 1 && seg_dist_it + 1 <= geodesic_coord.size() - 1)
      {
        double look_ahead_dist = fabs(geodesic_coord[seg_dist_it + 1] - geodesic_coord[seg_dist_it]);
        bool found_intersection = false;
        std::vector<double> intersection = {};

        for (size_t i = last_found_index; i + 1 <= alignment_node_idx + consecutive_visible_nodes_2.size() - 1; i++)
        {
          std::vector<MatrixXd> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i + 1), cur_center, look_ahead_dist);

          // if no intersection found
          if (intersections.size() == 0)
          {
            continue;
          }
          else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i + 1)) > pt2pt_dis(cur_center, guide_nodes.row(i + 1)))
          {
            continue;
          }
          else
          {
            found_intersection = true;
            last_found_index = i;

            if (intersections.size() == 2)
            {
              if (pt2pt_dis(intersections[0], guide_nodes.row(i + 1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i + 1)))
              {
                // the first solution is closer
                intersection = { intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2) };
                cur_center = intersections[0];
              }
              else
              {
                // the second one is closer
                intersection = { intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2) };
                cur_center = intersections[1];
              }
            }
            else
            {
              intersection = { intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2) };
              cur_center = intersections[0];
            }
            break;
          }
        }

        if (!found_intersection)
        {
          break;
        }
        else
        {
          MatrixXd temp = MatrixXd::Zero(1, 4);
          temp(0, 0) = seg_dist_it + 1;
          temp(0, 1) = intersection[0];
          temp(0, 2) = intersection[1];
          temp(0, 3) = intersection[2];
          node_pairs.push_back(temp);

          seg_dist_it += 1;
        }
      }

      // traverse from alignment node to head node
      std::vector<int> consecutive_visible_nodes_1 = { visible_nodes[alignment_node_idx] };
      for (size_t i = alignment_node_idx - 1; i >= 0; i++)
      {
        if (visible_nodes[i + 1] - visible_nodes[i] == 1)
        {
          consecutive_visible_nodes_1.push_back(visible_nodes[i]);
        }
        else
        {
          break;
        }
      }

      last_found_index = alignment_node_idx;
      seg_dist_it = visible_nodes[alignment_node_idx];
      cur_center = guide_nodes.row(alignment_node_idx);

      // basically pure pursuit
      while (last_found_index - 1 >= alignment_node_idx - consecutive_visible_nodes_1.size() && seg_dist_it - 1 >= 0)
      {
        double look_ahead_dist = fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it - 1]);
        bool found_intersection = false;
        std::vector<double> intersection = {};

        for (size_t i = last_found_index; i - 1 >= 0; i--)
        {
          std::vector<MatrixXd> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i - 1), cur_center, look_ahead_dist);

          // if no intersection found
          if (intersections.size() == 0)
          {
            continue;
          }
          else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i - 1)) > pt2pt_dis(cur_center, guide_nodes.row(i - 1)))
          {
            continue;
          }
          else
          {
            found_intersection = true;
            last_found_index = i;

            if (intersections.size() == 2)
            {
              if (pt2pt_dis(intersections[0], guide_nodes.row(i - 1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i - 1)))
              {
                // the first solution is closer
                intersection = { intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2) };
                cur_center = intersections[0];
              }
              else
              {
                // the second one is closer
                intersection = { intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2) };
                cur_center = intersections[1];
              }
            }
            else
            {
              intersection = { intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2) };
              cur_center = intersections[0];
            }
            break;
          }
        }

        if (!found_intersection)
        {
          break;
        }
        else
        {
          MatrixXd temp = MatrixXd::Zero(1, 4);
          temp(0, 0) = seg_dist_it - 1;
          temp(0, 1) = intersection[0];
          temp(0, 2) = intersection[1];
          temp(0, 3) = intersection[2];
          node_pairs.push_back(temp);

          seg_dist_it -= 1;
        }
      }
    }

    return node_pairs;
  }

  void TrackDLO::tracking_step(MatrixXd X_orig, std::vector<int> visible_nodes, std::vector<int> visible_nodes_extended, MatrixXd proj_matrix, int img_rows, int img_cols)
  {
    // variable initialization
    correspondence_priors_ = {};
    int state = 0;

    // copy visible nodes vec to guide nodes
    // not using topRows() because it caused weird bugs
    guide_nodes_ = MatrixXd::Zero(visible_nodes_extended.size(), 3);
    if (visible_nodes_extended.size() != Y_.rows())
    {
      for (size_t i = 0; i < visible_nodes_extended.size(); i++)
      {
        guide_nodes_.row(i) = Y_.row(visible_nodes_extended[i]);
      }
    }
    else
    {
      guide_nodes_ = Y_.replicate(1, 1);
    }

    // determine DLO state: heading visible, tail visible, both visible, or both
    // occluded priors_vec should be the final output; priors_vec[i] = {index,
    // x, y, z}
    double sigma2_pre_proc = sigma2_;
    // pre-processing registration
    cpd_lle(X_orig, guide_nodes_, sigma2_pre_proc, beta_pre_proc_, lambda_pre_proc_, lle_weight_, mu_, max_iter_, tol_, true);

    if (visible_nodes_extended.size() == Y_.rows())
    {
      if (visible_nodes.size() == visible_nodes_extended.size())
      {
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "All nodes visible");
      }
      else
      {
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Minor occlusion");
      }

      // remap visible node locations
      std::vector<MatrixXd> priors_vec_1 = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes_extended, 0);
      std::vector<MatrixXd> priors_vec_2 = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes_extended, 1);

      // priors vec 2 goes from last index -> first index
      std::reverse(priors_vec_2.begin(), priors_vec_2.end());

      // take average
      correspondence_priors_ = {};
      for (size_t i = 0; i < Y_.rows(); i++)
      {
        if (i < priors_vec_2[0](0, 0) && i < priors_vec_1.size())
        {
          correspondence_priors_.push_back(priors_vec_1[i]);
        }
        else if (i > priors_vec_1[priors_vec_1.size() - 1](0, 0) && (i - (Y_.rows() - priors_vec_2.size())) < priors_vec_2.size())
        {
          correspondence_priors_.push_back(priors_vec_2[i - (Y_.rows() - priors_vec_2.size())]);
        }
        else
        {
          correspondence_priors_.push_back((priors_vec_1[i] + priors_vec_2[i - (Y_.rows() - priors_vec_2.size())]) / 2.0);
        }
      }
    }
    else if (visible_nodes_extended[0] == 0 && visible_nodes_extended[visible_nodes_extended.size() - 1] == Y_.rows() - 1)
    {
      RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Mid-section occluded");

      correspondence_priors_ = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes_extended, 0);
      std::vector<MatrixXd> priors_vec_2 = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes_extended, 1);
      // priors_vec = traverse_geodesic(geodesic_coord, guide_nodes,
      // visible_nodes, 0); std::vector<MatrixXd> priors_vec_2 =
      // traverse_geodesic(geodesic_coord, guide_nodes, visible_nodes, 1);

      correspondence_priors_.insert(correspondence_priors_.end(), priors_vec_2.begin(), priors_vec_2.end());
    }
    else if (visible_nodes_extended[0] == 0)
    {
      RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Tail occluded");

      correspondence_priors_ = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes_extended, 0);
      // priors_vec = traverse_geodesic(geodesic_coord, guide_nodes,
      // visible_nodes, 0);
    }
    else if (visible_nodes_extended[visible_nodes_extended.size() - 1] == Y_.rows() - 1)
    {
      RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Head occluded");

      correspondence_priors_ = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes_extended, 1);
      // priors_vec = traverse_geodesic(geodesic_coord, guide_nodes,
      // visible_nodes, 1);
    }
    else
    {
      RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Both ends occluded");

      // determine which node moved the least
      int alignment_node_idx = -1;
      double moved_dist = 999999;
      for (size_t i = 0; i < visible_nodes.size(); i++)
      {
        if (pt2pt_dis(Y_.row(visible_nodes[i]), guide_nodes_.row(i)) < moved_dist)
        {
          moved_dist = pt2pt_dis(Y_.row(visible_nodes[i]), guide_nodes_.row(i));
          alignment_node_idx = i;
        }
      }

      // std::cout << "alignment node index: " << alignment_node_idx <<
      // std::endl;
      correspondence_priors_ = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes_extended, 2, alignment_node_idx);
    }

    // include_lle == false because we have no space to discuss it in the paper
    cpd_lle(X_orig, Y_, sigma2_, beta_, lambda_, lle_weight_, mu_, max_iter_, tol_, false, correspondence_priors_, alpha_, visible_nodes_extended, k_vis_, visibility_threshold_);
  }
}  // namespace trackdlo

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  std::shared_ptr<trackdlo::TrackDLO> trackdlo = std::make_shared<trackdlo::TrackDLO>(50);
  auto node = std::make_shared<trackdlo::TrackDLONode>(trackdlo);
  // Set up subscribers, publishers, etc. to configure the node
  node->setup();
  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}