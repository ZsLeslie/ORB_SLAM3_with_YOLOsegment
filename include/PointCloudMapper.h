#ifndef ORB_SLAM3_POINTCLOUDMAPPER_H
#define ORB_SLAM3_POINTCLOUDMAPPER_H
#include <iostream>
#include <fstream>
#include <atomic>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <boost/format.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include "KeyFrame.h"
#include <pcl/common/transforms.h>
using namespace ORB_SLAM3;
using namespace std;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class PointCloudMapper
{
public:
    PointCloudMapper();

    void InsertKeyFrame(KeyFrame* kf, cv::Mat& imRGB, cv::Mat& imDepth);

    PointCloud::Ptr GeneratePointCloud(KeyFrame* kf, cv::Mat& imRGB, cv::Mat& imDepth);

    void run();

    void RequestFinish();

    bool SaveMap(const std::string& filename) const;

    queue<KeyFrame*> mqKeyFrame;
    queue<cv::Mat> mqRGB;
    queue<cv::Mat> mqDepth;

    std::mutex mmLoadKFMutex;
    PointCloud::Ptr mpGlobalMap;
    int mKeyFrameSize;
    std::size_t mRejectedPersonPoints{0};

    std::atomic<bool> mbFinishRequested{false};
    std::atomic<bool> mbFinished{false};


};

#endif //ORB_SLAM3_POINTCLOUDMAPPER_H
