#ifndef ORB_SLAM3_POINTCLOUDMAPPER_H
#define ORB_SLAM3_POINTCLOUDMAPPER_H
#include <iostream>
#include <fstream>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "KeyFrame.h"
using namespace ORB_SLAM3;
using namespace std;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class PointCloudMapper
{
public:
    PointCloudMapper();

    // Submit only after LocalMapping has accepted the keyframe into its map.
    void InsertKeyFrame(KeyFrame* kf, const cv::Mat& imRGB, const cv::Mat& imDepth);

    void run();

    // Create, update and destroy the PCL window on this dedicated thread.
    // Closing the window leaves mapping running; finishing mapping closes it.
    void RunViewer();

    void RequestFinish();

    bool SaveMap(const std::string& filename) const;

    // Published clouds are immutable; readers can retain a snapshot while the
    // mapping thread builds and publishes the next one.
    PointCloud::ConstPtr GetGlobalMap() const;

private:
    struct KeyFrameInput
    {
        KeyFrame* keyFrame = nullptr;
        cv::Mat rgb;
        cv::Mat depth;
    };

    struct KeyFrameCloud
    {
        KeyFrame* keyFrame = nullptr;
        PointCloud::ConstPtr cameraCloud;
    };

    struct CloudPose
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        KeyFrame* keyFrame = nullptr;
        PointCloud::ConstPtr cameraCloud;
        Eigen::Matrix4f Twc;
    };
    using CloudPoses = std::vector<CloudPose, Eigen::aligned_allocator<CloudPose>>;

    PointCloud::Ptr GenerateCameraPointCloud(const KeyFrame* kf,
                                            const cv::Mat& imRGB,
                                            const cv::Mat& imDepth);
    CloudPoses CapturePoses() const;
    static bool SamePoses(const CloudPoses& lhs, const CloudPoses& rhs);
    static PointCloud::Ptr BuildGlobalMap(const CloudPoses& poses);
    void RebuildGlobalMap(bool force);

    std::queue<KeyFrameInput> mqKeyFrames;
    std::mutex mQueueMutex;
    std::condition_variable mQueueUpdated;
    bool mbFinishRequested = false; // Protected by mQueueMutex.
    std::atomic<bool> mbFinished{false};

    // Owned by run(); immutable after mbFinished is published.
    std::vector<KeyFrameCloud> mvKeyFrameClouds;
    CloudPoses mvPublishedPoses;
    std::size_t mRejectedPersonPoints = 0;

    mutable std::mutex mGlobalMapMutex;
    PointCloud::ConstPtr mpGlobalMap;

};

#endif //ORB_SLAM3_POINTCLOUDMAPPER_H
