#ifndef ORB_SLAM3_POINTCLOUDCHANGEDETECTOR_H
#define ORB_SLAM3_POINTCLOUDCHANGEDETECTOR_H

#include <cstddef>
#include <memory>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class PointCloudChangeDetector
{
public:
    using Cloud = pcl::PointCloud<pcl::PointXYZRGB>;

    PointCloudChangeDetector();
    ~PointCloudChangeDetector();

    const Cloud& GetCloud() const;
    void AppendChanges(const Cloud& current);
    void Truncate(std::size_t pointCount);

private:
    struct State;
    std::unique_ptr<State> mState;
};

#endif
