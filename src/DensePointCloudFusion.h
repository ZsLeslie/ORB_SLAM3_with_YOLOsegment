#ifndef ORB_SLAM3_DENSEPOINTCLOUDFUSION_H
#define ORB_SLAM3_DENSEPOINTCLOUDFUSION_H

#include <limits>
#include <unordered_map>

#include "PointCloudMapper.h"

struct DensePointSample
{
    Eigen::Vector3f position = Eigen::Vector3f::Zero();
    Eigen::Vector3f normal = Eigen::Vector3f::Zero();
    cv::Vec3b bgr = {0, 0, 0};
    int classId = -1;
};

struct DensePointCloudFrame
{
    std::vector<DensePointSample> samples;
};

// Each instance fuses the observations in a single snapshot of optimized poses.
class DensePointCloudFusion
{
public:
    DensePointCloudFusion(float voxelSize, bool useKalman);
    void AddKeyFrame(const DensePointCloudFrame& frame, const Eigen::Matrix4f& Twc);
    PointCloud::Ptr MakePointCloud() const;

private:
    static constexpr std::size_t NoIndex = std::numeric_limits<std::size_t>::max();

    struct Voxel
    {
        int x, y, z;

        bool operator==(const Voxel& other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct VoxelHash
    {
        std::size_t operator()(const Voxel& key) const;
    };

    struct Surface
    {
        Eigen::Vector3f anchor;
        Eigen::Vector3f position;
        Eigen::Vector3f normal;
        Eigen::Vector3f bgr = Eigen::Vector3f::Zero();
        int classId = -1;
        std::size_t nextInVoxel = NoIndex;
        std::size_t pendingIndex = NoIndex;
        unsigned int keyFrames = 0;
        float normalOffset = 0.0f;
        float variance = 0.0f;
    };

    struct FrameObservation
    {
        std::size_t surfaceIndex;
        Eigen::Vector3f positionSum = Eigen::Vector3f::Zero();
        Eigen::Vector3f bgrSum = Eigen::Vector3f::Zero();
        float varianceSum = 0.0f;
        unsigned int count = 0;
    };

    std::size_t FindSurface(const Voxel& voxel, const Eigen::Vector3f& position,
                            const Eigen::Vector3f& normal, const Eigen::Vector3f& ray,
                            int classId) const;
    std::size_t FindSurfaceInVoxel(const Voxel& voxel, const Eigen::Vector3f& position,
                                   const Eigen::Vector3f& normal, const Eigen::Vector3f& ray,
                                   int classId, float& bestDistanceSquared) const;
    std::size_t AddSurface(const Voxel& voxel, const Eigen::Vector3f& position,
                           const Eigen::Vector3f& normal);
    void UpdateSurface(const FrameObservation& observation);

    const float mVoxelSize;
    const bool mUseKalman;
    std::vector<Surface> mSurfaces;
    std::unordered_map<Voxel, std::size_t, VoxelHash> mVoxels;
    std::vector<FrameObservation> mFrameObservations;
};

#endif
