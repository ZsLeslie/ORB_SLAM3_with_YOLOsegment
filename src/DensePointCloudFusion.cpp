#include "DensePointCloudFusion.h"

#include <cmath>
#include <cstdint>

namespace
{
constexpr float kCosMaxNormalAngle = 0.8660254f; // 30 degrees.
constexpr float kMinimumRayIncidence = 0.1f;
constexpr float kRegistrationSigma = 0.01f; // Metres along the surface normal.

float MeasurementVariance(float depth, float axialProjection)
{
    // Approximate Kinect axial noise model. The registration term also limits
    // the confidence of close measurements when BA covariance isn't available.
    const float relativeDepth = depth - 0.4f;
    const float axialSigma = 0.0012f + 0.0019f * relativeDepth * relativeDepth;
    const float normalSigma = axialSigma * axialProjection;
    return normalSigma * normalSigma + kRegistrationSigma * kRegistrationSigma;
}
}

constexpr std::size_t DensePointCloudFusion::NoIndex;

DensePointCloudFusion::DensePointCloudFusion(float voxelSize, bool useKalman)
    : mVoxelSize(voxelSize), mUseKalman(useKalman)
{
}

std::size_t DensePointCloudFusion::VoxelHash::operator()(const Voxel& key) const
{
    return (static_cast<std::uint32_t>(key.x) * 73856093u) ^
           (static_cast<std::uint32_t>(key.y) * 19349663u) ^
           (static_cast<std::uint32_t>(key.z) * 83492791u);
}

std::size_t DensePointCloudFusion::FindSurfaceInVoxel(
    const Voxel& voxel, const Eigen::Vector3f& position, const Eigen::Vector3f& normal,
    const Eigen::Vector3f& ray, int classId, float& bestDistanceSquared) const
{
    const auto bucket = mVoxels.find(voxel);
    if (bucket == mVoxels.end())
        return NoIndex;

    const bool hasNormal = normal.squaredNorm() > 0.0f;
    const float planeTolerance = 0.30f * mVoxelSize;
    const float depthTolerance = 1.5f * mVoxelSize;
    const float unorientedTolerance = 0.25f * mVoxelSize;
    std::size_t match = NoIndex;
    for (std::size_t index = bucket->second; index != NoIndex; index = mSurfaces[index].nextInVoxel)
    {
        const Surface& surface = mSurfaces[index];
        if (classId >= 0 && surface.classId >= 0 && surface.classId != classId)
            continue;

        const Eigen::Vector3f residual = position - surface.position;
        const float distanceSquared = residual.squaredNorm();
        if (distanceSquared >= bestDistanceSquared ||
            (position - surface.anchor).squaredNorm() >= mVoxelSize * mVoxelSize)
            continue;

        const bool surfaceHasNormal = surface.normal.squaredNorm() > 0.0f;
        if (hasNormal && surfaceHasNormal)
        {
            if (normal.dot(surface.normal) < kCosMaxNormalAngle)
                continue;
            const float planeError = std::abs(surface.normal.dot(residual));
            // ray.z is one in the camera frame, so this projection tests the
            // predicted camera depth along the observation ray as well as the
            // plane distance. A back-facing or grazing surface cannot match.
            const float incidence = -surface.normal.dot(ray);
            if (incidence <= kMinimumRayIncidence || planeError > planeTolerance ||
                planeError > incidence * depthTolerance)
                continue;
        }
        else if (hasNormal != surfaceHasNormal ||
                 distanceSquared >= unorientedTolerance * unorientedTolerance)
            continue;

        bestDistanceSquared = distanceSquared;
        match = index;
    }
    return match;
}

std::size_t DensePointCloudFusion::FindSurface(
    const Voxel& voxel, const Eigen::Vector3f& position, const Eigen::Vector3f& normal,
    const Eigen::Vector3f& ray, int classId) const
{
    const float radius = normal.squaredNorm() > 0.0f ? mVoxelSize : 0.25f * mVoxelSize;
    float bestDistanceSquared = radius * radius;
    const std::size_t sameVoxel = FindSurfaceInVoxel(voxel, position, normal, ray, classId,
                                                       bestDistanceSquared);
    if (sameVoxel != NoIndex)
        return sameVoxel;

    // Search just across the voxel boundaries within reach of the sample.
    const Eigen::Array3f scaled = position.array() / mVoxelSize;
    const Eigen::Array3f fractional = scaled - scaled.floor();
    const float reach = radius / mVoxelSize;
    const Eigen::Array3i lower = (fractional < reach).cast<int>();
    const Eigen::Array3i upper = (fractional > 1.0f - reach).cast<int>();
    std::size_t match = NoIndex;
    for (int z = -lower.z(); z <= upper.z(); ++z)
        for (int y = -lower.y(); y <= upper.y(); ++y)
            for (int x = -lower.x(); x <= upper.x(); ++x)
            {
                if (x == 0 && y == 0 && z == 0)
                    continue;
                const Voxel neighbor{voxel.x + x, voxel.y + y, voxel.z + z};
                const std::size_t candidate = FindSurfaceInVoxel(neighbor, position, normal, ray,
                                                                  classId, bestDistanceSquared);
                if (candidate != NoIndex)
                    match = candidate;
            }
    return match;
}

std::size_t DensePointCloudFusion::AddSurface(const Voxel& voxel, const Eigen::Vector3f& position,
                                             const Eigen::Vector3f& normal)
{
    const std::size_t index = mSurfaces.size();
    auto bucket = mVoxels.emplace(voxel, NoIndex).first;
    Surface surface;
    surface.anchor = position;
    surface.position = position;
    surface.normal = normal;
    surface.nextInVoxel = bucket->second;
    mSurfaces.push_back(surface);
    bucket->second = index;
    return index;
}

void DensePointCloudFusion::AddKeyFrame(const DensePointCloudFrame& frame, const Eigen::Matrix4f& Twc)
{
    const Eigen::Matrix3f rotation = Twc.topLeftCorner<3, 3>();
    const Eigen::Vector3f translation = Twc.topRightCorner<3, 1>();
    mFrameObservations.clear();
    for (const DensePointSample& sample : frame.samples)
    {
        const Eigen::Vector3f position = rotation * sample.position + translation;
        const Eigen::Array3f scaled = position.array() / mVoxelSize;
        // Leave room for neighbor offsets and avoid invalid float-to-int casts.
        if (!(scaled.abs() < 1.0e9f).all())
            continue;
        const Eigen::Array3i cell = scaled.floor().cast<int>();
        const Voxel voxel{cell.x(), cell.y(), cell.z()};
        const Eigen::Vector3f normal = rotation * sample.normal;
        const Eigen::Vector3f ray = rotation * (sample.position / sample.position.z());
        std::size_t index = FindSurface(voxel, position, normal, ray, sample.classId);
        if (index == NoIndex)
            index = AddSurface(voxel, position, normal);

        Surface& surface = mSurfaces[index];
        const Eigen::Vector3f color(sample.bgr[0], sample.bgr[1], sample.bgr[2]);
        if (surface.classId < 0 && sample.classId >= 0)
        {
            surface.classId = sample.classId;
            surface.bgr = color;
        }
        if (surface.pendingIndex == NoIndex)
        {
            surface.pendingIndex = mFrameObservations.size();
            mFrameObservations.push_back(FrameObservation{index});
        }
        FrameObservation& observation = mFrameObservations[surface.pendingIndex];
        observation.positionSum += position;
        if (surface.classId < 0)
            observation.bgrSum += color;
        if (mUseKalman && surface.normal.squaredNorm() > 0.0f)
            observation.varianceSum += MeasurementVariance(sample.position.z(), surface.normal.dot(ray));
        ++observation.count;
    }

    for (const FrameObservation& observation : mFrameObservations)
        UpdateSurface(observation);
}

void DensePointCloudFusion::UpdateSurface(const FrameObservation& observation)
{
    Surface& surface = mSurfaces[observation.surfaceIndex];
    const float inverseCount = 1.0f / observation.count;
    const Eigen::Vector3f measuredPosition = observation.positionSum * inverseCount;
    const float meanGain = 1.0f / (surface.keyFrames + 1);
    surface.position += meanGain * (measuredPosition - surface.position);
    if (surface.classId < 0)
        surface.bgr += meanGain * (observation.bgrSum * inverseCount - surface.bgr);

    if (mUseKalman && surface.normal.squaredNorm() > 0.0f)
    {
        const float measuredOffset = surface.normal.dot(measuredPosition - surface.anchor);
        // Neighboring pixels share depth and pose errors. Average their
        // variances and use a single observation for this keyframe.
        const float measurementVariance = observation.varianceSum * inverseCount;
        if (surface.keyFrames == 0)
        {
            surface.normalOffset = measuredOffset;
            surface.variance = measurementVariance;
        }
        else
        {
            // Static surface, Q = 0. Correct only the position along its normal.
            const float gain = surface.variance / (surface.variance + measurementVariance);
            surface.normalOffset += gain * (measuredOffset - surface.normalOffset);
            surface.variance = gain * measurementVariance;
        }
    }
    ++surface.keyFrames;
    surface.pendingIndex = NoIndex;
}

PointCloud::Ptr DensePointCloudFusion::MakePointCloud() const
{
    auto cloud = pcl::make_shared<PointCloud>();
    cloud->reserve(mSurfaces.size());
    for (const Surface& surface : mSurfaces)
    {
        Eigen::Vector3f position = surface.position;
        if (mUseKalman)
        {
            const float meanOffset = surface.normal.dot(position - surface.anchor);
            position += (surface.normalOffset - meanOffset) * surface.normal;
        }
        PointT point;
        point.x = position.x();
        point.y = position.y();
        point.z = position.z();
        point.b = static_cast<std::uint8_t>(std::lround(surface.bgr.x()));
        point.g = static_cast<std::uint8_t>(std::lround(surface.bgr.y()));
        point.r = static_cast<std::uint8_t>(std::lround(surface.bgr.z()));
        cloud->push_back(point);
    }
    cloud->is_dense = true;
    return cloud;
}
