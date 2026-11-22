#include "PointCloudMapper.h"
#include "DensePointCloudFusion.h"
#include "Map.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <opencv2/calib3d.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

namespace
{
constexpr int kSampleStep = 3;

// Follow culled parents only to identify the latest keyframe's Atlas map.
// Their local point clouds must not be included in the global map.
KeyFrame* ResolveMapReference(KeyFrame* keyFrame)
{
    while (keyFrame && keyFrame->isBad())
        keyFrame = keyFrame->GetParent();
    return keyFrame;
}

void ComputeCameraNormals(std::vector<DensePointSample>& samples, int rows, int cols)
{
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < cols; ++col)
        {
            DensePointSample& center = samples[row * cols + col];
            if (center.position.z() <= 0.0f)
                continue;

            const float depthThreshold = std::max(0.025f, 0.015f * center.position.z());
            // Wider differences reduce normal jitter. Stop at invalid samples
            // or depth edges and use one-sided differences at holes.
            const auto neighbor = [&](int dv, int du) -> const Eigen::Vector3f& {
                const Eigen::Vector3f* last = &center.position;
                for (int step = 1; step <= 2; ++step)
                {
                    const int v = row + step * dv;
                    const int u = col + step * du;
                    if (v < 0 || v >= rows || u < 0 || u >= cols)
                        break;
                    const auto& candidate = samples[v * cols + u].position;
                    if (candidate.z() <= 0.0f ||
                        std::abs(candidate.z() - last->z()) > depthThreshold)
                        break;
                    last = &candidate;
                }
                return *last;
            };
            const Eigen::Vector3f across = neighbor(0, 1) - neighbor(0, -1);
            const Eigen::Vector3f down = neighbor(1, 0) - neighbor(-1, 0);
            const Eigen::Vector3f normal = across.cross(down);
            const float norm = normal.norm();
            if (norm > 1e-8f && std::isfinite(norm))
                center.normal = normal / (normal.dot(center.position) > 0.0f ? -norm : norm);
        }
}
}

static cv::Vec3b GetSemanticColor(int classId)
{
    switch (classId)
    {
    case 0:
        return cv::Vec3b(0, 0, 255); // person
    case 56:
        return cv::Vec3b(0, 255, 0); // chair
    case 60:
        return cv::Vec3b(0, 255, 255); // dining table
    case 62:
        return cv::Vec3b(255, 0, 0); // tv
    case 63:
        return cv::Vec3b(255, 0, 255); // laptop
    case 66:
        return cv::Vec3b(255, 255, 0); // keyboard
    default:
        return cv::Vec3b(128, 128, 128); // other semantic classes
    }
}

PointCloudMapper::PointCloudMapper(FusionMode mode, float voxelSize)
    : mFusionMode(mode), mVoxelSize(voxelSize)
{
    if (!std::isfinite(voxelSize) || voxelSize < 0.001f || voxelSize > 1.0f)
        throw std::invalid_argument("DenseMapping.VoxelSize must be between 0.001 and 1.0 metres");
    mpGlobalMap = pcl::make_shared<PointCloud>();
}

std::unique_ptr<PointCloudMapper> PointCloudMapper::FromSettings(const cv::FileStorage& settings)
{
    FusionMode mode = FusionMode::Voxel;
    const cv::FileNode fusion = settings["DenseMapping.Fusion"];
    if (!fusion.empty())
    {
        if (!fusion.isString())
            throw std::invalid_argument("DenseMapping.Fusion must be a string");
        const std::string name = fusion.string();
        if (name == "concat")
            mode = FusionMode::Concatenate;
        else if (name == "kalman")
            mode = FusionMode::Kalman;
        else if (name != "voxel")
            throw std::invalid_argument("DenseMapping.Fusion must be concat, voxel, or kalman");
    }
    const cv::FileNode voxelSize = settings["DenseMapping.VoxelSize"];
    if (!voxelSize.empty() && !voxelSize.isInt() && !voxelSize.isReal())
        throw std::invalid_argument("DenseMapping.VoxelSize must be numeric");
    if (voxelSize.empty())
        return std::make_unique<PointCloudMapper>(mode);
    return std::make_unique<PointCloudMapper>(mode, static_cast<float>(voxelSize));
}

void PointCloudMapper::InsertKeyFrame(KeyFrame *kf, const cv::Mat &imRGB, const cv::Mat &imDepth)
{
    if (!kf || imRGB.empty() || imDepth.empty())
        return;

    KeyFrameInput input;
    input.keyFrame = kf;
    input.rgb = imRGB.clone();
    input.depth = imDepth.clone();
    {
        std::lock_guard<std::mutex> lock(mQueueMutex);
        if (mbFinishRequested)
            return;
        mqKeyFrames.push(std::move(input));
    }
    mQueueUpdated.notify_one();
}

std::shared_ptr<const DensePointCloudFrame> PointCloudMapper::GenerateCameraPointCloud(
    const KeyFrame* kf, const cv::Mat& imRGB, const cv::Mat& imDepth)
{
    auto frame = std::make_shared<DensePointCloudFrame>();

    if (imRGB.type() != CV_8UC3 || imDepth.type() != CV_32FC1 ||
        imRGB.size() != imDepth.size())
    {
        cerr << "Skipping dense keyframe with incompatible RGB/depth images." << endl;
        return frame;
    }

    const int rows = (imRGB.rows + kSampleStep - 1) / kSampleStep;
    const int cols = (imRGB.cols + kSampleStep - 1) / kSampleStep;
    std::vector<DensePointSample>& samples = frame->samples;
    samples.resize(static_cast<std::size_t>(rows) * cols);

    // Depth, RGB and masks share the original registered pixel coordinates.
    // Undistort the projection rays, not the images, to match SLAM geometry.
    std::vector<cv::Point2f> cameraRays;
    if (!kf->mDistCoef.empty() && cv::norm(kf->mDistCoef, cv::NORM_INF) > 0.0)
    {
        std::vector<cv::Point2f> pixels;
        pixels.reserve(samples.size());
        for (int v = 0; v < imRGB.rows; v += kSampleStep)
            for (int u = 0; u < imRGB.cols; u += kSampleStep)
                pixels.emplace_back(u, v);
        const cv::Matx33f K(kf->fx, 0.0f, kf->cx,
                           0.0f, kf->fy, kf->cy,
                           0.0f, 0.0f, 1.0f);
        cv::undistortPoints(pixels, cameraRays, K, kf->mDistCoef);
    }

    std::size_t sampleIndex = 0;
    for (int v = 0; v < imRGB.rows; v += kSampleStep)
    {
        for (int u = 0; u < imRGB.cols; u += kSampleStep, ++sampleIndex)
        {
            cv::Point2i pt(u, v);

            int cls_id = -1;
            if (!kf->mSemanticClsMap.empty() && kf->mSemanticClsMap.type() == CV_32S &&
                u < kf->mSemanticClsMap.cols && v < kf->mSemanticClsMap.rows)
            {
                cls_id = kf->mSemanticClsMap.at<int>(v, u);
            }

            const bool isPersonCloudPoint =
                (!kf->mPersonCloudMask.empty() && kf->mPersonCloudMask.type() == CV_8U &&
                 u < kf->mPersonCloudMask.cols && v < kf->mPersonCloudMask.rows &&
                 kf->mPersonCloudMask.at<uchar>(v, u) > 0) ||
                cls_id == 0;

            // Feature extraction only rejects moving people. Dense mapping uses a
            // closed, filled and dilated mask for every detected person.
            if (isPersonCloudPoint)
            {
                ++mRejectedPersonPoints;
                continue;
            }

            bool IsDynamic = false;
            for (const auto& area : kf->mvDynamicArea)
            {
                if (!area.empty() && area.type() == CV_32F &&
                    pt.x >= 0 && pt.x < area.cols && pt.y >= 0 && pt.y < area.rows &&
                    area.at<float>(pt.y, pt.x) > 0.1f)
                {
                    IsDynamic = true;
                    break;
                }
            }
            const float depth = imDepth.ptr<float>(v)[u];
            if (IsDynamic || depth <= 0.0f || !std::isfinite(depth))
                continue;

            const float x = cameraRays.empty() ? (u - kf->cx) * depth / kf->fx
                                               : cameraRays[sampleIndex].x * depth;
            const float y = cameraRays.empty() ? (v - kf->cy) * depth / kf->fy
                                               : cameraRays[sampleIndex].y * depth;
            if (!std::isfinite(x) || !std::isfinite(y))
                continue;

            DensePointSample& sample = samples[sampleIndex];
            sample.position = Eigen::Vector3f(x, y, depth);
            sample.classId = cls_id;
            sample.bgr = cls_id >= 0 ? GetSemanticColor(cls_id) : imRGB.ptr<cv::Vec3b>(v)[u];
        }
    }

    if (mFusionMode != FusionMode::Concatenate)
        ComputeCameraNormals(samples, rows, cols);
    samples.erase(std::remove_if(samples.begin(), samples.end(), [](const DensePointSample& sample) {
        return sample.position.z() <= 0.0f;
    }), samples.end());
    return frame;
}

PointCloudMapper::CloudPoses PointCloudMapper::CapturePoses() const
{
    CloudPoses poses;
    if (mvKeyFrameClouds.empty())
        return poses;

    KeyFrame* reference = ResolveMapReference(mvKeyFrameClouds.back().keyFrame);
    Map* map = reference ? reference->GetMap() : nullptr;
    if (!map)
        return poses;

    // Hold SLAM's update lock only while copying poses, never during the much
    // more expensive point-cloud transforms and filtering.
    std::unique_lock<std::mutex> lock(map->mMutexMapUpdate);
    poses.reserve(mvKeyFrameClouds.size());
    for (const auto& entry : mvKeyFrameClouds)
    {
        // Recheck on every snapshot: culling can happen after this cloud has
        // already been fused, including while no new keyframes are arriving.
        KeyFrame* keyFrame = entry.keyFrame;
        if (!keyFrame || keyFrame->isBad() ||
            keyFrame->GetMap() != map || entry.cameraCloud->samples.empty())
            continue;

        CloudPose pose;
        pose.keyFrame = keyFrame;
        pose.cameraCloud = entry.cameraCloud;
        pose.Twc = keyFrame->GetPoseInverse().matrix();
        poses.push_back(std::move(pose));
    }

    // Separate Atlas maps have independent world frames. Only combine clouds
    // from valid keyframes in the map of the latest accepted dense keyframe.
    // After a merge, valid keyframes' updated map pointers include both maps.
    return poses;
}

bool PointCloudMapper::SamePoses(const CloudPoses& lhs, const CloudPoses& rhs)
{
    if (lhs.size() != rhs.size())
        return false;
    for (std::size_t i = 0; i < lhs.size(); ++i)
    {
        if (lhs[i].keyFrame != rhs[i].keyFrame ||
            lhs[i].cameraCloud != rhs[i].cameraCloud ||
            !lhs[i].Twc.isApprox(rhs[i].Twc, 1e-6f))
            return false;
    }
    return true;
}

PointCloud::Ptr PointCloudMapper::BuildGlobalMap(const CloudPoses& poses) const
{
    if (mFusionMode != FusionMode::Concatenate)
    {
        DensePointCloudFusion fusion(mVoxelSize, mFusionMode == FusionMode::Kalman);
        for (const auto& pose : poses)
            fusion.AddKeyFrame(*pose.cameraCloud, pose.Twc);
        return fusion.MakePointCloud();
    }

    PointCloud::Ptr map(new PointCloud);
    std::size_t pointCount = 0;
    for (const auto& pose : poses)
        pointCount += pose.cameraCloud->samples.size();
    map->reserve(pointCount);

    for (const auto& pose : poses)
    {
        const Eigen::Matrix3f rotation = pose.Twc.topLeftCorner<3, 3>();
        const Eigen::Vector3f translation = pose.Twc.topRightCorner<3, 1>();
        for (const DensePointSample& sample : pose.cameraCloud->samples)
        {
            PointT point;
            point.getVector3fMap() = rotation * sample.position + translation;
            point.b = sample.bgr[0];
            point.g = sample.bgr[1];
            point.r = sample.bgr[2];
            map->push_back(point);
        }
    }
    map->is_dense = false;
    return map;
}

void PointCloudMapper::RebuildGlobalMap(bool force)
{
    CloudPoses poses = CapturePoses();
    if (!force && SamePoses(poses, mvPublishedPoses))
        return;

    PointCloud::Ptr map = BuildGlobalMap(poses);
    {
        std::lock_guard<std::mutex> lock(mGlobalMapMutex);
        mpGlobalMap = map;
    }
    mvPublishedPoses = std::move(poses);
}

PointCloud::ConstPtr PointCloudMapper::GetGlobalMap() const
{
    std::lock_guard<std::mutex> lock(mGlobalMapMutex);
    return mpGlobalMap;
}

void PointCloudMapper::run()
{
    cout << endl
         << "PointCloudMapping thread start!" << endl;
    using Clock = std::chrono::steady_clock;
    const auto rebuildInterval = std::chrono::milliseconds(500);
    auto nextRebuild = Clock::now();
    while (true)
    {
        KeyFrameInput input;
        bool finish = false;

        {
            std::unique_lock<std::mutex> lock(mQueueMutex);
            mQueueUpdated.wait_until(lock, nextRebuild, [this] {
                return mbFinishRequested || !mqKeyFrames.empty();
            });
            if (!mqKeyFrames.empty())
            {
                input = std::move(mqKeyFrames.front());
                mqKeyFrames.pop();
            }
            else
                finish = mbFinishRequested;
        }
        if (input.keyFrame)
        {
            KeyFrameCloud entry;
            entry.keyFrame = input.keyFrame;
            entry.cameraCloud = GenerateCameraPointCloud(input.keyFrame, input.rgb, input.depth);
            mvKeyFrameClouds.push_back(std::move(entry));
        }

        // Check even when no new keyframes arrive, and also during a continuous
        // stream of inputs: BA/loop corrections must not wait for an empty queue.
        if (finish || Clock::now() >= nextRebuild)
        {
            RebuildGlobalMap(finish);
            nextRebuild = Clock::now() + rebuildInterval;
        }
        if (finish)
            break;
    }

    mbFinished.store(true);
    cout << "PointCloudMapping thread finished with " << GetGlobalMap()->size()
         << " points from " << mvPublishedPoses.size() << " keyframes; rejected "
         << mRejectedPersonPoints << " person-mask samples." << endl;
}

void PointCloudMapper::RequestFinish()
{
    {
        std::lock_guard<std::mutex> lock(mQueueMutex);
        mbFinishRequested = true;
    }
    mQueueUpdated.notify_one();
}

bool PointCloudMapper::SaveMap(const std::string &filename) const
{
    if (!mbFinished.load())
    {
        cerr << "Cannot save point cloud before the mapping thread has finished." << endl;
        return false;
    }

    // Re-read poses at export as well, in case a caller stopped this worker
    // before the optimizer. System::Shutdown normally already waits for BA.
    const CloudPoses poses = CapturePoses();
    PointCloud::ConstPtr globalMap = SamePoses(poses, mvPublishedPoses)
        ? GetGlobalMap() : BuildGlobalMap(poses);
    if (!globalMap || globalMap->empty())
    {
        cerr << "Cannot save an empty point cloud map." << endl;
        return false;
    }

    PointCloud::ConstPtr filtered = globalMap;

    try
    {
        if (mFusionMode == FusionMode::Concatenate)
        {
            auto voxelFiltered = pcl::make_shared<PointCloud>();
            pcl::VoxelGrid<PointT> voxelFilter;
            voxelFilter.setLeafSize(0.01f, 0.01f, 0.01f);
            voxelFilter.setInputCloud(globalMap);
            voxelFilter.filter(*voxelFiltered);
            cout << "Voxel filtering: " << globalMap->size() << " -> "
                 << voxelFiltered->size() << " points." << endl;
            filtered = voxelFiltered;
        }

        if (filtered->size() > 2)
        {
            auto statisticalFiltered = pcl::make_shared<PointCloud>();
            pcl::StatisticalOutlierRemoval<PointT> statisticalFilter;
            statisticalFilter.setMeanK(static_cast<int>(std::min<std::size_t>(100, filtered->size() - 1)));
            statisticalFilter.setStddevMulThresh(1.0);
            statisticalFilter.setInputCloud(filtered);
            statisticalFilter.filter(*statisticalFiltered);
            cout << "Statistical filtering: " << filtered->size() << " -> "
                 << statisticalFiltered->size() << " points." << endl;
            filtered = statisticalFiltered;
        }
    }
    catch (const std::exception &e)
    {
        cerr << "Point cloud filtering failed: " << e.what() << endl;
        return false;
    }

    if (filtered->empty())
    {
        cerr << "Point cloud filtering produced an empty point cloud." << endl;
        return false;
    }

    const int result = pcl::io::savePCDFileBinary(filename, *filtered);
    if (result < 0)
    {
        cerr << "Failed to save point cloud map to " << filename << endl;
        return false;
    }

    cout << "Point cloud map saved to " << filename << " ("
         << filtered->size() << " filtered points)." << endl;
    return true;
}
