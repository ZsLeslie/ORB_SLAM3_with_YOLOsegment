#include "PointCloudMapper.h"
#include "Map.h"
#include "PointCloudChangeDetector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <utility>
#include <opencv2/calib3d.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>

namespace
{
// Follow culled parents only to identify the latest keyframe's Atlas map.
// Their local point clouds must not be included in the global map.
KeyFrame* ResolveMapReference(KeyFrame* keyFrame)
{
    while (keyFrame && keyFrame->isBad())
        keyFrame = keyFrame->GetParent();
    return keyFrame;
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

PointCloudMapper::PointCloudMapper()
{
    mpGlobalMap = pcl::make_shared<PointCloud>();
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

PointCloud::Ptr PointCloudMapper::GenerateCameraPointCloud(const KeyFrame *kf,
                                                         const cv::Mat &imRGB,
                                                         const cv::Mat &imDepth)
{
    PointCloud::Ptr pointCloud_temp(new PointCloud);

    if (imRGB.type() != CV_8UC3 || imDepth.type() != CV_32FC1 ||
        imRGB.size() != imDepth.size())
    {
        cerr << "Skipping dense keyframe with incompatible RGB/depth images." << endl;
        return pointCloud_temp;
    }

    // Depth, RGB and masks share the original registered pixel coordinates.
    // Undistort the projection rays, not the images, to match SLAM geometry.
    std::vector<cv::Point2f> cameraRays;
    if (!kf->mDistCoef.empty() && cv::norm(kf->mDistCoef, cv::NORM_INF) > 0.0)
    {
        std::vector<cv::Point2f> pixels;
        pixels.reserve(((imRGB.rows + 2) / 3) * ((imRGB.cols + 2) / 3));
        for (int v = 0; v < imRGB.rows; v += 3)
            for (int u = 0; u < imRGB.cols; u += 3)
                pixels.emplace_back(u, v);
        const cv::Matx33f K(kf->fx, 0.0f, kf->cx,
                           0.0f, kf->fy, kf->cy,
                           0.0f, 0.0f, 1.0f);
        cv::undistortPoints(pixels, cameraRays, K, kf->mDistCoef);
    }

    std::size_t sampleIndex = 0;
    for (int v = 0; v < imRGB.rows; v += 3)
    {
        for (int u = 0; u < imRGB.cols; u += 3, ++sampleIndex)
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
            if (!IsDynamic)
            {
                float d = imDepth.ptr<float>(v)[u];
                if (d <= 0.0f || !std::isfinite(d))
                    continue;
                PointT p;
                p.z = d;
                if (cameraRays.empty())
                {
                    p.x = (u - kf->cx) * p.z / kf->fx;
                    p.y = (v - kf->cy) * p.z / kf->fy;
                }
                else
                {
                    p.x = cameraRays[sampleIndex].x * p.z;
                    p.y = cameraRays[sampleIndex].y * p.z;
                }

                if (cls_id >= 0)
                {
                    cv::Vec3b semColor = GetSemanticColor(cls_id);
                    p.b = semColor[0];
                    p.g = semColor[1];
                    p.r = semColor[2];
                }
                else
                {
                    p.b = imRGB.ptr<cv::Vec3b>(v)[u][0];
                    p.g = imRGB.ptr<cv::Vec3b>(v)[u][1];
                    p.r = imRGB.ptr<cv::Vec3b>(v)[u][2];
                }
                pointCloud_temp->push_back(p);
            }
        }
    }

    // Keep the original camera coordinates: world coordinates are rebuilt from
    // the latest optimized poses instead of being permanently accumulated here.
    pointCloud_temp->is_dense = false;
    return pointCloud_temp;
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
            keyFrame->GetMap() != map || entry.cameraCloud->empty())
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

std::size_t PointCloudMapper::CommonPosePrefix(const CloudPoses& lhs, const CloudPoses& rhs)
{
    const std::size_t limit = std::min(lhs.size(), rhs.size());
    for (std::size_t i = 0; i < limit; ++i)
    {
        if (lhs[i].keyFrame != rhs[i].keyFrame ||
            lhs[i].cameraCloud != rhs[i].cameraCloud ||
            lhs[i].Twc != rhs[i].Twc)
            return i;
    }
    return limit;
}

void PointCloudMapper::FuseKeyFrames(const CloudPoses& poses, PointCloudChangeDetector& detector,
                                    std::vector<std::size_t>& pointOffsets)
{
    PointCloud transformed;
    for (std::size_t i = pointOffsets.size(); i < poses.size(); ++i)
    {
        const auto& pose = poses[i];
        pcl::transformPointCloud(*pose.cameraCloud, transformed, pose.Twc);
        detector.AppendChanges(transformed);
        pointOffsets.push_back(detector.GetCloud().size());
    }
}

void PointCloudMapper::RebuildGlobalMap(PointCloudChangeDetector& detector,
                                       std::vector<std::size_t>& pointOffsets)
{
    CloudPoses poses = CapturePoses();
    const std::size_t prefix = CommonPosePrefix(poses, mvPublishedPoses);
    if (prefix == poses.size() && prefix == mvPublishedPoses.size())
        return;

    // Each decision depends on every earlier contribution in the same map.
    // Reprocess the suffix after a corrected, removed or newly visible frame.
    detector.Truncate(prefix == 0 ? 0 : pointOffsets[prefix - 1]);
    pointOffsets.resize(prefix);
    FuseKeyFrames(poses, detector, pointOffsets);

    const auto map = pcl::make_shared<PointCloud>(detector.GetCloud());
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
    PointCloudChangeDetector detector;
    std::vector<std::size_t> pointOffsets;
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
            RebuildGlobalMap(detector, pointOffsets);
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
    PointCloud::ConstPtr globalMap = GetGlobalMap();
    if (poses.size() != mvPublishedPoses.size() ||
        CommonPosePrefix(poses, mvPublishedPoses) != poses.size())
    {
        PointCloudChangeDetector detector;
        std::vector<std::size_t> pointOffsets;
        FuseKeyFrames(poses, detector, pointOffsets);
        globalMap = pcl::make_shared<PointCloud>(detector.GetCloud());
    }
    if (!globalMap || globalMap->empty())
    {
        cerr << "Cannot save an empty point cloud map." << endl;
        return false;
    }

    PointCloud statisticalFiltered;

    try
    {
        pcl::StatisticalOutlierRemoval<PointT> statisticalFilter;
        statisticalFilter.setMeanK(100);
        statisticalFilter.setStddevMulThresh(1.0);
        statisticalFilter.setInputCloud(globalMap);
        statisticalFilter.filter(statisticalFiltered);
        cout << "Statistical filtering: " << globalMap->size() << " -> "
             << statisticalFiltered.size() << " points." << endl;
    }
    catch (const std::exception &e)
    {
        cerr << "Point cloud filtering failed: " << e.what() << endl;
        return false;
    }

    if (statisticalFiltered.empty())
    {
        cerr << "Statistical filtering produced an empty point cloud." << endl;
        return false;
    }

    const int result = pcl::io::savePCDFileBinary(filename, statisticalFiltered);
    if (result < 0)
    {
        cerr << "Failed to save point cloud map to " << filename << endl;
        return false;
    }

    cout << "Point cloud map saved to " << filename << " ("
         << statisticalFiltered.size() << " filtered points)." << endl;
    return true;
}
