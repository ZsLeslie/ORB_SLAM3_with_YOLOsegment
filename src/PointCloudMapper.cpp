#include "PointCloudMapper.h"
#include "Map.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <utility>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

namespace
{
// Culling freezes a keyframe's pose relative to its parent. Follow the same
// chain used by trajectory export, so its cloud follows future parent updates.
KeyFrame* ResolveReference(KeyFrame* keyFrame, Sophus::SE3f& Tcr)
{
    Tcr = Sophus::SE3f();
    while (keyFrame && keyFrame->isBad())
    {
        Tcr = Tcr * keyFrame->mTcp;
        keyFrame = keyFrame->GetParent();
    }
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

    for (int v = 0; v < imRGB.rows; v += 3)
    {
        for (int u = 0; u < imRGB.cols; u += 3)
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
                p.x = (u - kf->cx) * p.z / kf->fx;
                p.y = (v - kf->cy) * p.z / kf->fy;

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

    Sophus::SE3f Tcr;
    KeyFrame* reference = ResolveReference(mvKeyFrameClouds.back().keyFrame, Tcr);
    Map* map = reference ? reference->GetMap() : nullptr;
    if (!map)
        return poses;

    // Hold SLAM's update lock only while copying poses, never during the much
    // more expensive point-cloud transforms and filtering.
    std::unique_lock<std::mutex> lock(map->mMutexMapUpdate);
    poses.reserve(mvKeyFrameClouds.size());
    for (const auto& entry : mvKeyFrameClouds)
    {
        reference = ResolveReference(entry.keyFrame, Tcr);
        if (!reference || reference->GetMap() != map || entry.cameraCloud->empty())
            continue;

        CloudPose pose;
        pose.keyFrame = entry.keyFrame;
        pose.cameraCloud = entry.cameraCloud;
        pose.Twc = (Tcr * reference->GetPose()).inverse().matrix();
        poses.push_back(std::move(pose));
    }

    // Separate Atlas maps have independent world frames. Only combine clouds
    // in the map of the most recently accepted dense keyframe. After a merge,
    // resolving the references naturally includes clouds from both old maps.
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

PointCloud::Ptr PointCloudMapper::BuildGlobalMap(const CloudPoses& poses)
{
    PointCloud::Ptr map(new PointCloud);
    std::size_t pointCount = 0;
    for (const auto& pose : poses)
        pointCount += pose.cameraCloud->size();
    map->reserve(pointCount);

    PointCloud transformed;
    for (const auto& pose : poses)
    {
        pcl::transformPointCloud(*pose.cameraCloud, transformed, pose.Twc);
        *map += transformed;
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

    PointCloud voxelFiltered;
    PointCloud statisticalFiltered;

    try
    {
        pcl::VoxelGrid<PointT> voxelFilter;
        voxelFilter.setLeafSize(0.01f, 0.01f, 0.01f);
        voxelFilter.setInputCloud(globalMap);
        voxelFilter.filter(voxelFiltered);
        cout << "Voxel filtering: " << globalMap->size() << " -> "
             << voxelFiltered.size() << " points." << endl;

        if (voxelFiltered.empty())
        {
            cerr << "Voxel filtering produced an empty point cloud." << endl;
            return false;
        }

        pcl::StatisticalOutlierRemoval<PointT> statisticalFilter;
        statisticalFilter.setMeanK(100);
        statisticalFilter.setStddevMulThresh(1.0);
        statisticalFilter.setInputCloud(voxelFiltered.makeShared());
        statisticalFilter.filter(statisticalFiltered);
        cout << "Statistical filtering: " << voxelFiltered.size() << " -> "
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
