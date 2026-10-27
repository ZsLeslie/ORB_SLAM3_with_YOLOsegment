#include "PointCloudMapper.h"

#include <chrono>
#include <cmath>
#include <thread>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

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

void PointCloudMapper::InsertKeyFrame(KeyFrame *kf, cv::Mat &imRGB, cv::Mat &imDepth)
{
    std::lock_guard<std::mutex> lck_loadKF(mmLoadKFMutex);
    mqKeyFrame.push(kf);
    mqRGB.push(imRGB.clone());
    mqDepth.push(imDepth.clone());
}

PointCloud::Ptr PointCloudMapper::GeneratePointCloud(KeyFrame *kf, cv::Mat &imRGB, cv::Mat &imDepth)
{
    PointCloud::Ptr pointCloud_temp(new PointCloud);

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
            for (auto area : kf->mvDynamicArea)
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

    Eigen::Isometry3d T = ORB_SLAM3::Converter::toSE3Quat(kf->GetPose());
    PointCloud::Ptr pointCloud(new PointCloud);
    pcl::transformPointCloud(*pointCloud_temp, *pointCloud, T.inverse().matrix());
    pointCloud->is_dense = false;
    return pointCloud;
}

void PointCloudMapper::run()
{
    cout << endl
         << "PointCloudMapping thread start!" << endl;
    int ID = 0;
    while (true)
    {
        KeyFrame *kf = nullptr;
        cv::Mat rgb, depth;

        {
            std::lock_guard<std::mutex> lock(mmLoadKFMutex);
            if (!mqKeyFrame.empty())
            {
                kf = mqKeyFrame.front();
                mqKeyFrame.pop();
                rgb = mqRGB.front().clone();
                mqRGB.pop();
                depth = mqDepth.front().clone();
                mqDepth.pop();
            }
        }
        if (kf)
        {
            PointCloud::Ptr pointCloud_new = GeneratePointCloud(kf, rgb, depth);

            ID++;
            *mpGlobalMap += *pointCloud_new;
            continue;
        }

        if (mbFinishRequested.load())
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    mbFinished.store(true);
    cout << "PointCloudMapping thread finished with " << mpGlobalMap->size()
         << " points from " << ID << " keyframes; rejected "
         << mRejectedPersonPoints << " person-mask samples." << endl;
}

void PointCloudMapper::RequestFinish()
{
    mbFinishRequested.store(true);
}

bool PointCloudMapper::SaveMap(const std::string &filename) const
{
    if (!mbFinished.load())
    {
        cerr << "Cannot save point cloud before the mapping thread has finished." << endl;
        return false;
    }

    if (!mpGlobalMap || mpGlobalMap->empty())
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
        voxelFilter.setInputCloud(mpGlobalMap);
        voxelFilter.filter(voxelFiltered);
        cout << "Voxel filtering: " << mpGlobalMap->size() << " -> "
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
