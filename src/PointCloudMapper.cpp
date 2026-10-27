#include "PointCloudMapper.h"

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
    cout << "voxel set start" << endl;
    mpVoxel = pcl::make_shared<pcl::VoxelGrid<PointT>>();
    mpStatistic = pcl::make_shared<pcl::StatisticalOutlierRemoval<PointT>>();
    mpVoxel->setLeafSize(0.01, 0.01, 0.01);
    mpStatistic->setMeanK(100);
    cout << "voxel set finish" << endl;
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
            bool IsDynamic = false;
            for (auto area : kf->mvDynamicArea)
            {
                if (*area.ptr<float>(pt.y, pt.x) > 0.1)
                    IsDynamic = true;
            }
            if (!IsDynamic)
            {
                float d = imDepth.ptr<float>(v)[u];
                if (d < 0.0)
                    continue;
                PointT p;
                p.z = d;
                p.x = (u - kf->cx) * p.z / kf->fx;
                p.y = (v - kf->cy) * p.z / kf->fy;

                int cls_id = -1;
                if (!kf->mSemanticClsMap.empty())
                {
                    cls_id = kf->mSemanticClsMap.at<int>(v, u);
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

    Eigen::Isometry3d T = ORB_SLAM3::Converter::toSE3Quat(kf->GetPose());
    PointCloud::Ptr pointCloud(new PointCloud);
    pcl::transformPointCloud(*pointCloud_temp, *pointCloud, T.inverse().matrix());
    pointCloud->is_dense = false;
    return pointCloud;
}

void PointCloudMapper::run()
{
    pcl::visualization::CloudViewer Viewer("Viewer");
    cout << endl
         << "PointCloudMapping thread start!" << endl;
    int ID = 0;
    while (1)
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
            PointCloud::Ptr new_filtered(new PointCloud);
            mpVoxel->setInputCloud(pointCloud_new);
            mpVoxel->filter(*new_filtered);
            mpStatistic->setInputCloud(new_filtered);
            mpStatistic->filter(*new_filtered);
            *mpGlobalMap += *new_filtered;
        }

        Viewer.showCloud(mpGlobalMap);
    }
}
