#include "PointCloudMapper.h"

#include <chrono>
#include <exception>
#include <thread>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <vtkCallbackCommand.h>
#include <vtkRenderWindow.h>
#include <vtkVersion.h>

void PointCloudMapper::RunViewer()
{
    try
    {
        // PCL/VTK objects must be created, used and destroyed on one thread.
        pcl::visualization::PCLVisualizer viewer("ORB-SLAM3: Dense Point Cloud");
        viewer.setBackgroundColor(0.05, 0.05, 0.05);
        viewer.setSize(960, 720);
        viewer.addCoordinateSystem(0.5);
        viewer.initCameraParameters();
        // RGB-D camera coordinates: x right, y down, z forward.
        viewer.setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0);

        auto renderWindow = viewer.getRenderWindow();
        auto* interactor = renderWindow->GetInteractor();
        bool closeRequested = false;
        auto closeCallback = vtkSmartPointer<vtkCallbackCommand>::New();
        closeCallback->SetClientData(&closeRequested);
        closeCallback->SetCallback([](vtkObject*, unsigned long, void* data, void*) {
            *static_cast<bool*>(data) = true;
        });
        // Defer finalization until ProcessEvents has returned. VTK 9.1 can
        // otherwise continue reading the X display after closing it.
        closeCallback->SetAbortFlagOnExecute(1);
        const auto closeObserver = interactor->AddObserver(vtkCommand::ExitEvent, closeCallback, 1.0);

        const std::string cloudId = "global_map";
        PointCloud::ConstPtr displayedCloud;
        while (!mbFinished.load() && !closeRequested && !viewer.wasStopped())
        {
            // GetGlobalMap holds the mutex only long enough to copy the shared
            // pointer. Rendering never holds up publication of the next map.
            PointCloud::ConstPtr cloud = GetGlobalMap();
            if (cloud != displayedCloud)
            {
                if (cloud && !cloud->empty())
                {
                    pcl::visualization::PointCloudColorHandlerRGBField<PointT> colors(cloud);
                    if (!viewer.updatePointCloud<PointT>(cloud, colors, cloudId))
                    {
                        viewer.addPointCloud<PointT>(cloud, colors, cloudId);
                        viewer.setPointCloudRenderingProperties(
                            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, cloudId);
                        // Fit the first cloud, then preserve mouse-controlled
                        // camera position as subsequent snapshots arrive.
                        viewer.resetCamera();
                    }
                }
                else if (displayedCloud && !displayedCloud->empty())
                    viewer.removePointCloud(cloudId);

                displayedCloud = cloud;
            }

            // Pump window events even while no new keyframes are arriving.
#if VTK_MAJOR_VERSION >= 9
            // Avoid PCL 1.12 spinOnce's Start/TerminateApp timer loop on VTK 9.1:
            // https://github.com/PointCloudLibrary/pcl/issues/5237
            interactor->ProcessEvents();
#else
            viewer.spinOnce(10);
#endif
            if (closeRequested || viewer.wasStopped())
                break;
            renderWindow->Render();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        interactor->RemoveObserver(closeObserver);
        // PCL 1.12 close() finalizes before TerminateApp(), which accesses an
        // already closed X display on VTK 9.1. Stop the interactor first.
        interactor->TerminateApp();
        renderWindow->Finalize();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Point cloud viewer failed: " << e.what() << std::endl;
    }
}
