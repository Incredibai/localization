#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_


#include <ros/ros.h>
#include <math.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>

#include "cloud_msgs/cloud_info.h"

#include <opencv/cv.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

#define PI 3.14159265

using namespace std;

typedef pcl::PointXYZI  PointType;

// HDL32

extern const int N_SCAN = 32;              //16
extern const int Horizon_SCAN = 2172;      //1800;
extern const float ang_res_x = 0.166;       //0.2
extern const float ang_res_y = 1.33;       //2.0
extern const float ang_bottom = 30.0+0.67;     //15.0+0.1;
extern const int groundScanInd = 23;//16;       //7


// HDL64E
/*
extern const int N_SCAN = 64;
extern const int Horizon_SCAN = 1800;
extern const float ang_res_x = 0.2;
extern const float ang_res_y = 0.427;
extern const float ang_bottom = 24.9;
extern const int groundScanInd = 50;

extern const bool loopClosureEnableFlag = true;
extern const bool mappingDSFEnableFlag = true;
extern const double mappingProcessInterval = 0.3;
extern const double localizationProcessInterval = 0.3;
extern const double localizationSearchRadius = 20;
extern const double DSFCornerLeafSize = 0.2; // Each Frame 
extern const double DSFSurfLeafSize = 0.4; // Each Frame
extern const double DSFOutlierLeafSize = 0.4;
extern const double DSFHistoryKeyFramesLeafSize = 0.4;
extern const double DSFSurroundingKeyPosesLeafSize = 1.0;
extern const double DSFGlobalMapKeyPosesLeafSize = 1.0;
extern const double DSFGlobalMapKeyFramesLeafSize = 1.0;
extern const double W_intensity = 0.0;
*/


// WILLIAM BEGIN
// extern const bool inputIsPointType = true;
extern const bool loopClosureEnableFlag = true;
extern const bool mappingDSFEnableFlag = true;
extern const double mappingProcessInterval = 0.3;
extern const double localizationProcessInterval = 0.5;
extern const double localizationSearchRadius = 20;
extern const double DSFCornerLeafSize = 0.2; // Each Frame 
extern const double DSFSurfLeafSize = 0.4; // Each Frame
extern const double DSFOutlierLeafSize = 0.4;
extern const double DSFHistoryKeyFramesLeafSize = 0.4;
extern const double DSFSurroundingKeyPosesLeafSize = 1.0;
extern const double DSFGlobalMapKeyPosesLeafSize = 1.0;
extern const double DSFGlobalMapKeyFramesLeafSize = 1.0;
extern const double W_intensity = 0.2;
// WILLIAM END


/*
        downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);
        downSizeFilterOutlier.setLeafSize(0.4, 0.4, 0.4);
        downSizeFilterHistoryKeyFrames.setLeafSize(0.4, 0.4, 0.4);
        downSizeFilterSurroundingKeyPoses.setLeafSize(1.0, 1.0, 1.0);
        downSizeFilterGlobalMapKeyPoses.setLeafSize(1.0, 1.0, 1.0);
        downSizeFilterGlobalMapKeyFrames.setLeafSize(1.0, 1.0, 1.0);
*/

extern const float scanPeriod = 0.1;
extern const int systemDelay = 0;
extern const int imuQueLength = 200;
extern const string imuTopic = "/imu/data";


extern const float sensorMountAngle = 0.0;
extern const float segmentTheta = 1.0472;
extern const int segmentValidPointNum = 5;
extern const int segmentValidLineNum = 3;
extern const float segmentAlphaX = ang_res_x / 180.0 * M_PI;
extern const float segmentAlphaY = ang_res_y / 180.0 * M_PI;

extern const int edgeFeatureNum = 3;//2;
extern const int surfFeatureNum = 5;//4;
extern const int sectionsTotal = 6;
extern const float edgeThreshold = 0.1;
extern const float surfThreshold = 0.1;
extern const float nearestFeatureSearchSqDist = 25;

extern const float surroundingKeyframeSearchRadius = 50.0;
extern const int   surroundingKeyframeSearchNum = 50;

extern const float historyKeyframeSearchRadius = 10;//30; //5.0;
extern const int   historyKeyframeSearchNum = 20;//10;    //25;
extern const float historyKeyframeFitnessScore = 3;//1.5;

//TODO GPS kd tree
extern const int GPSsearchKpoint = 5;
extern const float GPSSearchRadius = 0.0005; //unused now


extern const float globalMapVisualizationSearchRadius = 3000.0;


struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
} EIGEN_ALIGN16; // 确保new操作符对齐操作 // 强制SSE对齐

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time)
)

typedef PointXYZIRPYT  PointTypePose;

#endif
