// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.
#include "utility.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/ISAM2.h>

// WILLIAM BEGIN
// #include "pcl/io/point_cloud_image_extractors.h"

using namespace gtsam;

class mapOptimization{

private:

    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;

    noiseModel::Diagonal::shared_ptr priorNoise;
    noiseModel::Diagonal::shared_ptr odometryNoise;
    noiseModel::Diagonal::shared_ptr constraintNoise;

    ros::NodeHandle nh;

    ros::Publisher pubLaserCloudSurround;
    // WILLIAM BEGIN
    ros::Publisher pubLaserCloudCornerSurround;
    ros::Publisher pubLaserCloudSurfSurround;
    // WILLIAM END

    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubKeyPoses;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubGPSFrames;

    ros::Subscriber subLaserCloudCornerLast;
    ros::Subscriber subLaserCloudSurfLast;
    ros::Subscriber subOutlierCloudLast;
    ros::Subscriber subLaserOdometry;
    ros::Subscriber subImu;
    ros::Subscriber subGPS;//todo add gps
    float latitude, longitude, altitude;

    nav_msgs::Odometry odomAftMapped;
    tf::StampedTransform aftMappedTrans;
    tf::TransformBroadcaster tfBroadcaster;

    vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> surfCloudKeyFrames;
    vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> outlierCloudKeyFrames;

    deque<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> recentCornerCloudKeyFrames;
    deque<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> recentSurfCloudKeyFrames;
    deque<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> recentOutlierCloudKeyFrames;
    int latestFrameID;

    vector<int> surroundingExistingKeyPosesID;
    deque<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> surroundingCornerCloudKeyFrames;
    deque<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> surroundingSurfCloudKeyFrames;
    deque<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> surroundingOutlierCloudKeyFrames;
    
    pcl::PointXYZINormal previousRobotPosPoint;
    pcl::PointXYZINormal currentRobotPosPoint;

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    //todo
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudGPS3D;
    pcl::PointXYZI currentGPSPoint;
    bool PoseLoopCloseDetectFlag;
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeHistoryGPS;

    

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr surroundingKeyPoses;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr surroundingKeyPosesDS;

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserCloudCornerLast;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserCloudSurfLast;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserCloudCornerLastDS;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserCloudSurfLastDS;

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserCloudOutlierLast;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserCloudOutlierLastDS;

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserCloudSurfTotalLast;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserCloudSurfTotalLastDS;

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserCloudOri;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr coeffSel;

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kdtreeHistoryKeyPoses;

    
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr nearHistoryCornerKeyFrameCloud;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr nearHistoryCornerKeyFrameCloudDS;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr nearHistorySurfKeyFrameCloud;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr nearHistorySurfKeyFrameCloudDS;

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr latestCornerKeyFrameCloud;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr latestSurfKeyFrameCloud;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr latestSurfKeyFrameCloudDS;

    pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kdtreeGlobalMap;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr globalMapKeyPoses;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr globalMapKeyPosesDS;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr globalMapKeyFrames;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr globalMapKeyFramesDS;
    // WILLIAM BEGIN
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr globalMapCornerCloud;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr globalMapCornerCloudDS;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr globalMapSurfCloud;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr globalMapSurfCloudDS;
    pcl::PCDWriter saveMap;
    // WILLIAM END

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    pcl::VoxelGrid<pcl::PointXYZINormal> downSizeFilterCorner;
    pcl::VoxelGrid<pcl::PointXYZINormal> downSizeFilterSurf;
    pcl::VoxelGrid<pcl::PointXYZINormal> downSizeFilterOutlier;
    pcl::VoxelGrid<pcl::PointXYZINormal> downSizeFilterHistoryKeyFrames;
    pcl::VoxelGrid<pcl::PointXYZINormal> downSizeFilterSurroundingKeyPoses;
    pcl::VoxelGrid<pcl::PointXYZINormal> downSizeFilterGlobalMapKeyPoses;
    pcl::VoxelGrid<pcl::PointXYZINormal> downSizeFilterGlobalMapKeyFrames;

    double timeLaserCloudCornerLast;
    double timeLaserCloudSurfLast;
    double timeLaserOdometry;
    double timeLaserCloudOutlierLast;
    double timeLastGloalMapPublish;

    bool newLaserCloudCornerLast;
    bool newLaserCloudSurfLast;
    bool newLaserOdometry;
    bool newLaserCloudOutlierLast;


    float transformLast[6];
    float transformSum[6];
    float transformIncre[6];
    float transformTobeMapped[6];
    float transformBefMapped[6];
    float transformAftMapped[6];


    int imuPointerFront;
    int imuPointerLast;

    double imuTime[imuQueLength];
    float imuRoll[imuQueLength];
    float imuPitch[imuQueLength];

    std::mutex mtx;

    double timeLastProcessing;

    pcl::PointXYZINormal pointOri, pointSel, pointProj, coeff;

    cv::Mat matA0;
    cv::Mat matB0;
    cv::Mat matX0;

    cv::Mat matA1;
    cv::Mat matD1;
    cv::Mat matV1;

    bool isDegenerate;
    cv::Mat matP;

    int laserCloudCornerFromMapDSNum;
    int laserCloudSurfFromMapDSNum;
    int laserCloudCornerLastDSNum;
    int laserCloudSurfLastDSNum;
    int laserCloudOutlierLastDSNum;
    int laserCloudSurfTotalLastDSNum;
    // WILLIAM BEGIN
    int globalMapKeyFramesDSNum;
    int globalMapCornerCloudDSNum;
    int globalMapSurfCloudDSNum;
    double mapOptimizationTime_1;
    double mapOptimizationTime_2;
    double mapOptimizationDuration;
    // WILLIAM END

    // WILLIAM BEGIN
    bool savedOneFrameRawData = false;

    bool potentialLoopFlag;
    double timeSaveFirstCurrentScanForLoopClosure = 0.0;
    int closestHistoryFrameID;
    int latestFrameIDLoopCloure;
    float FitnessScore = historyKeyframeFitnessScore;

    bool aLoopIsClosed;

    float cRoll, sRoll, cPitch, sPitch, cYaw, sYaw, tX, tY, tZ;
    float ctRoll, stRoll, ctPitch, stPitch, ctYaw, stYaw, tInX, tInY, tInZ;

public:

    

    mapOptimization():
        nh("~")
    {
    	ISAM2Params parameters;
		parameters.relinearizeThreshold = 0.01;
		parameters.relinearizeSkip = 1;
    	isam = new ISAM2(parameters);

        pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("/key_pose_origin", 2);
        pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 2);
        // WILLIAM BEGIN
        pubLaserCloudCornerSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_surround", 2);
        pubLaserCloudSurfSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_surround", 2);
        // WILLIAM END


        pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> ("/aft_mapped_to_init", 5);

        subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2, &mapOptimization::laserCloudCornerLastHandler, this);
        subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2, &mapOptimization::laserCloudSurfLastHandler, this);
        subOutlierCloudLast = nh.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2, &mapOptimization::laserCloudOutlierLastHandler, this);
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, &mapOptimization::laserOdometryHandler, this);
        subGPS = nh.subscribe<nav_msgs::Odometry>("/GPS/data", 5, &mapOptimization::GPSOdomeHandler, this);

        pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/history_cloud", 2);
        pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/corrected_cloud", 2);
        pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/recent_cloud", 2);
        pubGPSFrames = nh.advertise<sensor_msgs::PointCloud2>("/GPS_cloud", 2);

        // downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);
        // downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);
        // downSizeFilterOutlier.setLeafSize(0.4, 0.4, 0.4);
        downSizeFilterCorner.setLeafSize(DSFCornerLeafSize, DSFCornerLeafSize, DSFCornerLeafSize);
        downSizeFilterSurf.setLeafSize(DSFSurfLeafSize, DSFSurfLeafSize, DSFSurfLeafSize);
        downSizeFilterOutlier.setLeafSize(DSFOutlierLeafSize, DSFOutlierLeafSize, DSFOutlierLeafSize);

        downSizeFilterHistoryKeyFrames.setLeafSize(DSFHistoryKeyFramesLeafSize, DSFHistoryKeyFramesLeafSize, DSFHistoryKeyFramesLeafSize);
        //todo here
        // downSizeFilterHistoryKeyFrames.setLeafSize(1.0, 1.0, 1.0);//todo gc 0.4->0.05
        downSizeFilterSurroundingKeyPoses.setLeafSize(DSFSurroundingKeyPosesLeafSize, DSFSurroundingKeyPosesLeafSize, DSFSurroundingKeyPosesLeafSize);

        downSizeFilterGlobalMapKeyPoses.setLeafSize(DSFGlobalMapKeyPosesLeafSize, DSFGlobalMapKeyPosesLeafSize, DSFGlobalMapKeyPosesLeafSize);
        downSizeFilterGlobalMapKeyFrames.setLeafSize(DSFGlobalMapKeyFramesLeafSize, DSFGlobalMapKeyFramesLeafSize, DSFGlobalMapKeyFramesLeafSize);
        // downSizeFilterGlobalMapKeyFrames.setLeafSize(1.0, 1.0, 1.0);////todo gc 0.4->0.05

        odomAftMapped.header.frame_id = "/camera_init";
        odomAftMapped.child_frame_id = "/aft_mapped";

        aftMappedTrans.frame_id_ = "/camera_init";
        aftMappedTrans.child_frame_id_ = "/aft_mapped";

        allocateMemory();
    }

    void allocateMemory(){

        cloudKeyPoses3D.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        //todo
        cloudGPS3D.reset(new pcl::PointCloud<pcl::PointXYZI>());
        latitude = 0;
        longitude = 0;
        altitude = 0;

        PoseLoopCloseDetectFlag = false;
        kdtreeHistoryGPS.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<pcl::PointXYZINormal>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<pcl::PointXYZINormal>());

        surroundingKeyPoses.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        surroundingKeyPosesDS.reset(new pcl::PointCloud<pcl::PointXYZINormal>());        

        laserCloudCornerLast.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        laserCloudSurfLast.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        laserCloudCornerLastDS.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        laserCloudSurfLastDS.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        laserCloudOutlierLast.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        laserCloudOutlierLastDS.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        laserCloudSurfTotalLast.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        laserCloudSurfTotalLastDS.reset(new pcl::PointCloud<pcl::PointXYZINormal>());

        laserCloudOri.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        coeffSel.reset(new pcl::PointCloud<pcl::PointXYZINormal>());

        laserCloudCornerFromMap.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<pcl::PointXYZINormal>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<pcl::PointXYZINormal>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<pcl::PointXYZINormal>());

        
        nearHistoryCornerKeyFrameCloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        nearHistoryCornerKeyFrameCloudDS.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        nearHistorySurfKeyFrameCloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        nearHistorySurfKeyFrameCloudDS.reset(new pcl::PointCloud<pcl::PointXYZINormal>());

        latestCornerKeyFrameCloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        latestSurfKeyFrameCloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        latestSurfKeyFrameCloudDS.reset(new pcl::PointCloud<pcl::PointXYZINormal>());

        kdtreeGlobalMap.reset(new pcl::KdTreeFLANN<pcl::PointXYZINormal>());
        globalMapKeyPoses.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        globalMapKeyPosesDS.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        globalMapKeyFrames.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        globalMapKeyFramesDS.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        // WILLIAM BEGIN
        globalMapCornerCloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        globalMapCornerCloudDS.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        globalMapSurfCloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        globalMapSurfCloudDS.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        // WILLIAM END


        timeLaserCloudCornerLast = 0;
        timeLaserCloudSurfLast = 0;
        timeLaserOdometry = 0;
        timeLaserCloudOutlierLast = 0;
        timeLastGloalMapPublish = 0;

        timeLastProcessing = -1;

        newLaserCloudCornerLast = false;
        newLaserCloudSurfLast = false;

        newLaserOdometry = false;
        newLaserCloudOutlierLast = false;

        for (int i = 0; i < 6; ++i){
            transformLast[i] = 0;
            transformSum[i] = 0;
            transformIncre[i] = 0;
            transformTobeMapped[i] = 0;
            transformBefMapped[i] = 0;
            transformAftMapped[i] = 0;
        }

        imuPointerFront = 0;
        imuPointerLast = -1;

        for (int i = 0; i < imuQueLength; ++i){
            imuTime[i] = 0;
            imuRoll[i] = 0;
            imuPitch[i] = 0;
        }

        gtsam::Vector Vector6(6);
        Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
        priorNoise = noiseModel::Diagonal::Variances(Vector6);
        odometryNoise = noiseModel::Diagonal::Variances(Vector6);

        matA0 = cv::Mat (5, 3, CV_32F, cv::Scalar::all(0));
        matB0 = cv::Mat (5, 1, CV_32F, cv::Scalar::all(-1));
        matX0 = cv::Mat (3, 1, CV_32F, cv::Scalar::all(0));

        matA1 = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));
        matD1 = cv::Mat (1, 3, CV_32F, cv::Scalar::all(0));
        matV1 = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));

        isDegenerate = false;
        matP = cv::Mat (6, 6, CV_32F, cv::Scalar::all(0));

        laserCloudCornerFromMapDSNum = 0;
        laserCloudSurfFromMapDSNum = 0;
        laserCloudCornerLastDSNum = 0;
        laserCloudSurfLastDSNum = 0;
        laserCloudOutlierLastDSNum = 0;
        laserCloudSurfTotalLastDSNum = 0;

        potentialLoopFlag = false;
        aLoopIsClosed = false;

        latestFrameID = 0;
    }

    void transformAssociateToMap()
    {
        float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) 
                 - sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
        float y1 = transformBefMapped[4] - transformSum[4];
        float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) 
                 + cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);

        float x2 = x1;
        float y2 = cos(transformSum[0]) * y1 + sin(transformSum[0]) * z1;
        float z2 = -sin(transformSum[0]) * y1 + cos(transformSum[0]) * z1;

        transformIncre[3] = cos(transformSum[2]) * x2 + sin(transformSum[2]) * y2;
        transformIncre[4] = -sin(transformSum[2]) * x2 + cos(transformSum[2]) * y2;
        transformIncre[5] = z2;

        float sbcx = sin(transformSum[0]);
        float cbcx = cos(transformSum[0]);
        float sbcy = sin(transformSum[1]);
        float cbcy = cos(transformSum[1]);
        float sbcz = sin(transformSum[2]);
        float cbcz = cos(transformSum[2]);

        float sblx = sin(transformBefMapped[0]);
        float cblx = cos(transformBefMapped[0]);
        float sbly = sin(transformBefMapped[1]);
        float cbly = cos(transformBefMapped[1]);
        float sblz = sin(transformBefMapped[2]);
        float cblz = cos(transformBefMapped[2]);

        float salx = sin(transformAftMapped[0]);
        float calx = cos(transformAftMapped[0]);
        float saly = sin(transformAftMapped[1]);
        float caly = cos(transformAftMapped[1]);
        float salz = sin(transformAftMapped[2]);
        float calz = cos(transformAftMapped[2]);

        float srx = -sbcx*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz)
                  - cbcx*sbcy*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                  - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                  - cbcx*cbcy*(calx*salz*(cblz*sbly - cbly*sblx*sblz) 
                  - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx);
        transformTobeMapped[0] = -asin(srx);

        float srycrx = sbcx*(cblx*cblz*(caly*salz - calz*salx*saly)
                     - cblx*sblz*(caly*calz + salx*saly*salz) + calx*saly*sblx)
                     - cbcx*cbcy*((caly*calz + salx*saly*salz)*(cblz*sbly - cbly*sblx*sblz)
                     + (caly*salz - calz*salx*saly)*(sbly*sblz + cbly*cblz*sblx) - calx*cblx*cbly*saly)
                     + cbcx*sbcy*((caly*calz + salx*saly*salz)*(cbly*cblz + sblx*sbly*sblz)
                     + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) + calx*cblx*saly*sbly);
        float crycrx = sbcx*(cblx*sblz*(calz*saly - caly*salx*salz)
                     - cblx*cblz*(saly*salz + caly*calz*salx) + calx*caly*sblx)
                     + cbcx*cbcy*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx)
                     + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) + calx*caly*cblx*cbly)
                     - cbcx*sbcy*((saly*salz + caly*calz*salx)*(cbly*sblz - cblz*sblx*sbly)
                     + (calz*saly - caly*salx*salz)*(cbly*cblz + sblx*sbly*sblz) - calx*caly*cblx*sbly);
        transformTobeMapped[1] = atan2(srycrx / cos(transformTobeMapped[0]), 
                                       crycrx / cos(transformTobeMapped[0]));
        
        float srzcrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                     - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                     - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                     - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                     + cbcx*sbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
        float crzcrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                     - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                     - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                     - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                     + cbcx*cbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
        transformTobeMapped[2] = atan2(srzcrx / cos(transformTobeMapped[0]), 
                                       crzcrx / cos(transformTobeMapped[0]));

        x1 = cos(transformTobeMapped[2]) * transformIncre[3] - sin(transformTobeMapped[2]) * transformIncre[4];
        y1 = sin(transformTobeMapped[2]) * transformIncre[3] + cos(transformTobeMapped[2]) * transformIncre[4];
        z1 = transformIncre[5];

        x2 = x1;
        y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
        z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

        transformTobeMapped[3] = transformAftMapped[3] 
                               - (cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2);
        transformTobeMapped[4] = transformAftMapped[4] - y2;
        transformTobeMapped[5] = transformAftMapped[5] 
                               - (-sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2);
    }

    void transformUpdate()
    {
		if (imuPointerLast >= 0) {
		    float imuRollLast = 0, imuPitchLast = 0;
		    while (imuPointerFront != imuPointerLast) {
		        if (timeLaserOdometry + scanPeriod < imuTime[imuPointerFront]) {
		            break;
		        }
		        imuPointerFront = (imuPointerFront + 1) % imuQueLength;
		    }

		    if (timeLaserOdometry + scanPeriod > imuTime[imuPointerFront]) {
		        imuRollLast = imuRoll[imuPointerFront];
		        imuPitchLast = imuPitch[imuPointerFront];
		    } else {
		        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
		        float ratioFront = (timeLaserOdometry + scanPeriod - imuTime[imuPointerBack]) 
		                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
		        float ratioBack = (imuTime[imuPointerFront] - timeLaserOdometry - scanPeriod) 
		                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

		        imuRollLast = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
		        imuPitchLast = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
		    }

		    transformTobeMapped[0] = 0.998 * transformTobeMapped[0] + 0.002 * imuPitchLast;
		    transformTobeMapped[2] = 0.998 * transformTobeMapped[2] + 0.002 * imuRollLast;
		  }

		for (int i = 0; i < 6; i++) {
		    transformBefMapped[i] = transformSum[i];
		    transformAftMapped[i] = transformTobeMapped[i];
		}
    }

    void updatePointAssociateToMapSinCos(){
        cRoll = cos(transformTobeMapped[0]);
        sRoll = sin(transformTobeMapped[0]);

        cPitch = cos(transformTobeMapped[1]);
        sPitch = sin(transformTobeMapped[1]);

        cYaw = cos(transformTobeMapped[2]);
        sYaw = sin(transformTobeMapped[2]);

        tX = transformTobeMapped[3];
        tY = transformTobeMapped[4];
        tZ = transformTobeMapped[5];
    }

    void pointAssociateToMap(pcl::PointXYZINormal const * const pi, pcl::PointXYZINormal * const po)
    {
        float x1 = cYaw * pi->x - sYaw * pi->y;
        float y1 = sYaw * pi->x + cYaw * pi->y;
        float z1 = pi->z;

        float x2 = x1;
        float y2 = cRoll * y1 - sRoll * z1;
        float z2 = sRoll * y1 + cRoll * z1;

        po->x = cPitch * x2 + sPitch * z2 + tX;
        po->y = y2 + tY;
        po->z = -sPitch * x2 + cPitch * z2 + tZ;
        po->intensity = pi->intensity;
    }

    void updateTransformPointCloudSinCos(PointTypePose *tIn){

        ctRoll = cos(tIn->roll);
        stRoll = sin(tIn->roll);

        ctPitch = cos(tIn->pitch);
        stPitch = sin(tIn->pitch);

        ctYaw = cos(tIn->yaw);
        stYaw = sin(tIn->yaw);

        tInX = tIn->x;
        tInY = tIn->y;
        tInZ = tIn->z;
    }

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr transformPointCloud(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloudIn){

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZINormal>());

        pcl::PointXYZINormal *pointFrom;
        pcl::PointXYZINormal pointTo;

        int cloudSize = cloudIn->points.size();
        cloudOut->resize(cloudSize);

        for (int i = 0; i < cloudSize; ++i){

            pointFrom = &cloudIn->points[i];
            float x1 = ctYaw * pointFrom->x - stYaw * pointFrom->y;
            float y1 = stYaw * pointFrom->x + ctYaw* pointFrom->y;
            float z1 = pointFrom->z;

            float x2 = x1;
            float y2 = ctRoll * y1 - stRoll * z1;
            float z2 = stRoll * y1 + ctRoll* z1;

            pointTo.x = ctPitch * x2 + stPitch * z2 + tInX;
            pointTo.y = y2 + tInY;
            pointTo.z = -stPitch * x2 + ctPitch * z2 + tInZ;
            pointTo.intensity = pointFrom->intensity;
            pointTo.curvature = pointFrom->curvature; // WILLIAM

            cloudOut->points[i] = pointTo;
        }
        return cloudOut;
    }

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr transformPointCloud(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloudIn, PointTypePose* transformIn){

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZINormal>());

        pcl::PointXYZINormal *pointFrom;
        pcl::PointXYZINormal pointTo;

        int cloudSize = cloudIn->points.size();
        cloudOut->resize(cloudSize);
        
        for (int i = 0; i < cloudSize; ++i){

            pointFrom = &cloudIn->points[i];
            float x1 = cos(transformIn->yaw) * pointFrom->x - sin(transformIn->yaw) * pointFrom->y;
            float y1 = sin(transformIn->yaw) * pointFrom->x + cos(transformIn->yaw)* pointFrom->y;
            float z1 = pointFrom->z;

            float x2 = x1;
            float y2 = cos(transformIn->roll) * y1 - sin(transformIn->roll) * z1;
            float z2 = sin(transformIn->roll) * y1 + cos(transformIn->roll)* z1;

            pointTo.x = cos(transformIn->pitch) * x2 + sin(transformIn->pitch) * z2 + transformIn->x;
            pointTo.y = y2 + transformIn->y;
            pointTo.z = -sin(transformIn->pitch) * x2 + cos(transformIn->pitch) * z2 + transformIn->z;
            pointTo.intensity = pointFrom->intensity;
            pointTo.curvature = pointFrom->curvature; // WILLIAM

            cloudOut->points[i] = pointTo;
        }
        return cloudOut;
    }

    void laserCloudOutlierLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudOutlierLast = msg->header.stamp.toSec();
        laserCloudOutlierLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudOutlierLast);
        newLaserCloudOutlierLast = true;
        // std::cout<<"###### map optimization   ->    laserCloudOutlierLastHandler" <<std::endl;
    }

    void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudCornerLast = msg->header.stamp.toSec();
        laserCloudCornerLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudCornerLast);
        newLaserCloudCornerLast = true;
        // std::cout<<"###### map optimization   ->    laserCloudCornerLastHandler" <<std::endl;
        // WILLIAM BEGIN
        if (!savedOneFrameRawData){
            std::ofstream outfile;
            outfile.open("/home/william/data/LeGO-LOAM/oneFrameRawData_4");
            outfile << *msg << std::endl;
            outfile.close();
            pcl::io::savePCDFile("/home/william/data/LeGO-LOAM/oneFrameRawData_4.pcd", *laserCloudCornerLast, false);
            savedOneFrameRawData = true;            
        }
        // WILLIAM END
    }

    void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudSurfLast = msg->header.stamp.toSec();
        laserCloudSurfLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudSurfLast);
        newLaserCloudSurfLast = true;
        // std::cout<<"###### map optimization   ->    laserCloudSurfLastHandler" <<std::endl;
    }

    void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry){
        timeLaserOdometry = laserOdometry->header.stamp.toSec();
        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;
        tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);
        transformSum[0] = -pitch;
        transformSum[1] = -yaw;
        transformSum[2] = roll;
        transformSum[3] = laserOdometry->pose.pose.position.x;
        transformSum[4] = laserOdometry->pose.pose.position.y;
        transformSum[5] = laserOdometry->pose.pose.position.z;
        newLaserOdometry = true;
        // std::cout<<"###### map optimization   ->    laserOdometryHandler" <<std::endl;
    }

    void GPSOdomeHandler(const nav_msgs::Odometry::ConstPtr& GPSOdometry){
        // timeGPSOdometry = GPSOdometry->header.stamp.toSec();
        // std::cout<<"here1"<<std::endl;
        
        latitude = (GPSOdometry->pose.pose.position.x)*1000000;
        longitude = (GPSOdometry->pose.pose.position.y)*1000000;
        altitude = (GPSOdometry->pose.pose.position.z)*10;
        // std::cout<<"GPS info latitude, longitude, altitude:"<<latitude<<","<<longitude<<","<<altitude<<std::endl;
        currentGPSPoint.x = latitude;
        currentGPSPoint.y = longitude;
        currentGPSPoint.z = altitude;
        currentGPSPoint.intensity = 1;
    }

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn){
        double roll, pitch, yaw;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(imuIn->orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        imuPointerLast = (imuPointerLast + 1) % imuQueLength;
        imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
        imuRoll[imuPointerLast] = roll;
        imuPitch[imuPointerLast] = pitch;
    }

    void publishTF(){

        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                                  (transformAftMapped[2], -transformAftMapped[0], -transformAftMapped[1]);

        odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        odomAftMapped.pose.pose.orientation.x = -geoQuat.y;
        odomAftMapped.pose.pose.orientation.y = -geoQuat.z;
        odomAftMapped.pose.pose.orientation.z = geoQuat.x;
        odomAftMapped.pose.pose.orientation.w = geoQuat.w;
        odomAftMapped.pose.pose.position.x = transformAftMapped[3];
        odomAftMapped.pose.pose.position.y = transformAftMapped[4];
        odomAftMapped.pose.pose.position.z = transformAftMapped[5];
        odomAftMapped.twist.twist.angular.x = transformBefMapped[0];
        odomAftMapped.twist.twist.angular.y = transformBefMapped[1];
        odomAftMapped.twist.twist.angular.z = transformBefMapped[2];
        odomAftMapped.twist.twist.linear.x = transformBefMapped[3];
        odomAftMapped.twist.twist.linear.y = transformBefMapped[4];
        odomAftMapped.twist.twist.linear.z = transformBefMapped[5];
        pubOdomAftMapped.publish(odomAftMapped);

        aftMappedTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
        aftMappedTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        aftMappedTrans.setOrigin(tf::Vector3(transformAftMapped[3], transformAftMapped[4], transformAftMapped[5]));
        tfBroadcaster.sendTransform(aftMappedTrans);
    }

    void publishKeyPosesAndFrames(){

        if (pubKeyPoses.getNumSubscribers() != 0){
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*cloudKeyPoses3D, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubKeyPoses.publish(cloudMsgTemp);
        }

        if (pubRecentKeyFrames.getNumSubscribers() != 0){
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*laserCloudSurfFromMapDS, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubRecentKeyFrames.publish(cloudMsgTemp);
        }
    }

    void visualizeGlobalMapThread(){
        // ros::Rate rate(0.2);
        ros::Rate rate(1);
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }
    }

    void publishGlobalMap(){
        
        // WILLIAM BEGIN
        if (pubLaserCloudSurround.getNumSubscribers() == 0 && pubLaserCloudCornerSurround.getNumSubscribers() == 0 && pubLaserCloudSurfSurround.getNumSubscribers() == 0)
            return;
        // WILLIAM END

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;

        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(currentRobotPosPoint, globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < pointSearchIndGlobalMap.size(); ++i)
          globalMapKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);

        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        // WILLIAM BEGIN
        if(!mappingDSFEnableFlag){
            *globalMapKeyPosesDS = *globalMapKeyPoses;
        }
        // WILLIAM END

        for (int i = 0; i < globalMapKeyPosesDS->points.size(); ++i){
			int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
			*globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],   &cloudKeyPoses6D->points[thisKeyInd]);
			*globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
			*globalMapKeyFrames += *transformPointCloud(outlierCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
            // WILLIAM BEGIN
            *globalMapCornerCloud += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],   &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapSurfCloud += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapSurfCloud += *transformPointCloud(outlierCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
            // WILLIAM END
        }

        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        // WILLIAM BEGIN
        if(!mappingDSFEnableFlag){
            *globalMapKeyFramesDS = *globalMapKeyFrames;
        }
        // WILLIAM END

        // WILLIAM BEGIN
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapCornerCloud);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapCornerCloudDS);
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapSurfCloud);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapSurfCloudDS);
        // WILLIAM END
 
        sensor_msgs::PointCloud2 cloudMsgTemp;
        pcl::toROSMsg(*globalMapKeyFramesDS, cloudMsgTemp);
        cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        cloudMsgTemp.header.frame_id = "/camera_init";
        pubLaserCloudSurround.publish(cloudMsgTemp);  

        // WILLIAM BEGIN
        sensor_msgs::PointCloud2 cloudCornerMsgTemp; 
        pcl::toROSMsg(*globalMapCornerCloud, cloudCornerMsgTemp);
        cloudCornerMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        cloudCornerMsgTemp.header.frame_id = "/camera_init";
        pubLaserCloudCornerSurround.publish(cloudCornerMsgTemp);
        
        sensor_msgs::PointCloud2 cloudSurfMsgTemp; 
        pcl::toROSMsg(*globalMapSurfCloud, cloudSurfMsgTemp);
        cloudSurfMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        cloudSurfMsgTemp.header.frame_id = "/camera_init";
        pubLaserCloudSurfSurround.publish(cloudSurfMsgTemp);
        // WILLIAM END

        // WILLIAM BEGIN
        globalMapKeyFramesDSNum = globalMapKeyFramesDS->points.size();
        // std::cout << "globalMapKeyFramesDSNum = " << globalMapKeyFramesDSNum << std::endl;
        globalMapCornerCloudDSNum = globalMapCornerCloudDS->points.size();
        // std::cout << "globalMapCornerCloudDSNum = " << globalMapCornerCloudDSNum << std::endl;
        globalMapSurfCloudDSNum = globalMapSurfCloudDS->points.size();
        // std::cout << "globalMapSurfCloudDSNum = " << globalMapSurfCloudDSNum << std::endl;
        // WILLIAM END

        // WILLIAM BEGIN
        saveMap.write("/home/william/data/LeGO-LOAM/temp/savedMap.pcd", *globalMapKeyFrames);

        pcl::io::savePCDFile("/home/william/data/LeGO-LOAM/temp/PCD_corner_DS.pcd", *globalMapCornerCloudDS, false);
        pcl::io::savePCDFile("/home/william/data/LeGO-LOAM/temp/PCD_surf_DS.pcd", *globalMapSurfCloudDS, false);

        pcl::io::savePCDFile("/home/william/data/LeGO-LOAM/temp/PCD_corner.pcd", *globalMapCornerCloud, false);
        pcl::io::savePCDFile("/home/william/data/LeGO-LOAM/temp/PCD_surf.pcd", *globalMapSurfCloud, false);

        pcl::io::savePCDFile("/home/william/data/LeGO-LOAM/temp/PCD_all_DS.pcd", *globalMapKeyFramesDS, false);
        pcl::io::savePCDFile("/home/william/data/LeGO-LOAM/temp/PCD_all.pcd", *globalMapKeyFrames, false);


        globalMapKeyPoses->clear();
        globalMapKeyPosesDS->clear();
        globalMapKeyFrames->clear();
        globalMapKeyFramesDS->clear();

        globalMapCornerCloud->clear();
        globalMapCornerCloudDS->clear();
        globalMapSurfCloud->clear();
        globalMapSurfCloudDS->clear();

        globalMapKeyFramesDSNum = globalMapKeyFramesDS->points.size();
        // std::cout << "globalMapKeyFramesDSNum after clear = " << globalMapKeyFramesDSNum << std::endl;
        // WILLIAM END   
    }

    void loopClosureThread(){

        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(3);
        while (ros::ok()){
            rate.sleep();
            performLoopClosure();
        }
    }

    bool detectLoopClosure(){

        latestSurfKeyFrameCloud->clear();
        nearHistorySurfKeyFrameCloud->clear();
        nearHistorySurfKeyFrameCloudDS->clear();

        std::lock_guard<std::mutex> lock(mtx);

        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(currentRobotPosPoint, historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        
        // WILLIAM BEGIN
        // std::cout << "----------------------------------------------------------------------" << endl;
        // std::cout << "currentRobotPosPoint = " << currentRobotPosPoint << endl;
        // std::cout << "----------------------------------------------------------------------" << endl;
        // WILLIAM END
        // WILLIAM BEGIN
        // What if it's actually a loop but the time is less than 50?
        // That will lead to a wrong match
        closestHistoryFrameID = -1;
        for (int i = 0; i < pointSearchIndLoop.size(); ++i){
            int id = pointSearchIndLoop[i];
            if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) > 50){
                closestHistoryFrameID = id;
                break;
            }
        }
        // WILLIAM END

        if (closestHistoryFrameID == -1){
            //todo
            // std::vector<int> GPSpointSearchIndLoop;
            // std::vector<float> GPSpointSearchSqDisLoop;
            // kdtreeHistoryGPS->setInputCloud(cloudGPS3D);
            // kdtreeHistoryGPS->radiusSearch(currentGPSPoint , GPSSearchRadius, GPSpointSearchIndLoop, GPSpointSearchSqDisLoop, 0);
            
            // K nearest neighbor search，进行k最近邻搜索
            
            kdtreeHistoryGPS->setInputCloud(cloudGPS3D);
            // std::cout<<"setInputCloud(cloudGPS3D) size:"<<cloudGPS3D->points.size()<<std::endl;
            // std::cout<<"currentGPSPoint x y z :"<<currentGPSPoint.x<<","<<currentGPSPoint.y<<","<<currentGPSPoint.z<<std::endl;
		    // int GPSsearchKpoint = 5;                               //设搜索一个点，即：K为1.
		    std::vector<int> pointIdxNKNSearch(GPSsearchKpoint);   //设置一个类似数组的东西，（我对于C++中vector的理解直接把它当成了数组），pointIdxNKNSearch（K）的长度为K，用于存放点索引
		    std::vector<float> pointNKNSquaredDistance(GPSsearchKpoint);//同样一个长度为K的数组pointNKNSquaredDistance，用于存放距离的平方
            if (kdtreeHistoryGPS->nearestKSearch(currentGPSPoint, GPSsearchKpoint, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)//前面定义了kdtree作为搜索对象，然后就可以用它来调用系统的nearestKSearch函数了，注意后面的参数，分别是上面已经定义好的。
		    {
			    for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i){
                    // std::cout << "    " << cloudGPS3D->points[pointIdxNKNSearch[i]].x
			    	// << " " << cloudGPS3D->points[pointIdxNKNSearch[i]].y
			    	// << " " << cloudGPS3D->points[pointIdxNKNSearch[i]].z
			    	// << " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
                    int id = pointIdxNKNSearch[i];
                    if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) > 50){
                        closestHistoryFrameID = id;
                        std::cout<<"closestHistoryFrameID from GPS:"<<closestHistoryFrameID<<std::endl;
                        PoseLoopCloseDetectFlag = true;
                        break;
                    }
                }    	
		    }

            // for (int i = 0; i < GPSpointSearchIndLoop.size(); ++i){
            //     int id = GPSpointSearchIndLoop[i];
            //     if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) > 50){
            //         closestHistoryFrameID = id;
            //         std::cout<<"closestHistoryFrameID from GPS:"<<closestHistoryFrameID<<std::endl;
            //         break;
            //     }
            // }
            if (closestHistoryFrameID == -1){
                return false;
            }
        }else{
            std::cout<<"closestHistoryFrameID from Pose:"<<closestHistoryFrameID<<std::endl;
        }

        if(!PoseLoopCloseDetectFlag){
            latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;
        }else{
            latestFrameIDLoopCloure = cloudGPS3D->points.size() - 1;
        }
        std::cout<<"latestFrameIDLoopCloure:"<<latestFrameIDLoopCloure<<std::endl;
        // latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;
        *latestSurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
        *latestSurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[latestFrameIDLoopCloure],   &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr hahaCloud(new pcl::PointCloud<pcl::PointXYZINormal>());
        int cloudSize = latestSurfKeyFrameCloud->points.size();
        for (int i = 0; i < cloudSize; ++i){
            if ((int)latestSurfKeyFrameCloud->points[i].intensity >= 0){
                hahaCloud->push_back(latestSurfKeyFrameCloud->points[i]);
            }
        }
        latestSurfKeyFrameCloud->clear();
        *latestSurfKeyFrameCloud   = *hahaCloud;

        std::cout<<"latestSurfKeyFrameCloud size:"<<latestSurfKeyFrameCloud->points.size()<<std::endl;

        for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j){
            if (closestHistoryFrameID + j < 0 || closestHistoryFrameID + j > latestFrameIDLoopCloure)
                continue;
            *nearHistorySurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[closestHistoryFrameID+j], &cloudKeyPoses6D->points[closestHistoryFrameID+j]);
            *nearHistorySurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[closestHistoryFrameID+j],   &cloudKeyPoses6D->points[closestHistoryFrameID+j]);
        }

        downSizeFilterHistoryKeyFrames.setInputCloud(nearHistorySurfKeyFrameCloud);
        downSizeFilterHistoryKeyFrames.filter(*nearHistorySurfKeyFrameCloudDS);
        // WILLIAM BEGIN
        if(!mappingDSFEnableFlag){
            *nearHistorySurfKeyFrameCloudDS = *nearHistorySurfKeyFrameCloud;
        }
        // WILLIAM END

        std::cout<<"nearHistorySurfKeyFrameCloudDS size:"<<nearHistorySurfKeyFrameCloudDS->points.size()<<std::endl;

        if (pubHistoryKeyFrames.getNumSubscribers() != 0){
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*nearHistorySurfKeyFrameCloudDS, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubHistoryKeyFrames.publish(cloudMsgTemp);
        }
        
        if (pubGPSFrames.getNumSubscribers() != 0){
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*cloudGPS3D, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubGPSFrames.publish(cloudMsgTemp);
        }

        return true;
    }


    void performLoopClosure(){

        // std::cout<<"###### map optmization -> perform loop closure ->" <<std::endl;
        // std::cout<<"###### cloudKeyPoses3D->points.empty() :"<<cloudKeyPoses3D->points.empty() <<std::endl;
        // std::cout<<"###### potentialLoopFlag :"<<potentialLoopFlag <<std::endl;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        if (potentialLoopFlag == false){
            if (detectLoopClosure() == true){
		        if (abs(timeSaveFirstCurrentScanForLoopClosure - timeLaserOdometry) > 30){
		            std::cout<<"#----------------A NEW LOOPCLOSURE----------------#" <<std::endl;
	                FitnessScore = historyKeyframeFitnessScore;
		        }
                
                potentialLoopFlag = true;
                timeSaveFirstCurrentScanForLoopClosure = timeLaserOdometry;
            }
            if (potentialLoopFlag == false)
                return;
        }

        potentialLoopFlag = false;

        pcl::IterativeClosestPoint<pcl::PointXYZINormal, pcl::PointXYZINormal> icp;
        // TODO
        // icp.setMaxCorrespondenceDistance(100);
        // icp.setMaximumIterations(100);
        icp.setMaxCorrespondenceDistance(50);
        icp.setMaximumIterations(150);

        icp.setTransformationEpsilon(1e-6);
        // icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(0.01);
        icp.setRANSACIterations(0);

        icp.setInputSource(latestSurfKeyFrameCloud);
        icp.setInputTarget(nearHistorySurfKeyFrameCloudDS);
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr unused_result(new pcl::PointCloud<pcl::PointXYZINormal>());
        icp.align(*unused_result);

        std::cout<<"      icp.getFitnessScore():"<<icp.getFitnessScore()<<std::endl;

	    //float deltaz = sqrt(cloudKeyPoses3D->points[closestHistoryFrameID].y - cloudKeyPoses3D->points[latestFrameIDLoopCloure].y);
        if (icp.hasConverged() == false || icp.getFitnessScore() >= FitnessScore)
            return;
	    FitnessScore = icp.getFitnessScore();
        if(FitnessScore<0.3) 
            FitnessScore = 0.3;
        
        if (pubIcpKeyFrames.getNumSubscribers() != 0){
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr closed_cloud(new pcl::PointCloud<pcl::PointXYZINormal>());
            pcl::transformPointCloud (*latestSurfKeyFrameCloud, *closed_cloud, icp.getFinalTransformation());
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*closed_cloud, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubIcpKeyFrames.publish(cloudMsgTemp);
        }   

        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionCameraFrame;
        correctionCameraFrame = icp.getFinalTransformation();
        pcl::getTranslationAndEulerAngles(correctionCameraFrame, x, y, z, roll, pitch, yaw);
        Eigen::Affine3f correctionLidarFrame = pcl::getTransformation(z, x, y, yaw, roll, pitch);
        Eigen::Affine3f tWrong = pclPointToAffine3fCameraToLidar(cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(cloudKeyPoses6D->points[closestHistoryFrameID]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        std::lock_guard<std::mutex> lock(mtx);
        gtSAMgraph.add(BetweenFactor<Pose3>(latestFrameIDLoopCloure, closestHistoryFrameID, poseFrom.between(poseTo), constraintNoise));
        isam->update(gtSAMgraph);
        isam->update();
        gtSAMgraph.resize(0);

        aLoopIsClosed = true;
    }

    Pose3 pclPointTogtsamPose3(PointTypePose thisPoint){
    	return Pose3(Rot3::RzRyRx(double(thisPoint.yaw), double(thisPoint.roll), double(thisPoint.pitch)),
                           Point3(double(thisPoint.z),   double(thisPoint.x),    double(thisPoint.y)));
    }

    Eigen::Affine3f pclPointToAffine3fCameraToLidar(PointTypePose thisPoint){
    	return pcl::getTransformation(thisPoint.z, thisPoint.x, thisPoint.y, thisPoint.yaw, thisPoint.roll, thisPoint.pitch);
    }

    void extractSurroundingKeyFrames(){

        if (cloudKeyPoses3D->points.empty() == true)
            return;	
		
		if (loopClosureEnableFlag == true){
            if (recentCornerCloudKeyFrames.size() < surroundingKeyframeSearchNum){
                recentCornerCloudKeyFrames. clear();
                recentSurfCloudKeyFrames.   clear();
                recentOutlierCloudKeyFrames.clear();
                int numPoses = cloudKeyPoses3D->points.size();
                for (int i = numPoses-1; i >= 0; --i){
                    int thisKeyInd = (int)cloudKeyPoses3D->points[i].intensity;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
                    updateTransformPointCloudSinCos(&thisTransformation);
                    recentCornerCloudKeyFrames. push_front(transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
                    recentSurfCloudKeyFrames.   push_front(transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
                    recentOutlierCloudKeyFrames.push_front(transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
                    if (recentCornerCloudKeyFrames.size() >= surroundingKeyframeSearchNum)
                        break;
                }
            }else{
                if (latestFrameID != cloudKeyPoses3D->points.size() - 1){

                    recentCornerCloudKeyFrames. pop_front();
                    recentSurfCloudKeyFrames.   pop_front();
                    recentOutlierCloudKeyFrames.pop_front();
                    latestFrameID = cloudKeyPoses3D->points.size() - 1;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[latestFrameID];
                    updateTransformPointCloudSinCos(&thisTransformation);
                    recentCornerCloudKeyFrames. push_back(transformPointCloud(cornerCloudKeyFrames[latestFrameID]));
                    recentSurfCloudKeyFrames.   push_back(transformPointCloud(surfCloudKeyFrames[latestFrameID]));
                    recentOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames[latestFrameID]));
                }
            }

            for (int i = 0; i < recentCornerCloudKeyFrames.size(); ++i){
                *laserCloudCornerFromMap += *recentCornerCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *recentSurfCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *recentOutlierCloudKeyFrames[i];
            }
		}else{
            surroundingKeyPoses->clear();
            surroundingKeyPosesDS->clear();

			kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);
			kdtreeSurroundingKeyPoses->radiusSearch(currentRobotPosPoint, (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis, 0);
            
            // WILLIAM BEGIN
            // std::cout << "--------------------------------------------------" << endl;
            // std::cout << "currentRobotPosPoint = " << currentRobotPosPoint << endl;
            // std::cout << "--------------------------------------------------" << endl;
            // WILLIAM END
			
            for (int i = 0; i < pointSearchInd.size(); ++i)
                surroundingKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchInd[i]]);
			downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
			downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

            int numSurroundingPosesDS = surroundingKeyPosesDS->points.size();
            for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i){
                bool existingFlag = false;
                for (int j = 0; j < numSurroundingPosesDS; ++j){
                    if (surroundingExistingKeyPosesID[i] == (int)surroundingKeyPosesDS->points[j].intensity){
                        existingFlag = true;
                        break;
                    }
                }
                if (existingFlag == false){
                    surroundingExistingKeyPosesID.   erase(surroundingExistingKeyPosesID.   begin() + i);
                    surroundingCornerCloudKeyFrames. erase(surroundingCornerCloudKeyFrames. begin() + i);
                    surroundingSurfCloudKeyFrames.   erase(surroundingSurfCloudKeyFrames.   begin() + i);
                    surroundingOutlierCloudKeyFrames.erase(surroundingOutlierCloudKeyFrames.begin() + i);
                    --i;
                }
            }

            for (int i = 0; i < numSurroundingPosesDS; ++i) {
                bool existingFlag = false;
                for (auto iter = surroundingExistingKeyPosesID.begin(); iter != surroundingExistingKeyPosesID.end(); ++iter){
                    if ((*iter) == (int)surroundingKeyPosesDS->points[i].intensity){
                        existingFlag = true;
                        break;
                    }
                }
                if (existingFlag == true){
                    continue;
                }else{
                    int thisKeyInd = (int)surroundingKeyPosesDS->points[i].intensity;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
                    updateTransformPointCloudSinCos(&thisTransformation);
                    surroundingExistingKeyPosesID.   push_back(thisKeyInd);
                    surroundingCornerCloudKeyFrames. push_back(transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
                    surroundingSurfCloudKeyFrames.   push_back(transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
                    surroundingOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
                }
            }

            for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i) {
                *laserCloudCornerFromMap += *surroundingCornerCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *surroundingSurfCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *surroundingOutlierCloudKeyFrames[i];
            }
		}

        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->points.size();

        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->points.size();

        // WILLIAM BEGIN
        if(!mappingDSFEnableFlag){
            *laserCloudCornerFromMapDS = *laserCloudCornerFromMap;
            *laserCloudSurfFromMapDS = *laserCloudSurfFromMap; 
        }
        // WILLIAM END
    }

    void downsampleCurrentScan(){

        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        // WILLIAM BEGIN
        if(!mappingDSFEnableFlag){
            *laserCloudCornerLastDS = *laserCloudCornerLast;
        }
        // WILLIAM END
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->points.size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        // WILLIAM BEGIN
        if(!mappingDSFEnableFlag){
            *laserCloudSurfLastDS = *laserCloudSurfLast;
        }
        // WILLIAM END
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->points.size();
	//std::cout<<"##### num of lasercloudsurf = "<< laserCloudSurfLastDSNum <<std::endl;

        laserCloudOutlierLastDS->clear();
        downSizeFilterOutlier.setInputCloud(laserCloudOutlierLast);
        downSizeFilterOutlier.filter(*laserCloudOutlierLastDS);
        // WILLIAM BEGIN
        if(!mappingDSFEnableFlag){
            *laserCloudOutlierLastDS = *laserCloudOutlierLast;
        }
        // WILLIAM END
        laserCloudOutlierLastDSNum = laserCloudOutlierLastDS->points.size();

        laserCloudSurfTotalLast->clear();
        laserCloudSurfTotalLastDS->clear();
        *laserCloudSurfTotalLast += *laserCloudSurfLastDS;
        *laserCloudSurfTotalLast += *laserCloudOutlierLastDS;
        downSizeFilterSurf.setInputCloud(laserCloudSurfTotalLast);
        downSizeFilterSurf.filter(*laserCloudSurfTotalLastDS);
        laserCloudSurfTotalLastDSNum = laserCloudSurfTotalLastDS->points.size();
	//std::cout<<"##### num of lasercloudcorner = "<< laserCloudSurfTotalLastDSNum <<std::endl;
	float ratio = laserCloudSurfTotalLastDSNum * 1.0 / (laserCloudCornerLastDSNum + laserCloudSurfTotalLastDSNum);
	//std::cout<<"##### ratio of lasercloudsurf = "<< ratio <<std::endl;
    }

    void cornerOptimization(int iterCount){

        updatePointAssociateToMapSinCos();
        for (int i = 0; i < laserCloudCornerLastDSNum; i++) {
            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
            
            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                    * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                    * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                                    * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1) {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

    void surfOptimization(int iterCount){
        updatePointAssociateToMapSinCos();
        for (int i = 0; i < laserCloudSurfTotalLastDSNum; i++) {
            pointOri = laserCloudSurfTotalLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0.at<float>(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0.at<float>(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0.at<float>(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }
                cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);

                float pa = matX0.at<float>(0, 0);
                float pb = matX0.at<float>(1, 0);
                float pc = matX0.at<float>(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

    bool LMOptimization(int iterCount){
        float srx = sin(transformTobeMapped[0]);
        float crx = cos(transformTobeMapped[0]);
        float sry = sin(transformTobeMapped[1]);
        float cry = cos(transformTobeMapped[1]);
        float srz = sin(transformTobeMapped[2]);
        float crz = cos(transformTobeMapped[2]);

        int laserCloudSelNum = laserCloudOri->points.size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        for (int i = 0; i < laserCloudSelNum; i++) {
            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = coeff.x;
            matA.at<float>(i, 4) = coeff.y;
            matA.at<float>(i, 5) = coeff.z;
            matB.at<float>(i, 0) = -coeff.intensity;
        }
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {
            return true;
        }
        return false;
    }

    void scan2MapOptimization(){

        if (laserCloudCornerFromMapDSNum > 10 && laserCloudSurfFromMapDSNum > 100) {

            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            for (int iterCount = 0; iterCount < 10; iterCount++) {

                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization(iterCount);
                surfOptimization(iterCount);

                if (LMOptimization(iterCount) == true)
                    break;              
            }

            transformUpdate();
        }
    }


    void saveKeyFramesAndFactor(){

        currentRobotPosPoint.x = transformAftMapped[3];
        currentRobotPosPoint.y = transformAftMapped[4];
        currentRobotPosPoint.z = transformAftMapped[5];
        // currentGPSPoint.x = latitude;
        // currentGPSPoint.y = longitude;
        // currentGPSPoint.z = altitude;
        // currentGPSPoint.intensity = 1;

        bool saveThisKeyFrame = true;
        if (sqrt((previousRobotPosPoint.x-currentRobotPosPoint.x)*(previousRobotPosPoint.x-currentRobotPosPoint.x)
                +(previousRobotPosPoint.y-currentRobotPosPoint.y)*(previousRobotPosPoint.y-currentRobotPosPoint.y)
                +(previousRobotPosPoint.z-currentRobotPosPoint.z)*(previousRobotPosPoint.z-currentRobotPosPoint.z)) < 0.3){
            saveThisKeyFrame = false;
        }

        //std::cout<<"######  saveThisKeyFrame = :"<<saveThisKeyFrame<<std::endl;

        if (saveThisKeyFrame == false && !cloudKeyPoses3D->points.empty())
        	return;
	        //std::cout<<"###### lastpose.x = :"<<previousRobotPosPoint.x;
	        //std::cout<<"       lastpose.y = :"<<previousRobotPosPoint.y;
	        //std::cout<<"       lastpose.z = :"<<previousRobotPosPoint.z<<std::endl;

        previousRobotPosPoint = currentRobotPosPoint;

	    //std::cout<<"###### curpose.x = :"<<previousRobotPosPoint.x;
	    //std::cout<<"       curpose.y = :"<<previousRobotPosPoint.y<<std::endl;
	    //std::cout<<"       curpose.z = :"<<previousRobotPosPoint.z<<std::endl;

        if (cloudKeyPoses3D->points.empty()){
            gtSAMgraph.add(PriorFactor<Pose3>(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                                       		 Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])), priorNoise));
            initialEstimate.insert(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                                  Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])));
            for (int i = 0; i < 6; ++i)
            	transformLast[i] = transformTobeMapped[i];
        }
        else{
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(transformLast[2], transformLast[0], transformLast[1]),
                                                Point3(transformLast[5], transformLast[3], transformLast[4]));
            gtsam::Pose3 poseTo   = Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                                                Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4]));
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->points.size()-1, cloudKeyPoses3D->points.size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->points.size(), Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                                                                     		   Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4])));
        }

        isam->update(gtSAMgraph, initialEstimate);
        isam->update();
        

        gtSAMgraph.resize(0);
	    initialEstimate.clear();

        pcl::PointXYZINormal thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;
        //todo GPS
        pcl::PointXYZI thisGPS3D;
        thisGPS3D.x = latitude;
        thisGPS3D.y = longitude;
        thisGPS3D.z = altitude;

        isamCurrentEstimate = isam->calculateEstimate();
        //std::cout<<"###### mp -> correctPoses()-> numPoses = isamCurrentEstimate.size():"<<isamCurrentEstimate.size()<<std::endl;
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);

        thisPose3D.x = latestEstimate.translation().y();
        thisPose3D.y = latestEstimate.translation().z();
        thisPose3D.z = latestEstimate.translation().x();
        thisPose3D.intensity = cloudKeyPoses3D->points.size();
        //todo
        thisGPS3D.intensity = thisPose3D.intensity;
        cloudGPS3D->push_back(thisGPS3D);
        cloudKeyPoses3D->push_back(thisPose3D);

        // std::cout<<"cloudGPS "<<cloudGPS3D->points.size()<<std::endl;

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity;
        thisPose6D.roll  = latestEstimate.rotation().pitch();
        thisPose6D.pitch = latestEstimate.rotation().yaw();
        thisPose6D.yaw   = latestEstimate.rotation().roll();
        thisPose6D.time = timeLaserOdometry;
        cloudKeyPoses6D->push_back(thisPose6D);
	    //std::cout<<"###### mp -> correctPoses()-> numPoses = cloudKeyPoses6D->points.size():"<<cloudKeyPoses6D->points.size()<<std::endl;

        if (cloudKeyPoses3D->points.size() > 1){
            transformAftMapped[0] = latestEstimate.rotation().pitch();
            transformAftMapped[1] = latestEstimate.rotation().yaw();
            transformAftMapped[2] = latestEstimate.rotation().roll();
            transformAftMapped[3] = latestEstimate.translation().y();
            transformAftMapped[4] = latestEstimate.translation().z();
            transformAftMapped[5] = latestEstimate.translation().x();

            for (int i = 0; i < 6; ++i){
            	transformLast[i] = transformAftMapped[i];
            	transformTobeMapped[i] = transformAftMapped[i];
            }
        }

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr thisCornerKeyFrame(new pcl::PointCloud<pcl::PointXYZINormal>());
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr thisSurfKeyFrame(new pcl::PointCloud<pcl::PointXYZINormal>());
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr thisOutlierKeyFrame(new pcl::PointCloud<pcl::PointXYZINormal>());

        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);
        pcl::copyPointCloud(*laserCloudOutlierLastDS, *thisOutlierKeyFrame);

        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);
        outlierCloudKeyFrames.push_back(thisOutlierKeyFrame);
    }

    void correctPoses(){
    	if (aLoopIsClosed == true){
            std::cout<<"###### mp -> correctPoses()-> a loop"<<std::endl;
            recentCornerCloudKeyFrames. clear();
            recentSurfCloudKeyFrames.   clear();
            recentOutlierCloudKeyFrames.clear();

            int numPoses = isamCurrentEstimate.size();
            //std::cout<<"###### mp -> correctPoses()-> numPoses = isamCurrentEstimate.size():"<<numPoses<<std::endl;
			for (int i = 0; i < numPoses; ++i)
			{
				cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().y();
				cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().z();
				cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().x();

				cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
	            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
	            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
	            cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
	            cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();
	            cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
			}

	    	aLoopIsClosed = false;
	    }
    }

    void clearCloud(){
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();  
        laserCloudCornerFromMapDS->clear();
        laserCloudSurfFromMapDS->clear();   
    }

    void run(){
        //std::cout<<"###### start a run() in map optmization"<<std::endl;
        //std::cout<<"###### first condition-----------------"<<std::endl;
        //std::cout<<"###### newLaserCloudCornerLast: "<<newLaserCloudCornerLast<<std::endl;
        //std::cout<<"###### newLaserCloudSurfLast:"<<newLaserCloudSurfLast<<std::endl;
        //std::cout<<"###### newLaserCloudOutlierLast: "<<newLaserCloudOutlierLast<<std::endl;
        //std::cout<<"###### newLaserOdometry:"<<newLaserOdometry<<std::endl;
        //std::cout<<"###### timeLaserCloudCornerLast:"<<timeLaserCloudCornerLast<<std::endl;
        //std::cout<<"###### timeLaserCloudSurfLast:"<<timeLaserCloudSurfLast<<std::endl;
        //std::cout<<"###### timeLaserCloudOutlierLast:"<<timeLaserCloudOutlierLast<<std::endl;
        //std::cout<<"###### timeLaserOdometry:"<<timeLaserOdometry<<std::endl;
        //std::cout<<"###### --------------------------------"<<std::endl;
        // if(newLaserCloudCornerLast && newLaserCloudSurfLast && newLaserCloudOutlierLast && newLaserOdometry){
        //     std::cout<<"###### timeLaserCloudCornerLast:"<<timeLaserCloudCornerLast<<std::endl;
        //     std::cout<<"###### timeLaserCloudSurfLast:"<<timeLaserCloudSurfLast<<std::endl;
        //     std::cout<<"###### timeLaserCloudOutlierLast:"<<timeLaserCloudOutlierLast<<std::endl;
        //     std::cout<<"###### timeLaserOdometry:"<<timeLaserOdometry<<std::endl;
        // }
	

        if (newLaserCloudCornerLast  && std::abs(timeLaserCloudCornerLast  - timeLaserOdometry) < 0.005 &&
            newLaserCloudSurfLast    && std::abs(timeLaserCloudSurfLast    - timeLaserOdometry) < 0.005 &&
            newLaserCloudOutlierLast && std::abs(timeLaserCloudOutlierLast - timeLaserOdometry) < 0.005 &&
            newLaserOdometry)
        {
            // std::cout<<"###### 1"<<std::endl;

            newLaserCloudCornerLast = false; newLaserCloudSurfLast = false; newLaserCloudOutlierLast = false; newLaserOdometry = false;

            std::lock_guard<std::mutex> lock(mtx);

            //std::cout<<"###### second condition----------------"<<std::endl;
            //std::cout<<"###### timeLastProcessing:"<<timeLastProcessing<<std::endl;
            //std::cout<<"###### mappingProcessInterval:"<<mappingProcessInterval<<std::endl;
            //std::cout<<"###### timeLaserOdometry - timeLastProcessing:"<<timeLaserOdometry - timeLastProcessing<<std::endl;
            if (timeLaserOdometry - timeLastProcessing >= mappingProcessInterval) {

                timeLastProcessing = timeLaserOdometry;

                transformAssociateToMap();

                extractSurroundingKeyFrames();

                downsampleCurrentScan();

                mapOptimizationTime_1 = ros::Time::now().toSec();
                scan2MapOptimization();
                mapOptimizationTime_2 = ros::Time::now().toSec();
                mapOptimizationDuration = mapOptimizationTime_2 - mapOptimizationTime_1;


                saveKeyFramesAndFactor();

                correctPoses();

                publishTF();

                publishKeyPosesAndFrames();

                clearCloud();

                // WILLIAM BEGIN
                // std::cout << "laserCloudCornerFromMapDSNum = " << laserCloudCornerFromMapDSNum << std::endl;
                std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-" << std::endl;
                std::cout << "-----------------------------------------------------------------------" << std::endl;
                std::cout << "------------- You are running Map Optimization version ----------------" << std::endl;
                std::cout << "-----------------------------------------------------------------------" << std::endl;               
                std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-" << std::endl;
                std::cout << "--------------------------- One Period --------------------------------" << std::endl;            
                std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-" << std::endl;
                std::cout << "------------------- Down Size Filter Parameter ------------------------" << std::endl;
                std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-" << std::endl;
                std::cout << "Corner = " << DSFCornerLeafSize << std::endl;
                std::cout << "Surf = " << DSFSurfLeafSize << std::endl; 
                std::cout << "Outlier = " << DSFOutlierLeafSize << std::endl; 
                std::cout << "HistoryKey Frames = " << DSFHistoryKeyFramesLeafSize << std::endl; 
                std::cout << "Surrounding Key Poses = " << DSFSurroundingKeyPosesLeafSize << std::endl; 
                std::cout << "Global Map Key Poses = " << DSFGlobalMapKeyPosesLeafSize << std::endl; 
                std::cout << "Global Map Key Frames = " << DSFGlobalMapKeyFramesLeafSize << std::endl;                
                std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-" << std::endl;
                std::cout << "------------------------ Robot Position -------------------------------" << std::endl;
                std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-" << std::endl;
                std::cout << "currentRobotPosPoint (x, y, z)  = " << "(" << currentRobotPosPoint.x << ", " 
                << currentRobotPosPoint.y << ", " << currentRobotPosPoint.z << ")" << std::endl;
                std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-" << std::endl;
                std::cout << "---------------------- Time Durations (ms) ----------------------------" << std::endl;
                std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-" << std::endl;
                std::cout << "Map Optimization Duration = " << mapOptimizationDuration*1000 << std::endl;
                std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-" << std::endl;
                std::cout << "------------------- Local Map Points Quantity -------------------------" << std::endl;
                std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-" << std::endl;
                std::cout << "Corner Points Quantity = " << laserCloudCornerFromMapDSNum << std::endl;
                std::cout << "Surf Points Quantity = " << laserCloudSurfFromMapDSNum << std::endl;
                std::cout << "Local Map Points Quantity = " << laserCloudCornerFromMapDSNum + laserCloudSurfFromMapDSNum << std::endl;
                std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-" << std::endl;
                std::cout << "----------------- Current Frame Points Quantity -----------------------" << std::endl;
                std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-" << std::endl;
                std::cout << "Corner Points Quantity = " << laserCloudCornerLastDSNum << std::endl;
                std::cout << "Surf + Outlier Points Quantity = " << laserCloudSurfTotalLastDSNum << std::endl;
                std::cout << "Current Frame Points Quantity = " << laserCloudCornerLastDSNum + laserCloudSurfTotalLastDSNum << std::endl;
                std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-" << std::endl;
            }
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("\033[1;32m---->\033[0m Map Optimization Started.");

    mapOptimization MO;

    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::Rate rate(200);
    while (ros::ok())
    {
        ros::spinOnce();

        MO.run();

        rate.sleep();
    }

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}
