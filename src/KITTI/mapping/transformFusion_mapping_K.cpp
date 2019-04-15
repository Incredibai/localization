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

#include "utility.h"
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include "matrix.h"
#include <vector>

std::ofstream outfile;

class TransformFusion{

private:

    ros::NodeHandle nh;

    ros::Publisher pubLaserOdometry2;
    ros::Subscriber subLaserOdometry;
    ros::Subscriber subOdomAftMapped;
  

    nav_msgs::Odometry laserOdometry2;
    tf::StampedTransform laserOdometryTrans2;
    tf::TransformBroadcaster tfBroadcaster2;

    // WILLIAM BEGIN
    tf::StampedTransform tf_begin;
    tf::StampedTransform tf_end;
    tf::TransformListener listener;
    tf::StampedTransform transform;
    tf::TransformBroadcaster myTFBroadcaster;
    bool tf_begin_saved = false;
    bool tf_end_saved = false;
    bool firstFlag = true;
    double timeLastSavedTF;
    double timeCurrentTF;
    int count = 0;
    nav_msgs::Odometry GPSPosition;
    ros::Publisher pubGPSPosition;

    double translationErrorVectorXSum = 0;
    double translationErrorVectorYSum = 0;
    double translationErrorVectorZSum = 0;

    tf::StampedTransform map_2_camera_init_Trans;
    tf::TransformBroadcaster tfBroadcasterMap2CameraInit;

    tf::StampedTransform camera_2_base_link_Trans;
    tf::TransformBroadcaster tfBroadcasterCamera2Baselink;

    float transformSum[6];
    float transformIncre[6];
    float transformMapped[6];
    float transformBefMapped[6];
    float transformAftMapped[6];
    float transformMappedLast[6];

    std_msgs::Header currentHeader;
    
    string gt_dir = "/home/william/data/KITTI/dataset/poses";
    string file_name = "00.txt";
    vector<Matrix> poses_gt = loadPoses(gt_dir + "/" + file_name);

public:

    TransformFusion(){

        pubLaserOdometry2 = nh.advertise<nav_msgs::Odometry> ("/integrated_to_init", 5);
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, &TransformFusion::laserOdometryHandler, this);
        subOdomAftMapped = nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 5, &TransformFusion::odomAftMappedHandler, this);
        pubGPSPosition = nh.advertise<nav_msgs::Odometry>("/GPSPosition", 1);

        laserOdometry2.header.frame_id = "/camera_init";
        laserOdometry2.child_frame_id = "/camera";

        laserOdometryTrans2.frame_id_ = "/camera_init";
        laserOdometryTrans2.child_frame_id_ = "/camera";

        // WILLIAM BEGIN
        tf_begin.frame_id_ = "/camera_init";
        tf_begin.child_frame_id_ = "/tf_begin";
        tf_end.frame_id_ = "/camera_init";
        tf_end.child_frame_id_ = "/tf_end";

        map_2_camera_init_Trans.frame_id_ = "/map";
        map_2_camera_init_Trans.child_frame_id_ = "/camera_init";

        camera_2_base_link_Trans.frame_id_ = "/camera";
        camera_2_base_link_Trans.child_frame_id_ = "/base_link";

        GPSPosition.header.frame_id = "/camera_init";

        for (int i = 0; i < 6; ++i)
        {
            transformSum[i] = 0;
            transformIncre[i] = 0;
            transformMapped[i] = 0;
            transformBefMapped[i] = 0;
            transformAftMapped[i] = 0;
            transformMappedLast[i] = 0;
        }
        
    }

    void transformAssociateToMap()
    {
        float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) // laserOdometry during mapping process
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
        transformMapped[0] = -asin(srx);

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
        transformMapped[1] = atan2(srycrx / cos(transformMapped[0]), 
                                   crycrx / cos(transformMapped[0]));
        
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
        transformMapped[2] = atan2(srzcrx / cos(transformMapped[0]), 
                                   crzcrx / cos(transformMapped[0]));

        x1 = cos(transformMapped[2]) * transformIncre[3] - sin(transformMapped[2]) * transformIncre[4];
        y1 = sin(transformMapped[2]) * transformIncre[3] + cos(transformMapped[2]) * transformIncre[4];
        z1 = transformIncre[5];

        x2 = x1;
        y2 = cos(transformMapped[0]) * y1 - sin(transformMapped[0]) * z1;
        z2 = sin(transformMapped[0]) * y1 + cos(transformMapped[0]) * z1;

        transformMapped[3] = transformAftMapped[3] 
                           - (cos(transformMapped[1]) * x2 + sin(transformMapped[1]) * z2);
        transformMapped[4] = transformAftMapped[4] - y2;
        transformMapped[5] = transformAftMapped[5] 
                           - (-sin(transformMapped[1]) * x2 + cos(transformMapped[1]) * z2);
    }

    void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry)
    { 
        currentHeader = laserOdometry->header;
        // WILLIAM BEGIN
        // cout << "header: " << laserOdometry->header << "***********laserOdom Time: " << ros::Time::now() << endl;
        // WILLIAM END
        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;
        tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);

        transformSum[0] = -pitch;
        transformSum[1] = -yaw;
        transformSum[2] = roll;

        transformSum[3] = laserOdometry->pose.pose.position.x;
        transformSum[4] = laserOdometry->pose.pose.position.y;
        transformSum[5] = laserOdometry->pose.pose.position.z;

        // WILLIAM BEGIN
        // double roll_begin, pitch_begin, yaw_begin;

/*
std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-" << std::endl;
std::cout << "--------------------------- One Period --------------------------------" << std::endl;            
std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-" << std::endl;        
std::cout << "transformSum[0] " << transformSum[0] << std::endl;
std::cout << "transformSum[1] " << transformSum[1] << std::endl;
std::cout << "transformSum[2] " << transformSum[2] << std::endl;
std::cout << "transformSum[3] " << transformSum[3] << std::endl;
std::cout << "transformSum[4] " << transformSum[4] << std::endl;
std::cout << "transformSum[5] " << transformSum[5] << std::endl;
std::cout << std::endl;
std::cout << "TF From map status: " << tf_end_saved << std::endl;
std::cout << "transformBefMapped[0] " << transformBefMapped[0] << std::endl;
std::cout << "transformBefMapped[1] " << transformBefMapped[1] << std::endl;
std::cout << "transformBefMapped[2] " << transformBefMapped[2] << std::endl;
std::cout << "transformBefMapped[3] " << transformBefMapped[3] << std::endl;
std::cout << "transformBefMapped[4] " << transformBefMapped[4] << std::endl;
std::cout << "transformBefMapped[5] " << transformBefMapped[5] << std::endl;
std::cout << std::endl;
*/

        if(firstFlag)
        {
            firstFlag = false;
            timeLastSavedTF = laserOdometry->header.stamp.toSec();
        }

        timeCurrentTF = laserOdometry->header.stamp.toSec(

        );
        
        if(!tf_begin_saved && abs(timeLastSavedTF - timeCurrentTF)>=0.49)
        {
            tf_begin_saved = true;
            timeLastSavedTF = timeCurrentTF;

            geometry_msgs::Quaternion geoQuat_begin = laserOdometry->pose.pose.orientation;
            tf_begin.stamp_ = laserOdometry->header.stamp;
            tf_begin.setOrigin(tf::Vector3(transformSum[3], transformSum[4], transformSum[5]));
            tf_begin.setRotation(tf::Quaternion(-geoQuat_begin.y, -geoQuat_begin.z, geoQuat_begin.x, geoQuat_begin.w));
            myTFBroadcaster.sendTransform(tf_begin);
        }

        if(tf_begin_saved && tf_end_saved)
        {
            geometry_msgs::Quaternion geoQuat_end = laserOdometry->pose.pose.orientation;
            tf_end.stamp_ = laserOdometry->header.stamp;
            tf_end.setOrigin(tf::Vector3(transformSum[3], transformSum[4], transformSum[5]));
            tf_end.setRotation(tf::Quaternion(-geoQuat_end.y, -geoQuat_end.z, geoQuat_end.x, geoQuat_end.w));
            myTFBroadcaster.sendTransform(tf_end);
            tf_begin_saved = false;
            tf_end_saved = false;
        }


        // tf::Matrix3x3(tf::Quaternion(geoQuat_begin.z, -geoQuat_begin.x, -geoQuat_begin.y, geoQuat_begin.w)).getRPY(roll_begin, pitch_begin, yaw_begin);

        transformAssociateToMap();

        geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                  (transformMapped[2], -transformMapped[0], -transformMapped[1]);

        laserOdometry2.header.stamp = laserOdometry->header.stamp;
        laserOdometry2.pose.pose.orientation.x = -geoQuat.y;
        laserOdometry2.pose.pose.orientation.y = -geoQuat.z;
        laserOdometry2.pose.pose.orientation.z = geoQuat.x;
        laserOdometry2.pose.pose.orientation.w = geoQuat.w;
        laserOdometry2.pose.pose.position.x = transformMapped[3];
        laserOdometry2.pose.pose.position.y = transformMapped[4];
        laserOdometry2.pose.pose.position.z = transformMapped[5];
        pubLaserOdometry2.publish(laserOdometry2);

        GPSPosition.header.stamp = laserOdometry->header.stamp;
        GPSPosition.pose.pose.orientation.x = 0;
        GPSPosition.pose.pose.orientation.y = 0;
        GPSPosition.pose.pose.orientation.z = 0;
        GPSPosition.pose.pose.orientation.w = 0;
        GPSPosition.pose.pose.position.x = -poses_gt[count+2].val[0][3];
        GPSPosition.pose.pose.position.y = poses_gt[count+2].val[1][3];
        GPSPosition.pose.pose.position.z = poses_gt[count+2].val[2][3];
        GPSPosition.twist.twist.angular.x = 0;
        GPSPosition.twist.twist.angular.y = 0;
        GPSPosition.twist.twist.angular.z = 0;
        GPSPosition.twist.twist.linear.x = 0;
        GPSPosition.twist.twist.linear.y = 0;
        GPSPosition.twist.twist.linear.z = 0;
        pubGPSPosition.publish(GPSPosition);

        if(count == 0){
            transformMappedLast[3] = transformMapped[3];
            transformMappedLast[4] = transformMapped[4];
            transformMappedLast[5] = transformMapped[5];
        }
        std::cout << "----------------------- transformFusion.cpp ---------------------------" << std::endl;
        std::cout << "--------------------------- One Period --------------------------------" << std::endl;
        std::cout << "frame " << count << ": " << std::endl;
        std::cout << "header: " << laserOdometry2.header.stamp << std::endl;

        outfile << "----------------------- transformFusion.cpp ---------------------------" << std::endl;
        outfile << "--------------------------- One Period --------------------------------" << std::endl;
        outfile << "frame " << count << ": " << std::endl;
        outfile << "header: " << laserOdometry2.header.stamp << std::endl;   
        // std::cout << "transformMapped[0] " << transformMapped[0] << std::endl;
        // std::cout << "transformMapped[1] " << transformMapped[1] << std::endl;
        // std::cout << "transformMapped[2] " << transformMapped[2] << std::endl;
        // std::cout << "transformMapped[3] " << transformMapped[3] << std::endl;
        // std::cout << "transformMapped[4] " << transformMapped[4] << std::endl;
        // std::cout << "transformMapped[5] " << transformMapped[5] << std::endl;
        if(count > 0){
            translationErrorVectorXSum += abs((-transformMapped[3] - (-transformMappedLast[3])) - (poses_gt[count+2].val[0][3] - poses_gt[count+1].val[0][3]));
            translationErrorVectorZSum += abs((transformMapped[5] - transformMappedLast[5]) - (poses_gt[count+2].val[2][3] - poses_gt[count+1].val[2][3]));
        }
        if(count > 0 && count < 500){
            std::cout << "frame 0-500 ---->" << std::endl;
            std::cout << "Translation Vector X Average Error: " << translationErrorVectorXSum/count << std::endl;
            std::cout << "Translation Vector Z Average Error: " << translationErrorVectorZSum/count << std::endl;       
            outfile << "frame 0-500 ---->" << std::endl;
            outfile << "Translation Vector X Average Error: " << translationErrorVectorXSum/count << std::endl;
            outfile << "Translation Vector Z Average Error: " << translationErrorVectorZSum/count << std::endl; 
        }
        if(count == 500){
            translationErrorVectorXSum = 0;
            translationErrorVectorZSum = 0;   
        }
        if(count > 500 && count < 1000){
            std::cout << "frame 500-1000 ---->" << std::endl;
            std::cout << "Translation Vector X Average Error: " << translationErrorVectorXSum/(count-500) << std::endl;
            std::cout << "Translation Vector Z Average Error: " << translationErrorVectorZSum/(count-500) << std::endl;
            outfile << "frame 500-1000 ---->" << std::endl;
            outfile << "Translation Vector X Average Error: " << translationErrorVectorXSum/(count-500) << std::endl;
            outfile << "Translation Vector Z Average Error: " << translationErrorVectorZSum/(count-500) << std::endl;      
        }
        if(count == 1000){
            translationErrorVectorXSum = 0;
            translationErrorVectorZSum = 0;   
        }
        if(count > 1000 && count < 2000){
            std::cout << "frame 1000-2000 ---->" << std::endl;
            std::cout << "Translation Vector X Average Error: " << translationErrorVectorXSum/(count-1000) << std::endl;
            std::cout << "Translation Vector Z Average Error: " << translationErrorVectorZSum/(count-1000) << std::endl;
            outfile << "frame 1000-2000 ---->" << std::endl;
            outfile << "Translation Vector X Average Error: " << translationErrorVectorXSum/(count-1000) << std::endl;
            outfile << "Translation Vector Z Average Error: " << translationErrorVectorZSum/(count-1000) << std::endl;     
        }
        if(count == 2000){
            translationErrorVectorXSum = 0;
            translationErrorVectorZSum = 0;   
        }
        if(count > 2000 && count < 4000){
            std::cout << "frame 2000-4000 ---->" << std::endl;
            std::cout << "Translation Vector X Average Error: " << translationErrorVectorXSum/(count-2000) << std::endl;
            std::cout << "Translation Vector Z Average Error: " << translationErrorVectorZSum/(count-2000) << std::endl;
            outfile << "frame 2000-4000 ---->" << std::endl;
            outfile << "Translation Vector X Average Error: " << translationErrorVectorXSum/(count-2000) << std::endl;
            outfile << "Translation Vector Z Average Error: " << translationErrorVectorZSum/(count-2000) << std::endl;     
        }

        count++;
        transformMappedLast[3] = transformMapped[3];
        transformMappedLast[4] = transformMapped[4];
        transformMappedLast[5] = transformMapped[5];

        laserOdometryTrans2.stamp_ = laserOdometry->header.stamp;
        laserOdometryTrans2.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        laserOdometryTrans2.setOrigin(tf::Vector3(transformMapped[3], transformMapped[4], transformMapped[5]));
        tfBroadcaster2.sendTransform(laserOdometryTrans2);

    }

    void odomAftMappedHandler(const nav_msgs::Odometry::ConstPtr& odomAftMapped)
    {
        // WILLIAM BEGIN
        tf_end_saved = true;
        // cout << "header: " << odomAftMapped->header << "***********mapOdom Time: " << ros::Time::now() << endl; 
        // WILLIAM END       
        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = odomAftMapped->pose.pose.orientation;
        tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);

        transformAftMapped[0] = -pitch;
        transformAftMapped[1] = -yaw;
        transformAftMapped[2] = roll;

        transformAftMapped[3] = odomAftMapped->pose.pose.position.x;
        transformAftMapped[4] = odomAftMapped->pose.pose.position.y;
        transformAftMapped[5] = odomAftMapped->pose.pose.position.z;

        transformBefMapped[0] = odomAftMapped->twist.twist.angular.x;
        transformBefMapped[1] = odomAftMapped->twist.twist.angular.y;
        transformBefMapped[2] = odomAftMapped->twist.twist.angular.z;

        transformBefMapped[3] = odomAftMapped->twist.twist.linear.x;
        transformBefMapped[4] = odomAftMapped->twist.twist.linear.y;
        transformBefMapped[5] = odomAftMapped->twist.twist.linear.z;
    }

    vector<Matrix> loadPoses(string file_name) {    
        vector<Matrix> poses;
        FILE *fp = fopen(file_name.c_str(),"r");
        if (!fp)
            return poses;
        while (!feof(fp)) {
            Matrix P = Matrix::eye(4);
            if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                        &P.val[0][0], &P.val[0][1], &P.val[0][2], &P.val[0][3],
                        &P.val[1][0], &P.val[1][1], &P.val[1][2], &P.val[1][3],
                        &P.val[2][0], &P.val[2][1], &P.val[2][2], &P.val[2][3] )==12) {
            poses.push_back(P);
            }
        }
        fclose(fp);
        return poses;
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    outfile.open("/home/william/data/LeGO-LOAM/TranslationError");

    TransformFusion TFusion;

    ROS_INFO("\033[1;32m---->\033[0m Transform Fusion Started.");

    ros::spin();

    outfile.close();
    
    return 0;
}
