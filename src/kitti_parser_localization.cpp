#include <ctime>
#include "ros/ros.h"
#include <fstream>
#include <string>
#include <vector>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "parameterUse.h"
#include "matrix.h"
#include <std_msgs/Bool.h>


using namespace std;



bool flag = false;
bool GetMap = false;
bool parseFinish = false;
string gt_dir = "/home/william/data/KITTI/dataset/poses";
string file_name = "00.txt";

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

void GetMapFlagHandler(const std_msgs::Bool msg){
    GetMap = true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "kitti_parser");
    ros::NodeHandle n;
    ros::Publisher velodyne_pub = n.advertise<sensor_msgs::PointCloud2> ("/velodyne_points", 2);
    ros::Subscriber subGetMapFlag = n.subscribe<std_msgs::Bool>("/GetMapFlag", 1, &GetMapFlagHandler);
    //pub frequence
    ros::Rate loop_rate(frequence);

    string bin_path;
    stringstream ss;
    ss<<lidar_base_dir<<sequence<<"/velodyne/";
    ss>>bin_path;
    std::vector<std::string> file_lists;
    read_filelists( bin_path, file_lists, "bin" );
    sort_filelists( file_lists, "bin" );

    vector<Matrix> poses_gt = loadPoses(gt_dir + "/" + file_name);

    while(ros::ok()){ // "if" rather than "while": only do file parsing once
      ros::spinOnce();

      for (int i = 0; i < file_lists.size() && GetMap && ros::ok() && !parseFinish; i++){
        
          if(i == file_lists.size()){
            parseFinish = true;
          }
          std::string bin_file = bin_path + file_lists[i];
          // load point cloud
          std::fstream input(bin_file.c_str(), std::ios::in | std::ios::binary);
          if(!input.good()){
            std::cerr << "Could not read file: " << bin_file << std::endl;
            exit(EXIT_FAILURE);
          }
          if(!flag){
            std::cerr << "-*-*- KITTI Parser read file success "<< std::endl;
            std::cerr << "-*-*- Data directory "  << lidar_base_dir << std::endl;
            std::cerr << "-*-*- Start publishing KITTI sequence "  << sequence << " " << std::endl;
            std::cerr << "-*-*- Frequency = " << frequence  << " " << std::endl;
            
            flag = true;
          }

          std::cout << "frame " << i << ": " << std::endl << poses_gt[i] << std::endl;

          input.seekg(0, std::ios::beg);

          pcl::PointCloud<pcl::PointXYZINormal> cloud64;
          cloud64.clear();
          cloud64.width = 150000;			// 预设大一点的空间
          cloud64.height = 1;
          cloud64.is_dense = true;
          cloud64.resize(cloud64.width*cloud64.height);
          int point_count;
          for (point_count=0; input.good() && !input.eof(); ) {
            pcl::PointXYZINormal point;
            input.read((char *) &point.x, sizeof(float)); //m
            input.read((char *) &point.y, sizeof(float));
            input.read((char *) &point.z, sizeof(float));
            input.read((char *) &point.curvature, sizeof(float));
            cloud64.points[point_count].x=point.x;
            cloud64.points[point_count].y=point.y;
            cloud64.points[point_count].z=point.z;
            cloud64.points[point_count].curvature=point.curvature;
            point_count++;
          }
          cloud64.width =point_count;
          cloud64.resize(cloud64.width*cloud64.height); // 重新调整点云尺寸至真实值
          input.close();

#ifdef SAVE_PARSER_LIDAR
          string file_path;
          stringstream ss;
          ss<<save_lidar_location<<i<<".txt";
          ss>>file_path;

          fstream fwriter;
          fwriter.open(file_path,ios::out);
          if(!fwriter.is_open()){
            cerr<<"can't open file: "<<file_path<<endl;
          }else{
            for(int index=0;index<cloud64.points.size();index++){
              double angle=cloud64.points[index].z/sqrt(cloud64.points[index].x*cloud64.points[index].x+cloud64.points[index].y*cloud64.points[index].y);
              angle=angle/3.1415926*180;
              fwriter<<cloud64.points[index].x<<"  "
                    <<cloud64.points[index].y<<"  "
                    <<cloud64.points[index].z<<"  "
                    <<angle<<endl;
            }
            fwriter.close();
          }
#endif

          // 将pcl::PCLPointCLoud2格式转换成sensor_msgs::PointCloud2格式
          sensor_msgs::PointCloud2 output64;
          pcl::toROSMsg(cloud64,output64);

          output64.header.seq = i;
          output64.header.frame_id="/velodyne";
          output64.header.stamp=ros::Time::now();
          velodyne_pub.publish(output64);

          loop_rate.sleep();
      }

    if(parseFinish){
      ROS_INFO("KITTI Sequence reach the end!");
    }
    }
    return 0;
}
