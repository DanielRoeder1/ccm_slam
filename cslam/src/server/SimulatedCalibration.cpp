#include "ros/ros.h"
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Empty.h>
#include <ccmslam_msgs/Calibration.h>
#include <cmath>

#include <ccmslam_msgs/StampedInt.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float32.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <vector>
#include <numeric>
#include <sstream>
#include <string.h>


/*
- The Listener clas subscribes to the tof_heigth topic of the drone and the PoseOut topic of the SLAM algorithm. For the
  prupose of using rosbags the tof msgs are republished using a current timestamp to make match them with the pose msgs
  using approximate time sysnchronization. Then 20 diverse points are collected and the Zoffset and Scale factor are caluclated
  using linear regression.
*/

class Listener{

    public:
        std::vector<float> tof_height;
        std::vector<float> pose_z;
        int client_id_num;

    private:
        ros::NodeHandle n;
        ros::Subscriber calibrate_sub;
        ros::Publisher calibrate_pub;

        ros::Subscriber sub_tof_height;
        ros::Publisher pub_tof_height;
        void republish_height(std_msgs::Int32 msg);

        message_filters::Subscriber<ccmslam_msgs::StampedInt> tof_sub;
        message_filters::Subscriber<geometry_msgs::PoseStamped> pose_sub;
        typedef message_filters::sync_policies::ApproximateTime<ccmslam_msgs::StampedInt, geometry_msgs::PoseStamped> MySyncPolicy;
        message_filters::Synchronizer<MySyncPolicy> sync;

        void data_callback(const ccmslam_msgs::StampedInt::ConstPtr& tof_msg, const geometry_msgs::PoseStamped::ConstPtr& pose_msg);
        void callibrate_callback(std_msgs::Empty msg);
        void calibrate(std::vector<float> tof_heights, std::vector<float> pose_z);

    public:
        Listener(char *client_id):sync(MySyncPolicy(10),tof_sub,pose_sub){
    
            client_id_num = *client_id - '0';

            //n.param("ClientId",client_id,0);
            ROS_INFO("Starting Calibration node for ClientId %c", *client_id);

            std::stringstream* ss1;
            ss1 = new std::stringstream;
            *ss1 << "/ccmslam/PoseOutClient" << *client_id;
            std::string subPoseName = ss1->str();

            ss1 = new std::stringstream;
            *ss1 << "/tellos/ScaleFactorClient" << *client_id;
            std::string pubCalibrationName = ss1->str();
            pubCalibrationName = "/tellos/ScaleFactorClient";


            ss1 = new std::stringstream;
            *ss1 << "stamped_tof" << *client_id;
            std::string subStampedPose = ss1->str();

            tof_sub.subscribe(n, subStampedPose, 10);
            pose_sub.subscribe(n,subPoseName, 10);
            sync.registerCallback(boost::bind(&Listener::data_callback,this, _1, _2));

            calibrate_sub = n.subscribe<std_msgs::Empty>("calibrate_z",1000, &Listener::callibrate_callback,this);
            calibrate_pub = n.advertise<ccmslam_msgs::Calibration>(pubCalibrationName, 1000);
            
            std::string tello_id;

            if(strcmp(client_id,"0") == 0){
                tello_id = "a";
            }
            else if(strcmp(client_id,"1") == 0)
            {
                tello_id = "b";
            }
            
            ss1 = new std::stringstream;
            *ss1 << "tellos/" << tello_id << "/tof_height";
            std::string subTofHeight = ss1->str();


            sub_tof_height = n.subscribe<std_msgs::Int32>(subTofHeight, 1000, &Listener::republish_height, this);
            pub_tof_height = n.advertise<ccmslam_msgs::StampedInt>(subStampedPose,1000);
        }
};

void Listener::data_callback(const ccmslam_msgs::StampedInt::ConstPtr& tof_msg,const geometry_msgs::PoseStamped::ConstPtr& pose_msg)
{
    if(tof_height.empty())
    {
        tof_height.push_back(float(tof_msg->value)/100);
        pose_z.push_back(pose_msg->pose.position.z);
    }
    else if(tof_height.size() > 20)
    {
        tof_sub.unsubscribe();
        pose_sub.unsubscribe();

        this->calibrate(tof_height, pose_z);
    }
    else if(abs(float(tof_msg->value)/100 - tof_height.back()) > 0.03)
    {
        tof_height.push_back(float(tof_msg->value)/100);
        pose_z.push_back(pose_msg->pose.position.z);
    }
}

void Listener::callibrate_callback(std_msgs::Empty msg)
{
    std::cout << "Starting sleep" << std::endl;
    usleep(5000000);
    std::cout << "Finished sleep" << std::endl;
}

void Listener::calibrate(std::vector<float> tof_heights, std::vector<float> pose_z)
{
    int n = tof_heights.size();
    double x_mean = std::accumulate(tof_heights.begin(), tof_heights.end(), 0.0) / tof_heights.size();
    //std::cout << "Xmean "<< x_mean << std::endl;
    double y_mean = std::accumulate(pose_z.begin(), pose_z.end(), 0.0) / pose_z.size();
    //std::cout << "Ymean "<< y_mean << std::endl;
    double SS_xy = std::inner_product(tof_heights.begin(), tof_heights.end(), pose_z.begin(), 0.0);
    SS_xy = SS_xy - n * x_mean * y_mean;
    //std::cout << "SS_xy "<< SS_xy << std::endl;

    double SS_xx = std::inner_product(tof_heights.begin(), tof_heights.end(), tof_heights.begin(), 0.0);
    //std::cout << "SS_xx "<< SS_xx << std::endl;
    SS_xx = SS_xx - n * x_mean * x_mean;


    double b_1 = SS_xy / SS_xx;
    double b_0 = y_mean - b_1 * x_mean;

    b_1 = 1 / b_1;
    b_0 = abs(b_0);


    ROS_INFO("CalibrationNode: ZOffset %f", b_0);
    ROS_INFO("CalibrationNode: Coefficient %f", b_1);

    
    //std::cout << "ZOffset: " << b_0 << std::endl;
    //std::cout << "Coefficient: " << b_1 << std::endl;

    /* Using only two points for calibration
    float diff_tof = tof_heights[0] - tof_heights[1];
    float diff_pose = pose_z[0] - pose_z[1];
    double zoffset = abs(pose_z[0]- (tof_heights[0] / diff_tof * diff_pose));
    double scale_factor = float(diff_tof) / diff_pose;
    std::cout << "Scaling factor: " << scale_factor << " ZOffset: " << zoffset<< std::endl;
    */
   if(b_1 > 10){
       std::cout << "### Encountered large scale factor and tunrcated to 2.5 0.35" << "\n";
       b_1 = 2.5;
       b_0 = 0.35;

   }

    ccmslam_msgs::Calibration calibration_msg;
    calibration_msg.client_id = client_id_num;
    calibration_msg.z_offset = b_0;
    calibration_msg.scale_factor = b_1;

    calibrate_pub.publish(calibration_msg);
}

void Listener::republish_height(std_msgs::Int32 msg)
{
    ccmslam_msgs::StampedInt stamped_msg;
    stamped_msg.header.stamp = ros::Time::now();
    stamped_msg.value = msg.data;
    pub_tof_height.publish(stamped_msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "SimulatedCalibrator");
    //ros::NodeHandle n;

    Listener l(argv[1]);

    ros::Rate r(100);

    while(ros::ok())
    {
        ros::spinOnce();
        r.sleep();
    }
}