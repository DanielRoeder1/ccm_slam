/**
* This file is part of CCM-SLAM.
*
* Copyright (C): Patrik Schmuck <pschmuck at ethz dot ch> (ETH Zurich)
* For more information see <https://github.com/patriksc/CCM-SLAM>
*
* CCM-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* CCM-SLAM is based in the monocular version of ORB-SLAM2 by Raúl Mur-Artal.
* CCM-SLAM partially re-uses modules of ORB-SLAM2 in modified or unmodified condition.
* For more information see <https://github.com/raulmur/ORB_SLAM2>.
*
* CCM-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with CCM-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include <cslam/ClientHandler.h>


namespace cslam {

ClientHandler::ClientHandler(ros::NodeHandle Nh, ros::NodeHandle NhPrivate, vocptr pVoc, dbptr pDB, mapptr pMap, size_t ClientId, uidptr pUID, eSystemState SysState, const string &strCamFile, viewptr pViewer, bool bLoadMap)
    : mpVoc(pVoc),mpKFDB(pDB),mpMap(pMap),
      mNh(Nh),mNhPrivate(NhPrivate),
      mClientId(ClientId), mpUID(pUID), mSysState(SysState),
      mstrCamFile(strCamFile),
      mpViewer(pViewer),mbReset(false),
      mbLoadedMap(bLoadMap)
{
    if(mpVoc == nullptr || mpKFDB == nullptr || mpMap == nullptr || (mpUID == nullptr && mSysState == eSystemState::SERVER))
    {
        cout << ("In \" ClientHandler::ClientHandler(...)\": nullptr exception") << endl;
        throw estd::infrastructure_ex();
    }

    mpMap->msuAssClients.insert(mClientId);

    mg2oS_wcurmap_wclientmap = g2o::Sim3(); //identity transformation

    if(mSysState == eSystemState::CLIENT)
    {
        std::string TopicNameCamSub;
        string TopicNameScale = "/tellos/ScaleFactorClient" + std::to_string(ClientId);
        TopicNameScale = "/tellos/ScaleFactorClient";
        mSubScale = mNh.subscribe<ccmslam_msgs::Calibration>(TopicNameScale,10,&ClientHandler::SetScale,this);

        mNhPrivate.param("TopicNameCamSub",TopicNameCamSub,string("nospec"));
        mSubCam = mNh.subscribe<sensor_msgs::Image>(TopicNameCamSub,10,boost::bind(&ClientHandler::CamImgCb,this,_1));

        cout << "Camera Input topic: " << TopicNameCamSub << endl;
    }
}
#ifdef LOGGING
void ClientHandler::InitializeThreads(boost::shared_ptr<estd::mylog> pLogger)
#else
void ClientHandler::InitializeThreads()
#endif
{
    #ifdef LOGGING
    this->InitializeCC(pLogger);
    #else
    this->InitializeCC();
    #endif

    if(mSysState == eSystemState::CLIENT)
    {
        this->InitializeClient();
    }
    else if(mSysState == eSystemState::SERVER)
    {
        this->InitializeServer(mbLoadedMap);
    }
    else
    {
        cout << "\033[1;31m!!!!! ERROR !!!!!\033[0m ClientHandler::InitializeThreads(): invalid systems state: " << mpCC->mSysState << endl;
        throw infrastructure_ex();
    }
}

#ifdef LOGGING
void ClientHandler::InitializeCC(boost::shared_ptr<mylog> pLogger)
#else
void ClientHandler::InitializeCC()
#endif
{
    std::stringstream* ss;

    mpCC.reset(new CentralControl(mNh,mNhPrivate,mClientId,mSysState,shared_from_this(),mpUID));

    if(mSysState == eSystemState::CLIENT)
    {
        ss = new stringstream;
        *ss << "FrameId";
        mNhPrivate.param(ss->str(),mpCC->mNativeOdomFrame,std::string("nospec"));
    }
    else if(mSysState == eSystemState::SERVER)
    {
        ss = new stringstream;
        *ss << "FrameId" << mClientId;
        mNhPrivate.param(ss->str(),mpCC->mNativeOdomFrame,std::string("nospec"));
    }
    else
    {
        cout << "\033[1;31m!!!!! ERROR !!!!!\033[0m ClientHandler::InitializeThreads(): invalid systems state: " << mpCC->mSysState << endl;
        throw infrastructure_ex();
    }

    if(mpCC->mNativeOdomFrame=="nospec")
    {
        ROS_ERROR_STREAM("In \" ServerCommunicator::ServerCommunicator(...)\": bad parameters");
        throw estd::infrastructure_ex();
    }

    {
        if(mSysState==CLIENT)
        {
            cv::FileStorage fSettings(mstrCamFile, cv::FileStorage::READ);

            float c0t00 = fSettings["Cam0.T00"];
            float c0t01 = fSettings["Cam0.T01"];
            float c0t02 = fSettings["Cam0.T02"];
            float c0t03 = fSettings["Cam0.T03"];
            float c0t10 = fSettings["Cam0.T10"];
            float c0t11 = fSettings["Cam0.T11"];
            float c0t12 = fSettings["Cam0.T12"];
            float c0t13 = fSettings["Cam0.T13"];
            float c0t20 = fSettings["Cam0.T20"];
            float c0t21 = fSettings["Cam0.T21"];
            float c0t22 = fSettings["Cam0.T22"];
            float c0t23 = fSettings["Cam0.T23"];
            float c0t30 = fSettings["Cam0.T30"];
            float c0t31 = fSettings["Cam0.T31"];
            float c0t32 = fSettings["Cam0.T32"];
            float c0t33 = fSettings["Cam0.T33"];
            mpCC->mT_SC << c0t00,c0t01,c0t02,c0t03,c0t10,c0t11,c0t12,c0t13,c0t20,c0t21,c0t22,c0t23,c0t30,c0t31,c0t32,c0t33;
        }
        else
        {
            //no mstrCamFile on Server...
        }
    }

    mpMap->mOdomFrame = mpCC->mNativeOdomFrame;
    mpMap->AddCCPtr(mpCC);

    #ifdef LOGGING
    mpCC->mpLogger = pLogger;
    #endif

    delete ss;
}

void ClientHandler::InitializeClient()
{
    cout << "Client " << mClientId << " --> Initialize Threads" << endl;

    /* ---------------------------------------- */
    std::stringstream* ss;
    ss = new stringstream;
    *ss << "PoseOut" << "Client" << mClientId;
    std::stringstream* ss2;
    ss2 = new stringstream;
    *ss2 << "PathOut" << "Client" << mClientId;
    std::stringstream* ss3;
    ss3 = new stringstream;
    *ss3 << "PoseOut" << "Client" << mClientId << "_test";
    string PubPoseTopicName = ss->str();
    string PathTopicName = ss2->str();
    string PubPoseTopicName_t = ss3->str();
    mPubPose = mNh.advertise<geometry_msgs::PoseStamped>(PubPoseTopicName, 10);
    mPubPath = mNh.advertise<nav_msgs::Path>(PathTopicName, 10);



    /*  ----------------------------------------- */

    //+++++ Create Drawers. These are used by the Viewer +++++
    mpViewer.reset(new Viewer(mpMap,mpCC));
    usleep(10000);
    //+++++ Initialize the Local Mapping thread +++++
    mpMapping.reset(new LocalMapping(mpCC,mpMap,mpKFDB,mpViewer));
    usleep(10000);
//    +++++ Initialize the communication thread +++++
    mpComm.reset(new Communicator(mpCC,mpVoc,mpMap,mpKFDB));
    mpComm->SetMapping(mpMapping);
    usleep(10000);
    mpMap->SetCommunicator(mpComm);
    mpMapping->SetCommunicator(mpComm);
    usleep(10000);
    //+++++ Initialize the tracking thread +++++
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracking.reset(new Tracking(mpCC, mpVoc, mpViewer, mpMap, mpKFDB, mstrCamFile, mClientId));
    usleep(10000);
    mpTracking->SetCommunicator(mpComm);
    mpTracking->SetLocalMapper(mpMapping);
    mpViewer->SetTracker(mpTracking);
    usleep(10000);
    //Launch Threads
    //Should no do that before, a fast system might already use a pointe before it was set -> segfault
    mptMapping.reset(new thread(&LocalMapping::RunClient,mpMapping));
    mptComm.reset(new thread(&Communicator::RunClient,mpComm));
    mptViewer.reset(new thread(&Viewer::RunClient,mpViewer));
    /* --------------------------------------------- */
    ptrPoseStamped.reset(new thread(&ClientHandler::PublishPoseThread, this));
    usleep(10000);
}

void ClientHandler::InitializeServer(bool bLoadMap)
{
    cout << "Client " << mClientId << " --> Initialize Threads" << endl;

    //+++++ Initialize the Loop Finder thread and launch +++++
    mpLoopFinder.reset(new LoopFinder(mpCC,mpKFDB,mpVoc,mpMap));
    mptLoopClosure.reset(new thread(&LoopFinder::Run,mpLoopFinder));
    usleep(10000);
    //+++++ Initialize the Local Mapping thread +++++
    mpMapping.reset(new LocalMapping(mpCC,mpMap,mpKFDB,mpViewer));
    mpMapping->SetLoopFinder(mpLoopFinder); //tempout
    usleep(10000);
    //+++++ Initialize the communication thread +++++
    mpComm.reset(new Communicator(mpCC,mpVoc,mpMap,mpKFDB,bLoadMap));
    mpComm->SetMapping(mpMapping);
    usleep(10000);
    mpMapping->SetCommunicator(mpComm);
    mpMap->SetCommunicator(mpComm);
    usleep(10000);
    //Launch Threads
    //Should not do that before, a fast system might already use a pointer before it was set -> segfault
    mptMapping.reset(new thread(&LocalMapping::RunServer,mpMapping));
    mptComm.reset(new thread(&Communicator::RunServer,mpComm));
    usleep(10000);
    if(mpCC->mpCH == nullptr)
    {
        ROS_ERROR_STREAM("ClientHandler::InitializeThreads()\": mpCC->mpCH is nullptr");
        throw estd::infrastructure_ex();
    }

    // Publishing MapPoints from KeyFrames for DepthPrediction
    std::stringstream* ss;
    ss = new stringstream;
    *ss << "KeyFramePoints" << mClientId;
    pubKF = mNh.advertise<ccmslam_msgs::KF_PointCloud>(ss->str(),10);
    ptrKFpub.reset(new thread(&ClientHandler::PublishKFThread, this));

    // Receive depthmaps from DepthPrediction
    ss = new stringstream;
    *ss << "DepthmapsOut" << mClientId;
    subDepth = mNh.subscribe<ccmslam_msgs::KF_DepthMap>(ss->str(),10,&ClientHandler::SetDepthMap,this);
}

void ClientHandler::ChangeMap(mapptr pMap, g2o::Sim3 g2oS_wnewmap_wcurmap)
{
    mpMap = pMap;

    mg2oS_wcurmap_wclientmap = g2oS_wnewmap_wcurmap*mg2oS_wcurmap_wclientmap;
    mpCC->mg2oS_wcurmap_wclientmap = mg2oS_wcurmap_wclientmap;

    bool bLockedComm = mpCC->LockComm(); //should be locked and therefore return false
    bool bLockedMapping = mpCC->LockMapping(); //should be locked and therefore return false

    if(bLockedComm || bLockedMapping)
    {
        if(bLockedComm) cout << "\033[1;31m!!!!! ERROR !!!!!\033[0m ClientHandler::ChangeMap(): Comm not locked: " << endl;
        if(bLockedMapping) cout << "\033[1;31m!!!!! ERROR !!!!!\033[0m ClientHandler::ChangeMap(): Mapping not locked: " << endl;
        throw infrastructure_ex();
    }

    mpComm->ChangeMap(mpMap);
    mpMapping->ChangeMap(mpMap);
    mpLoopFinder->ChangeMap(mpMap);
}

void ClientHandler::SaveMap(const string &path_name) {
        std::cout << "--> Lock System" << std::endl;
        while(!mpCC->LockMapping()){usleep(params::timings::miLockSleep);}
        while(!mpCC->LockComm()){usleep(params::timings::miLockSleep);}
        while(!mpCC->LockPlaceRec()){usleep(params::timings::miLockSleep);}
        std::cout << "----> done" << std::endl;

        mpMap->SaveMap(path_name);

        std::cout << "--> Unlock System" << std::endl;
        mpCC->UnLockMapping();
        mpCC->UnLockComm();
        mpCC->UnLockPlaceRec();
        std::cout << "----> done" << std::endl;
}

void ClientHandler::SetMapMatcher(matchptr pMatch)
{
    mpMapMatcher = pMatch;
    mpComm->SetMapMatcher(mpMapMatcher);
    mpMapping->SetMapMatcher(mpMapMatcher);
}

void ClientHandler::CamImgCb(sensor_msgs::ImageConstPtr pMsg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;

    try
    {
        cv_ptr = cv_bridge::toCvShare(pMsg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracking->Reset();
            mbReset = false;
        }
    }

    mpTracking->GrabImageMonocular(cv_ptr->image,cv_ptr->header.stamp.toSec(),cv_ptr->header.stamp.toNSec());

    /* ---------------------------------------  */
    if (mpTracking->mState == 2) {
        receivedImageFlag = true;
    }
}

/* --------------------------------------------------- */
// Todo: 12 deg rotation not yet applied to rotation matrix
// Todo: read 12 deg from config file
void ClientHandler::PublishPoseThread(){
    geometry_msgs::PoseStamped pose_msg;
    nav_msgs::Path path_pose_msg;

    std::string frame_id = "world";
    pose_msg.header.frame_id = frame_id;
    path_pose_msg.header.frame_id = frame_id;

    tf::Matrix3x3 tf_orb_to_ros(
        0,  0,  1,
        -1,  0,  0,
        0, -1,  0);

    // Rotate to account for Tello camera looking slightly downwards
    double deg = 12;
    double rads = deg * 3.14159265 / 180;
    tf::Matrix3x3 rotate_12degY(
        cos(rads), 0, sin(rads),
        0,         1,         0,
        -sin(rads), 0, cos(rads));

    // Needed to make sure that ScaleFactor is always true to world reference i.e not affected by previous scales
    float old_scale_factor = ScaleFactors[mClientId];
    float old_zoffset = ZOffsets[mClientId];
    float world_scale_factor;
    int old_map_id = mClientId;

    while(1) {
        // suspend operation for microsecond interval
        usleep(3333);
        if (receivedImageFlag) {
            receivedImageFlag = false;
            if (mpTracking->mState == mpTracking->OK) {
                // Get transform / pose matrix from current frame
                cv::Mat Tcw = mpTracking->mCurrentFrame->mTcw;
                
                tf::Matrix3x3 tf_camera_rotation(
                    Tcw.at<float> (0, 0), Tcw.at<float> (0, 1), Tcw.at<float> (0, 2),
                    Tcw.at<float> (1, 0), Tcw.at<float> (1, 1), Tcw.at<float> (1, 2),
                    Tcw.at<float> (2, 0), Tcw.at<float> (2, 1), Tcw.at<float> (2, 2)
                );
                
                // After a map merger these ids will differ set updates scale to true to updated pose path
                if(old_map_id != mpComm->client_map_id)
                {
                    std::cout << "ClientHandler: Changing scale index to: " << mpComm->client_map_id << " for Client " << mClientId << "\n";
                    old_map_id = mpComm->client_map_id;
                    updated_scale = true;
                }


                tf::Vector3 tf_camera_translation(Tcw.at<float> (0,3), Tcw.at<float> (1,3), Tcw.at<float> (2,3));

                // Transform from orb coordinate system to ros coordinate system on camera coordinates
                tf_camera_rotation    = tf_orb_to_ros * tf_camera_rotation;
                tf_camera_translation = tf_orb_to_ros * tf_camera_translation;

                // Inverse matrix
                tf_camera_rotation    = tf_camera_rotation.transpose();
                tf_camera_translation = -(tf_camera_rotation * tf_camera_translation);

                // Transform from orb coordinate system to ros coordinate system on map coordinates
                tf_camera_rotation    = tf_orb_to_ros * tf_camera_rotation;
                tf::Vector3 scaled_zoffset(0,0,ScaleFactors[mpComm->client_map_id] * ZOffsets[mpComm->client_map_id]);
                tf_camera_translation = rotate_12degY * tf_orb_to_ros * tf_camera_translation * ScaleFactors[mpComm->client_map_id] + scaled_zoffset;

                // Transform R T into pose object and subsequently into msg
                tf::Transform tf_transform = tf::Transform(tf_camera_rotation, tf_camera_translation);
                tf::Stamped<tf::Pose> grasp_tf_pose(tf_transform,ros::Time::now(), frame_id);
                tf::poseStampedTFToMsg(grasp_tf_pose, pose_msg);

                if(updated_scale){
                    world_scale_factor = ScaleFactors[mpComm->client_map_id] / old_scale_factor;
                    for(geometry_msgs::PoseStamped & pose_stamped: path_pose_msg.poses){
                        pose_stamped.pose.position.x *= world_scale_factor;
                        pose_stamped.pose.position.y *= world_scale_factor;
                        pose_stamped.pose.position.z = (pose_stamped.pose.position.z - (old_scale_factor * old_zoffset)) * world_scale_factor + ZOffsets[mpComm->client_map_id] * ScaleFactors[mpComm->client_map_id];
                    }
                    old_scale_factor = ScaleFactors[mpComm->client_map_id];
                    old_zoffset = ZOffsets[mpComm->client_map_id];
                    updated_scale = false;
                }

                
                // Change axis to change to correct alignment?
                //float x = pose_msg.pose.position.x;
                //pose_msg.pose.position.x = - pose_msg.pose.position.y;
                //pose_msg.pose.position.y = x;

                
                path_pose_msg.poses.push_back(pose_msg);

   
                mPubPose.publish(pose_msg);

                mPubPath.publish(path_pose_msg);
            }
        }
    }
}

void ClientHandler::SetScale(ccmslam_msgs::Calibration msg){
    ScaleFactors[msg.client_id] = msg.scale_factor;
    ZOffsets[msg.client_id] = msg.z_offset;
    
    if(msg.client_id == mpComm->client_map_id)
    {
        updated_scale = true;
        std::cout << "ClientHandler: Client "<< mClientId <<"; Scale factor: " << ScaleFactors[mpComm->client_map_id] << "; Z offset: " << ZOffsets[mpComm->client_map_id] << std::endl;
    }
}

void ClientHandler::PublishKFThread()
{
    ccmslam_msgs::KF_PointCloud kf_msg;
    geometry_msgs::Point32 p;

    while (1)
    {
        // suspend operation for microsecond interval
        usleep(3333);

        if(mpMapping->receivedKeyFrame)
        {
            mpMapping->receivedKeyFrame = false;
            kf_msg.pc.points.clear();
            
            
            boost::shared_ptr<KeyFrame> KF = mpMapping->mpCurrentKeyFrame;

            // The Timestamp of the KF equals the timestamp of the input image
            ros::Time t_stamp;
            t_stamp.fromNSec(KF->mTimeStamp_nsec);

            set<KeyFrame::mpptr> MapPoints = KF->GetMapPoints();   

            //std::cout << "##### Number of MapPoints in KF #####" << MapPoints.size() << std::endl;
            
            cv::Mat Rcw1 = KF->GetRotation();
            cv::Mat Rwc1 = Rcw1.t();
            cv::Mat tcw1 = KF->GetTranslation();

            // Camera parameters
            const float &fx1 = KF->fx;
            const float &fy1 = KF->fy;
            const float &cx1 = KF->cx;
            const float &cy1 = KF->cy;

            std::vector<cv::Point3f> projected_points;

            for(KeyFrame::mpptr MP : MapPoints)
            {
                cv::Mat x3D = MP->GetWorldPos();
                cv::Mat x3Dt = x3D.t();
                
                const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
                const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
                const float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
                const float invz1 = 1.0/z1;
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;

                p.x = u1;
                p.y = v1;
                p.z = z1;

                /*
                if(z1 < 0 || z1 > 10){
                    std::cout << z1 << std::endl;
                }
                */
                kf_msg.pc.points.push_back(p);
                
            }

            kf_msg.header.stamp = t_stamp;
            kf_msg.kf_id = KF->mId.first;
            kf_msg.client_id = KF->mId.second;
            pubKF.publish(kf_msg);
        }
    }
}

void ClientHandler::SetDepthMap(ccmslam_msgs::KF_DepthMap msg){
    int x_downsampling = 3;
    int y_downsamlping = 2;
    int rescale_factor = 3;

    std::cout << "Received DepthMap"<< std::endl;
    std::cout << msg.client_ids.size() << std::endl;
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg.img);

    int x_diff = 960 - (cv_ptr->image.cols * x_downsampling * rescale_factor);
    int y_diff = 720 - (cv_ptr->image.rows * y_downsamlping * rescale_factor);

    cv::Mat DepthMaps[cv_ptr->image.channels()];
    cv::split(cv_ptr->image, DepthMaps);
    cv::Mat DepthMap;
    int nRows = cv_ptr->image.rows;
    int nCols = cv_ptr->image.cols;
    float* p;
    int i,j,k;
    cv::Point3f point;
    std::vector<cv::Point3f> pointcloud;  

    

    for (i = 0; i != msg.client_ids.size(); ++i){
        pointcloud.clear();
        int client_id = msg.client_ids[i];
        int kf_id = msg.kf_ids[i];
        if(kf_id < 2){continue;}
        DepthMap = DepthMaps[i];

        std::cout << "client_id" << client_id << std::endl;
        std::cout << "kf_id" << kf_id << std::endl;
        
        boost::shared_ptr<KeyFrame> KF = mpMap->GetKfPtr(kf_id,client_id);
        // Cant access GetPoseInverse at this point ? -> Manually calc Twc

        cv::Mat Twc = KF->GetPoseInverse();
        cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
        cv::Mat twc = Twc.rowRange(0,3).col(3);


        for(j = 0; j < nRows; j++)
        {
            p = DepthMap.ptr<float>(j);
            for(k = 0; k < nCols; k++)
            {
                // https://stackoverflow.com/questions/31265245/extracting-3d-coordinates-given-2d-image-points-depth-map-and-camera-calibratio/31266627#31266627
                //        ((Reconstructing image coordinate from downsampled image) - cx) * Depth / fx 
                float x = ((k * x_downsampling * rescale_factor + x_diff / 2) - KF->cx) * p[k] * KF->invfx;
                float y = ((j * y_downsamlping * rescale_factor + y_diff / 2) - KF->cy) * p[k] * KF->invfy;

                cv::Mat x3D = (cv::Mat_<float>(3,1) << x, y, p[k]);
                cv::Mat x3Dt = x3D.t();
                point.x = Rwc.row(0).dot(x3Dt)+twc.at<float>(0);
                point.y = Rwc.row(1).dot(x3Dt)+twc.at<float>(1);
                point.z = Rwc.row(2).dot(x3Dt)+twc.at<float>(2);

                pointcloud.push_back(point);
            }
            KF->pc_depth = pointcloud;
        }

    }

}

/* --------------------------------------------------- */

void ClientHandler::LoadMap(const std::string &path_name) {

    std::cout << "--> Load Map" << std::endl;
    mpMap->LoadMap(path_name,mpVoc,mpComm,mpKFDB,mpUID);
    std::cout << "----> Done" << std::endl;

    std::cout << "--> Register KFs to database" << std::endl;
    auto kfs = mpMap->GetAllKeyFrames();
    for(auto kf : kfs) {
        mpKFDB->add(kf);
    }
    std::cout << "----> Done" << std::endl;

    mpMap->msnFinishedAgents.insert(this->mClientId);

    std::cout << "--> Show map" << std::endl;
    if(params::vis::mbActive)
        mpViewer->DrawMap(mpMap);
    std::cout << "----> Done" << std::endl;

//    cout << "Trigger GBA" << endl;
//    mpMap->RequestBA(mpCC->mClientId);

//    std::cout << "--> Show map" << std::endl;
//    if(params::vis::mbActive)
//        mpViewer->DrawMap(mpMap);
//    std::cout << "----> Done" << std::endl;
}

void ClientHandler::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

ClientHandler::kfptr ClientHandler::GetCurrentRefKFfromTracking()
{
    if(mpTracking->mState < 2)
        return nullptr;
    else
        return mpTracking->GetReferenceKF();
}

int ClientHandler::GetNumKFsinLoopFinder()
{
    if(mpLoopFinder)
        return mpLoopFinder->GetNumKFsinQueue();
    else
        return -1;
}

int ClientHandler::GetNumKFsinMapMatcher()
{
    if(mpMapMatcher)
        return mpMapMatcher->GetNumKFsinQueue();
    else
        return -1;
}

void ClientHandler::ClearCovGraph(size_t MapId)
{
    mpMapping->ClearCovGraph(MapId);
}

//#ifdef LOGGING
//void ClientHandler::SetLogger(boost::shared_ptr<mylog> pLogger)
//{
//    mpCC->mpLogger = pLogger;
//}
//#endif

} //end ns
