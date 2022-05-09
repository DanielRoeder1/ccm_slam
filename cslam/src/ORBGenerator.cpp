#include <iostream>
#include "ros/ros.h"

// CCMSLAM
#include <cslam/ORBextractor.h>
#include <cslam/ORBmatcher.h>

#include <opencv2/opencv.hpp>
#include <random>

// For loading npy matrix files
#include"cnpy.h"

// For creating directories and checking the number of files 
#include <boost/filesystem.hpp>


// Some functions to experiment with ORB matching
namespace cslam{
void ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}

void visualize_KPMatches(cv::Mat image1, std::vector<cv::KeyPoint> mvKeys1, cv::Mat image2, std::vector<cv::KeyPoint> mvKeys2, std::vector<int> vnMatches12, std::vector<int> vMatchedDistance)
{
    // Create a DMatch vecotr for KP matching visualizazation using CV
    std::vector<cv::DMatch> matches1to2;
    std::vector<cv::KeyPoint> matchedKeys1;
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
    {
        if(vnMatches12[i1]>=0)
        {
            matches1to2.push_back(cv::DMatch(i1, vnMatches12[i1], vMatchedDistance[vnMatches12[i1]]));
            matchedKeys1.push_back(mvKeys1[i1]);
        }

    }
    std::cout << matches1to2.size() << std::endl;
    cv::Mat img_matches;
    cv::drawMatches(image1,mvKeys1, image2,mvKeys2, matches1to2, img_matches);
    imshow("KeyPoint matches", img_matches);
    waitKey(0);

    cv::Mat img_keypoints;
    cv::drawKeypoints(image1, matchedKeys1, img_keypoints);
    imshow("Matched Keypoints Name", img_keypoints);
    waitKey(0);

    std::random_device rd;
    std::mt19937 g(rd());
 
    std::shuffle(mvKeys1.begin(), mvKeys1.end(), g);

    std::uniform_int_distribution<> distr(200, 400);
    mvKeys1.resize(distr(g));

    cv::Mat img_keypoints2;
    cv::drawKeypoints(image1, mvKeys1, img_keypoints2);
    imshow("Random sample", img_keypoints2);
    waitKey(0);

    cv::Mat C = (cv::Mat_<double>(3,3) << 5.700000000000000000e+02, -0.000000000000000000e+00, 3.200000000000000000e+02, 0.000000000000000000e+00, 5.700000000000000000e+02, 2.400000000000000000e+02, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00);

    cv::Mat img_undistorted;
    cv::Mat distortion;


    cv::undistort(image1, img_undistorted, C, distortion);
}

void orbextraction()
{
    // Params from ccmslam conf file
    const int nFeatures = 5000;
    const float fScaleFactor = 1.2;
    const int nLevels = 8;
    const int iIniThFAST = 20;
    const int iMinThFAST = 7;

    std::vector<cv::KeyPoint> mvKeys1;
    cv::Mat mDescriptors1;
    std::vector<cv::KeyPoint> mvKeys2;
    cv::Mat mDescriptors2;

    extractorptr mpORBextractor;
    mpORBextractor.reset(new ORBextractor(nFeatures,fScaleFactor,nLevels,iIniThFAST,iMinThFAST));

    cv::Mat image1 = cv::imread("../Pictures/image/0.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread("../Pictures/image/1.jpg", cv::IMREAD_GRAYSCALE);

    std::cout << image1.size() << std::endl;

    imshow("Window Name", image1);
    waitKey(0);
  
    (*mpORBextractor)(image1,cv::Mat(),mvKeys1,mDescriptors1);
    (*mpORBextractor)(image2,cv::Mat(),mvKeys2,mDescriptors2);

    std::cout << mvKeys1.size() << std::endl;

    cv::Mat img_keypoints;
    cv::drawKeypoints(image1, mvKeys1, img_keypoints);
    imshow("KeyPoints Image1", img_keypoints);
    waitKey(0);

    // ***** ORB MATCHING ***** ///
    const int HISTO_LENGTH = 30;
    const int TH_HIGH = 100;
    const int TH_LOW = 50;
    float mfNNratio = 0.9;
    bool mbCheckOrientation = true;

    int nmatches=0;

    // Vector storing the index of the matched Kp
    std::vector<int> vnMatches12(mvKeys1.size(),-1);
    std::vector<int> vnMatches21(mvKeys2.size(),-1);

    std::vector<int> vMatchedDistance(mvKeys2.size(),INT_MAX);

    std::vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    for(size_t i1=0, iend1=mvKeys1.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = mvKeys1[i1];
        int level1 = kp1.octave;
        if(level1>0)
            continue;

        cv::Mat d1 = mDescriptors1.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        for(size_t i2=0, iend1=mvKeys2.size(); i2<iend1; i2++)
        {

            cv::Mat d2 = mDescriptors2.row(i2);

            int dist = ORBmatcher::DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }
    
        if(bestDist<=TH_LOW)
        {
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                // Check if the KP from the second image is already matched to one from the first image
                // If this is the case then remove this link from the KP1 list. Resulting in the KP2 point having no match
                // The lines afterwards then add the new KP link
                if(vnMatches21[bestIdx2]>=0)
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;

                if(mbCheckOrientation)
                {
                    float rot = mvKeys1[i1].angle-mvKeys2[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }
    //visualize_KPMatches(image1,mvKeys1, image2,mvKeys2, vnMatches12, vMatchedDistance);
}

}


void sampleORB(std::string basepath)
{
    // Check if image and depth directories exist
    if(!boost::filesystem::exists(basepath+"/depth") || !boost::filesystem::exists(basepath+"/image"))
    {
        std::cout << "Subdiretories not found" << std::endl;
        throw std::exception();
    }

    boost::filesystem::create_directory(basepath+"/sparse_depth");
    int num_files = boost::distance(boost::filesystem::directory_iterator(basepath+"/image"));
    std::cout << "Found  " << num_files << " files: " <<  std::endl;

    // OrbExtractor Parameters
    const int nFeatures = 5000;
    const float fScaleFactor = 1.2;
    const int nLevels = 8;
    const int iIniThFAST = 20;
    const int iMinThFAST = 7;
    
    cslam::extractorptr mpORBextractor;
    mpORBextractor.reset(new cslam::ORBextractor(nFeatures,fScaleFactor,nLevels,iIniThFAST,iMinThFAST));

    for(int j = 0; j < num_files; j++)
    {
        int i = j +55101;
        cv::Mat image_gray;
        cv::Mat image_rgb = cv::imread(basepath + "/image/" + std::to_string(i) + ".jpg", cv::IMREAD_COLOR);
        cv::cvtColor(image_rgb, image_rgb, cv::COLOR_BGR2RGB);
        cv::cvtColor(image_rgb, image_gray,  cv::COLOR_RGB2GRAY);

        // Read depth matrix from npy file
        cnpy::NpyArray arr = cnpy::npy_load(basepath + "/depth/" + std::to_string(i) + ".npy");
        float* data = arr.data<float>();
        // Mats are saved as 480x640 in the npy
        cv::Mat depth_matrix = cv::Mat(arr.shape[0], arr.shape[1],CV_32FC1, data);

        // ORB extraction
        std::vector<cv::KeyPoint> mvKeys1;
        cv::Mat mDescriptors1;
        (*mpORBextractor)(image_gray,cv::Mat(),mvKeys1,mDescriptors1);

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(mvKeys1.begin(), mvKeys1.end(), g);
        std::uniform_int_distribution<> distr(200, 400);

        int num_kp = distr(g);
        
        cv::Mat sparse_depth = cv::Mat::zeros(image_gray.size(), CV_32FC1);

        int x, y;
        int kp_added = 0;
        for(cv::KeyPoint kp : mvKeys1)
        {
            if(kp_added == num_kp){break;}

            x = (int)floor(kp.pt.x);
            y = (int)floor(kp.pt.y);

            if(depth_matrix.at<float>(y,x) != 0.f && sparse_depth.at<float>(y,x) == 0.f)
            {
                sparse_depth.at<float>(y,x) = depth_matrix.at<float>(y,x);
                kp_added++;
            }
        }
        std::cout << i << std::endl;
        // Debugging
        /*
        int num_non_zero2 = cv::countNonZero(sparse_depth); 
        double minVal; 
        double maxVal; 
        Point minLoc; 
        Point maxLoc;
        cv::minMaxLoc( sparse_depth, &minVal, &maxVal, &minLoc, &maxLoc );
        std::cout << i << std::endl;
        std::cout << "Num kps " << num_kp << std::endl;
        std::cout << "Non zero "<< num_non_zero2 << std::endl;
        std::cout << "min val: " << minVal << std::endl;
        std::cout << "max val: " << maxVal << std::endl;

        cv::Mat image_rgbd = cv::Mat(image_rgb.rows,image_rgb.cols,CV_32FC4);
        image_rgb.convertTo(image_rgb, CV_32F);

        std::vector<cv::Mat> src{image_rgb, sparse_depth};
        cv::merge(src, image_rgbd);
        
        cnpy::npy_save(basepath + "/sparse_depth/rgbd_"+ std::to_string(i) +".npy", (float*)image_rgbd.data, {480,640,4},"w");
        */
        // The pointer returned by sparse_depth.data needs to be cast into a float pointer as its a uchar pointer otherwhise
        // which results in the memory being read incorrectly (float interpreted as ints)
        cnpy::npy_save(basepath + "/sparse_depth/sparse_"+ std::to_string(i) +".npy", (float*)sparse_depth.data, {480,640},"w");
    }

}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "ORBExtractor");

    //cslam::orbextraction();
    
    if(argc > 1){
        sampleORB(argv[1]);
    }
    else{
        std::cout << "Provide a path to the root directory of the images / depth files" << std::endl;
    }
    
    return 0;
}


