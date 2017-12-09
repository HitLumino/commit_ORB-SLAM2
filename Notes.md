# ORBSLAM 程序分析

    主要针对ORB-SLAM2代码的研读，作应有的总结与思考。
        伏久者飞必高，开先者谢独早。
        知此，可以免蹭蹬之忧，可以消躁之念。

<!-- TOC -->

- [ORBSLAM 程序分析](#orbslam-%E7%A8%8B%E5%BA%8F%E5%88%86%E6%9E%90)
    - [Converter](#converter)
        - [主要函数接口/方法](#%E4%B8%BB%E8%A6%81%E5%87%BD%E6%95%B0%E6%8E%A5%E5%8F%A3%E6%96%B9%E6%B3%95)
        - [`Converter`源码解析](#converter%E6%BA%90%E7%A0%81%E8%A7%A3%E6%9E%90)
            - [Converter Tips:](#converter-tips)
            - [`Converter.h`](#converterh)
            - [`Converter.cpp`](#convertercpp)
    - [Frame](#frame)
        - [Frame函数接口/方法](#frame%E5%87%BD%E6%95%B0%E6%8E%A5%E5%8F%A3%E6%96%B9%E6%B3%95)
        - [Frame源码分析](#frame%E6%BA%90%E7%A0%81%E5%88%86%E6%9E%90)
            - [Frame Tips:](#frame-tips)
            - [frame.h](#frameh)
    - [KeyFrameDatabase](#keyframedatabase)
        - [KeyFrameDatabase方法与函数接口](#keyframedatabase%E6%96%B9%E6%B3%95%E4%B8%8E%E5%87%BD%E6%95%B0%E6%8E%A5%E5%8F%A3)
            - [KeyFrameDatabase公有成员函数](#keyframedatabase%E5%85%AC%E6%9C%89%E6%88%90%E5%91%98%E5%87%BD%E6%95%B0)
            - [KeyFrameDatabase公有成员变量](#keyframedatabase%E5%85%AC%E6%9C%89%E6%88%90%E5%91%98%E5%8F%98%E9%87%8F)
            - [KeyFrameDatabase私有成员函数](#keyframedatabase%E7%A7%81%E6%9C%89%E6%88%90%E5%91%98%E5%87%BD%E6%95%B0)
            - [KeyFrameDatabase私有成员变量](#keyframedatabase%E7%A7%81%E6%9C%89%E6%88%90%E5%91%98%E5%8F%98%E9%87%8F)
        - [KeyFrameDatabase源码分析](#keyframedatabase%E6%BA%90%E7%A0%81%E5%88%86%E6%9E%90)
    - [KeyFrame](#keyframe)
        - [KeyFrame方法与函数接口](#keyframe%E6%96%B9%E6%B3%95%E4%B8%8E%E5%87%BD%E6%95%B0%E6%8E%A5%E5%8F%A3)
            - [KeyFrame公有成员函数](#keyframe%E5%85%AC%E6%9C%89%E6%88%90%E5%91%98%E5%87%BD%E6%95%B0)
            - [KeyFrame公有成员变量](#keyframe%E5%85%AC%E6%9C%89%E6%88%90%E5%91%98%E5%8F%98%E9%87%8F)
            - [KeyFrame私有成员函数](#keyframe%E7%A7%81%E6%9C%89%E6%88%90%E5%91%98%E5%87%BD%E6%95%B0)
            - [KeyFrame私有成员变量](#keyframe%E7%A7%81%E6%9C%89%E6%88%90%E5%91%98%E5%8F%98%E9%87%8F)
        - [KeyFrame源码分析](#keyframe%E6%BA%90%E7%A0%81%E5%88%86%E6%9E%90)
            - [KeyFrame Tips：](#keyframe-tips%EF%BC%9A)
            - [KeyFrame.cpp](#keyframecpp)
    - [Map](#map)
        - [Map方法与函数接口](#map%E6%96%B9%E6%B3%95%E4%B8%8E%E5%87%BD%E6%95%B0%E6%8E%A5%E5%8F%A3)
            - [公有成员函数](#%E5%85%AC%E6%9C%89%E6%88%90%E5%91%98%E5%87%BD%E6%95%B0)
            - [公有成员变量](#%E5%85%AC%E6%9C%89%E6%88%90%E5%91%98%E5%8F%98%E9%87%8F)
            - [私有函数](#%E7%A7%81%E6%9C%89%E5%87%BD%E6%95%B0)
            - [私有变量](#%E7%A7%81%E6%9C%89%E5%8F%98%E9%87%8F)
        - [Map源码分析](#map%E6%BA%90%E7%A0%81%E5%88%86%E6%9E%90)
            - [Map Tips:](#map-tips)
    - [MapPoint](#mappoint)
        - [MapPoint方法与函数接口](#mappoint%E6%96%B9%E6%B3%95%E4%B8%8E%E5%87%BD%E6%95%B0%E6%8E%A5%E5%8F%A3)
            - [公有成员函数](#%E5%85%AC%E6%9C%89%E6%88%90%E5%91%98%E5%87%BD%E6%95%B0)
            - [公有成员变量](#%E5%85%AC%E6%9C%89%E6%88%90%E5%91%98%E5%8F%98%E9%87%8F)
            - [私有函数](#%E7%A7%81%E6%9C%89%E5%87%BD%E6%95%B0)
            - [私有变量](#%E7%A7%81%E6%9C%89%E5%8F%98%E9%87%8F)
        - [MapPoint源码分析](#mappoint%E6%BA%90%E7%A0%81%E5%88%86%E6%9E%90)
            - [MapPoint.h](#mappointh)
    - [ORBmatcher](#orbmatcher)
        - [ORBmatcher方法与函数接口](#orbmatcher%E6%96%B9%E6%B3%95%E4%B8%8E%E5%87%BD%E6%95%B0%E6%8E%A5%E5%8F%A3)
            - [公有成员函数](#%E5%85%AC%E6%9C%89%E6%88%90%E5%91%98%E5%87%BD%E6%95%B0)
            - [公有成员变量](#%E5%85%AC%E6%9C%89%E6%88%90%E5%91%98%E5%8F%98%E9%87%8F)
            - [私有函数](#%E7%A7%81%E6%9C%89%E5%87%BD%E6%95%B0)
            - [私有变量](#%E7%A7%81%E6%9C%89%E5%8F%98%E9%87%8F)
    - [主要线程](#%E4%B8%BB%E8%A6%81%E7%BA%BF%E7%A8%8B)
        - [system入口](#system%E5%85%A5%E5%8F%A3)
            - [主要函数](#%E4%B8%BB%E8%A6%81%E5%87%BD%E6%95%B0)
        - [Tracking 线程](#tracking-%E7%BA%BF%E7%A8%8B)
            - [Tracking公有函数](#tracking%E5%85%AC%E6%9C%89%E5%87%BD%E6%95%B0)
            - [Tracking公有成员函数](#tracking%E5%85%AC%E6%9C%89%E6%88%90%E5%91%98%E5%87%BD%E6%95%B0)
            - [Tracking私有成员函数](#tracking%E7%A7%81%E6%9C%89%E6%88%90%E5%91%98%E5%87%BD%E6%95%B0)
            - [Tracking私有变量](#tracking%E7%A7%81%E6%9C%89%E5%8F%98%E9%87%8F)
        - [LocalMapping线程](#localmapping%E7%BA%BF%E7%A8%8B)
            - [步骤:](#%E6%AD%A5%E9%AA%A4)
        - [LoopClosing线程](#loopclosing%E7%BA%BF%E7%A8%8B)
            - [Tips：](#tips%EF%BC%9A)
            - [步骤](#%E6%AD%A5%E9%AA%A4)

<!-- /TOC -->

## Converter

### 主要函数接口/方法
`orb`中以`cv::Mat`为基本存储结构，到`g2o`和`Eigen`需要一个转换, 这些转换都很简单，整个文件可以单独从`orbslam`里抽出来而不影响其他功能。

- 描述子矩阵转化一维描述子向量
    + `static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);`
- `cv::Mat` 转化成 `g2o::SE3Quat(SE3)`
    + `static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);`
- `g2o::SE3Quat(SE3)`/`Eigen` 转化成 `CvMat`
    + `static cv::Mat toCvMat(const g2o::SE3Quat &SE3);`
    + `static cv::Mat toCvMat(const g2o::Sim3 &Sim3);`
    + `static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);`
    + `static cv::Mat toCvMat(const Eigen::Matrix3d &m);`
    + `static cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);`
    + `static cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);`
- `cv::Mat` 转化成 `Eigen`
    + `static Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);`
    + `static Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);`
    + `static Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);`
    + `static std::vector<float> toQuaternion(const cv::Mat &M);`

### `Converter`源码解析

#### Converter Tips:
* 有关`std::vector`的方法需要好好回顾一下
    * [`std::vector::reserve`](http://www.cplusplus.com/reference/vector/vector/reserve/)
    * [`std::vector::capacity`](http://www.cplusplus.com/reference/vector/vector/capacity/)
* 总结`g2o::SE3Quat（SE3）`的参数构成，以及赋值/初始化方式: `g2o::SE3Quat(R,t)`
* 总结Eigen::Matrix赋值/初始化方式: `<<`
* `g2o::SE3Quat`的方法总结 (主要在`se3quat.h`）
    + `map`//SE.map(v),映射
    + `log`//vector to se3
    + `exp`//se3 to vector
    + `to_homogeneous_matrix()`//转换为Matrix
    + `operator Eigen::Isometry3d() const`//cast SE3Quat into an Eigen::Isometry3d


#### `Converter.h`

```c
namespace ORB_SLAM2
{

/**
 * @brief 提供了一些常见的转换
 * 
 * orb中以cv::Mat为基本存储结构，到g2o和Eigen需要一个转换
 * 这些转换都很简单，整个文件可以单独从orbslam里抽出来而不影响其他功能
 */
class Converter
{
public:
    /**
     * @brief 一个描述子矩阵到一串单行的描述子向量
     */
    static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);//静态方法

    /**
     * @name toSE3Quat
     */
    ///@{
    /** cv::Mat to g2o::SE3Quat */
    //静态工作方法
    static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);//cv::Mat to SE3Quat
    /** unimplemented 未实现的*/
    static g2o::SE3Quat toSE3Quat(const g2o::Sim3 &gSim3);
    ///@}

    /**
     * @name toCvMat
     */
    ///@{
    static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
    static cv::Mat toCvMat(const g2o::Sim3 &Sim3);
    static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
    static cv::Mat toCvMat(const Eigen::Matrix3d &m);
    static cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
    static cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);
    ///@}

    /**
     * @name toEigen
     */
    ///@{
    static Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);
    static Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);
    static Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);
    static std::vector<float> toQuaternion(const cv::Mat &M);
    ///@}
};

}// namespace ORB_SLAM
```

#### `Converter.cpp`

```c
#include "Converter.cpp"

namespace ORB_SLAM2
{

std::vector<cv::Mat> Converter::toDescriptorVector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);//在创建容器后，第一时间为容器分配足够大的空间，避免重新分配内存
    for (int j=0;j<Descriptors.rows;j++)
        vDesc.push_back(Descriptors.row(j));//也就是将描述子转化成<vector>类型保存。

    return vDesc;
}
//总结g2o::SE3Quat的参数构成，以及赋值/初始化方式
g2o::SE3Quat Converter::toSE3Quat(const cv::Mat &cvT)
{
    Eigen::Matrix<double,3,3> R;
    R << cvT.at<float>(0,0), cvT.at<float>(0,1), cvT.at<float>(0,2),
         cvT.at<float>(1,0), cvT.at<float>(1,1), cvT.at<float>(1,2),
         cvT.at<float>(2,0), cvT.at<float>(2,1), cvT.at<float>(2,2);

    Eigen::Matrix<double,3,1> t(cvT.at<float>(0,3), cvT.at<float>(1,3), cvT.at<float>(2,3));

    return g2o::SE3Quat(R,t);
}

cv::Mat Converter::toCvMat(const g2o::SE3Quat &SE3)
{
    Eigen::Matrix<double,4,4> eigMat = SE3.to_homogeneous_matrix();//SE3 to matrix
    return toCvMat(eigMat);
}

cv::Mat Converter::toCvMat(const g2o::Sim3 &Sim3)
{
    Eigen::Matrix3d eigR = Sim3.rotation().toRotationMatrix();//获取旋转矩阵->Eigen矩阵型矩阵
    Eigen::Vector3d eigt = Sim3.translation();
    double s = Sim3.scale();
    return toCvSE3(s*eigR,eigt);
}

cv::Mat Converter::toCvMat(const Eigen::Matrix<double,4,4> &m)
{
    cv::Mat cvMat(4,4,CV_32F);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat Converter::toCvMat(const Eigen::Matrix3d &m)
{
    cv::Mat cvMat(3,3,CV_32F);
    for(int i=0;i<3;i++)
        for(int j=0; j<3; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat Converter::toCvMat(const Eigen::Matrix<double,3,1> &m)
{
    cv::Mat cvMat(3,1,CV_32F);
    for(int i=0;i<3;i++)
            cvMat.at<float>(i)=m(i);

    return cvMat.clone();
}

cv::Mat Converter::toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t)
{
    cv::Mat cvMat = cv::Mat::eye(4,4,CV_32F);
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            cvMat.at<float>(i,j)=R(i,j);
        }
    }
    for(int i=0;i<3;i++)
    {
        cvMat.at<float>(i,3)=t(i);
    }

    return cvMat.clone();
}

// 将OpenCVS中Mat类型的向量转化为Eigen中Matrix类型的变量
Eigen::Matrix<double,3,1> Converter::toVector3d(const cv::Mat &cvVector)
{
    Eigen::Matrix<double,3,1> v;
    v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

    return v;
}

Eigen::Matrix<double,3,1> Converter::toVector3d(const cv::Point3f &cvPoint)
{
    Eigen::Matrix<double,3,1> v;
    v << cvPoint.x, cvPoint.y, cvPoint.z;

    return v;
}

Eigen::Matrix<double,3,3> Converter::toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
         cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
         cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

std::vector<float> Converter::toQuaternion(const cv::Mat &M)
{
    Eigen::Matrix<double,3,3> eigMat = toMatrix3d(M);
    Eigen::Quaterniond q(eigMat);

    std::vector<float> v(4);
    v[0] = q.x();
    v[1] = q.y();
    v[2] = q.z();
    v[3] = q.w();

    return v;
}

} //namespace ORB_SLAM
```
## Frame

### Frame函数接口/方法

* 公有成员变量
    + `FRAME_GRID_ROWS` 48
    + `FRAME_GRID_COLS` 64
    + ORBVocabulary* mpORBvocabulary;// Vocabulary used for relocalization.
    + ORBextractor* mpORBextractorLeft, *mpORBextractorRight;//双目
    + double mTimeStamp;//时间戳
    + cv::Mat mK;
        - static float fx;
        - static float fy;
        - static float cx;
        - static float cy;
        - static float invfx;
        - static float invfy;
    + cv::Mat mDistCoef;
    + float mbf;//双目基线（像素单位）
    + float mb;//双目基线（米单位）
    + float mThDepth;//深度阈值（远点/近点）
    + int N; //< KeyPoints数量
    + `std::vector<cv::KeyPoint>` mvKeys, mvKeysRight;//原始左图像提取出的特征点（未校正）
    + `std::vector<cv::KeyPoint>` mvKeysUn;//校正mvKeys后的特征点,对于双目摄像头，一般得到的图像都是校正好的，再校正一次有点多余
    + `std::vector<float> mvuRight`;//对于双目，mvuRight存储了左目像素点在右目中的对应点的横坐标
    + `std::vector<float>` mvDepth;// mvDepth对应的深度
    + DBoW2::BowVector mBowVec;//Bag of Words Vector structures.
    + DBoW2::FeatureVector mFeatVec;
    + cv::Mat mDescriptors, mDescriptorsRight;// 左目摄像头和右目摄像头特征点对应的描述子
    + std::vector<MapPoint*> mvpMapPoints;// 每个特征点对应的MapPoint
    + `std::vector<bool>` mvbOutlier;// 观测不到Map中的3D点
    + static float mfGridElementWidthInv;// 坐标乘以mfGridElementWidthInv和mfGridElementHeightInv就可以确定在哪个格子
    + static float mfGridElementHeightInv;
    + `std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS]`;//每个格子分配的特征点数，将图像分成格子，保证提取的特征点比较均匀
    + static long unsigned int nNextId; ///< Next Frame id.
    + long unsigned int mnId; ///< Current Frame id.
    + KeyFrame* mpReferenceKF;//指针，指向参考关键帧
    + int mnScaleLevels;//图像提金字塔的层数
    + float mfScaleFactor;//图像提金字塔的尺度因子
    + float mfLogScaleFactor;
    + `vector<float>` mvScaleFactors;//图像提金字塔的层数
    + `vector<float>` mvInvScaleFactors;//图像提金字塔的尺度因子
    + `vector<float>` mvLevelSigma2;
    + `vector<float>` mvInvLevelSigma2;
    + static float mnMinX;
    + static float mnMaxX;
    + static float mnMinY;
    + static float mnMaxY;
    + cv::Mat mTcw; ///< 相机姿态 世界坐标系到相机坐标坐标系的变换矩阵
* 私有成员变量
    + cv::Mat mRcw; ///< Rotation from world to camera
    + cv::Mat mtcw;///< Translation from world to camera
    + cv::Mat mRwc; ///< Rotation from camera to world
    + cv::Mat mOw;  ///光心
* 构造函数
    + `Frame(const Frame &frame);`// Copy constructor.
    + `Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);`//双目
    + `Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);`//RBGD
    + `Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);`//单目
* 公有方法
    + void ExtractORB(int flag, const cv::Mat &im);//提取的关键点
    + void ComputeBoW();// Compute Bag of Words representation
    + void SetPose(cv::Mat Tcw);
    + void UpdatePoseMatrices();
    + inline cv::Mat GetCameraCenter()；
    + bool isInFrustum(MapPoint* pMP, float viewingCosLimit);// 判断路标点是否在视野中
    + bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);//(return false if outside the grid)
    + void ComputeStereoMatches();
    + void ComputeStereoFromRGBD(const cv::Mat &imDepth);
    + cv::Mat UnprojectStereo(const int &i);
* 私有方法
    + void UndistortKeyPoints();// Only for the RGB-D case. Stereo must be already rectified!（called in the constructor）
    + void ComputeImageBounds(const cv::Mat &imLeft);// Computes image bounds for the undistorted image (called in the constructor).
    + void AssignFeaturesToGrid();// Assign keypoints to the grid for speed up feature matching (called in the constructor).

### Frame源码分析
#### Frame Tips:
* [线程执行函数对引用参数的修改实际上是对对象的副本进行的修改，修改并不会反应到原始的对象上。那么如何传参呢？需要注意什么呢？（章节2.2传递参数给线程函数）](https://github.com/HitLumino/studynotes/blob/master/Cpp_Concurrency_In_Action.pdf)
* void Frame::UndistortKeyPoints()里`cv::undistortPoints`自带失真矫正的用法

1. 双目初始化
    * 同时对左右目提特征（两个线程）
    * 不需要对特征点进行矫正（双目特有）
    * 计算双目间的匹配, 匹配成功的特征点会计算其深度
        * 为左图的每一个特征点在右图中找到匹配点 
        * 根据基线(有冗余范围)上描述子距离找到匹配, 再进行SAD精确定位 
        * 最后对所有SAD的值进行排序, 剔除SAD值较大的匹配对，然后利用抛物线拟合得到亚像素精度的匹配 
        * 匹配成功后会更新 mvuRight(ur) 和 mvDepth(Z)
```c
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mb(0), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    // 同时对左右目提特征
    //使用引用函数传递  前提是提供一个合适的对象指针作为第一个参数，这里用的是this指的是-->Frame类对象
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);//注意这个‘this’
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();
///注释1: Frame::ExtractORB.线程传参（int, cv::Mat）
/*void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}*/
    N = mvKeys.size();

    if(mvKeys.empty())
        return;
    // Undistort特征点，这里没有对双目进行校正，因为要求输入的图像已经进行极线校正
    //https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=undistortpoints#undistortpoints
    UndistortKeyPoints();

    // 计算双目间的匹配, 匹配成功的特征点会计算其深度
    // 深度存放在mvuRight 和 mvDepth 中
    ComputeStereoMatches();

    // 对应的mappoints
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));   
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}
```

#### frame.h
```c

#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;

class Frame
{
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for stereo cameras.
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for RGB-D cameras.
    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for Monocular cameras.
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Extract ORB on the image. 0 for left image and 1 for right image.
    // 提取的关键点存放在mvKeys和mDescriptors中
    // ORB是直接调orbExtractor提取的
    void ExtractORB(int flag, const cv::Mat &im);

    // Compute Bag of Words representation.
    // 存放在mBowVec中
    void ComputeBoW();

    // Set the camera pose.
    // 用Tcw更新mTcw
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline cv::Mat GetCameraCenter()
	{
        return mOw.clone();
    }

    // Returns inverse of rotation
    inline cv::Mat GetRotationInverse()
	{
        return mRwc.clone();
    }

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    // 判断路标点是否在视野中
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;

    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    void ComputeStereoMatches();

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    cv::Mat UnprojectStereo(const int &i);

public:
    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;//深度阈值

    // Number of KeyPoints.
    int N; //< KeyPoints数量

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    // mvKeys:原始左图像提取出的特征点（未校正）
    // mvKeysRight:原始右图像提取出的特征点（未校正）
    // mvKeysUn:校正mvKeys后的特征点，对于双目摄像头，一般得到的图像都是校正好的，再校正一次有点多余
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
    std::vector<cv::KeyPoint> mvKeysUn;

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    // 对于双目，mvuRight存储了左目像素点在右目中的对应点的横坐标
    // mvDepth对应的深度
    // 单目摄像头，这两个容器中存的都是-1
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.
    // 左目摄像头和右目摄像头特征点对应的描述子
    cv::Mat mDescriptors, mDescriptorsRight;

    // MapPoints associated to keypoints, NULL pointer if no association.
    // 每个特征点对应的MapPoint
    std::vector<MapPoint*> mvpMapPoints;

    // Flag to identify outlier associations.
    // 观测不到Map中的3D点
    std::vector<bool> mvbOutlier;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    // 坐标乘以mfGridElementWidthInv和mfGridElementHeightInv就可以确定在哪个格子
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    // 每个格子分配的特征点数，将图像分成格子，保证提取的特征点比较均匀
    // FRAME_GRID_ROWS 48
    // FRAME_GRID_COLS 64
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Camera pose.
    cv::Mat mTcw; ///< 相机姿态 世界坐标系到相机坐标坐标系的变换矩阵

    // Current and Next Frame id.
    static long unsigned int nNextId; ///< Next Frame id.
    long unsigned int mnId; ///< Current Frame id.

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;//指针，指向参考关键帧

    // Scale pyramid info.
    int mnScaleLevels;//图像提金字塔的层数
    float mfScaleFactor;//图像提金字塔的尺度因子
    float mfLogScaleFactor;//
    vector<float> mvScaleFactors;
    vector<float> mvInvScaleFactors;
    vector<float> mvLevelSigma2;
    vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once).
    // 用于确定画格子时的边界
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;


private:

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    // Rotation, translation and camera center
    cv::Mat mRcw; ///< Rotation from world to camera
    cv::Mat mtcw; ///< Translation from world to camera
    cv::Mat mRwc; ///< Rotation from camera to world
    cv::Mat mOw;  ///< mtwc,Translation from camera to world
};

}// namespace ORB_SLAM

#endif // FRAME_H
```
## KeyFrameDatabase
主要和词典相关的函数，比如关键帧的闭环检测、重定位等。
### KeyFrameDatabase方法与函数接口
#### KeyFrameDatabase公有成员函数
* KeyFrameDatabase(const ORBVocabulary &voc);
```c
KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size()); // number of words
}
```
* void add(KeyFrame* pKF);//根据关键帧的词包，更新数据库的倒排索引
```c
void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);

    // 为每一个word添加该KeyFrame
    //std::map<WordId, WordValue>
    //pKF->mBowVec:  Vector of words to represent images  
    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);
}
``` 
* void erase(KeyFrame* pKF);
```c
void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    // 每一个KeyFrame包含多个words，遍历mvInvertedFile中的这些words，然后在word中删除该KeyFrame
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            if(pKF==*lit)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}
```
* void clear();
```c
void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();// mvInvertedFile[i]表示包含了第i个word id的所有关键帧
    mvInvertedFile.resize(mpVoc->size());// mpVoc：预先训练好的词典
}
```
* std::vector<KeyFrame *> DetectLoopCandidates(KeyFrame* pKF, float minScore);// Loop Detection
    * 1. 找出和当前帧具有公共单词的所有关键帧（不包括与当前帧相连的关键帧）
    * 2. 只和具有共同单词较多的关键帧进行相似度计算
    * 3. 将与关键帧相连（权值最高）的前十个关键帧归为一组，计算累计得分
    * 4. 只返回累计得分较高的组中分数最高的那几个关键帧

```c
/**
 * @brief 在闭环检测中找到与该关键帧可能闭环的关键帧
 *
 * 1. 找出和当前帧具有公共单词的所有关键帧（不包括与当前帧相连的关键帧）
 * 2. 只和具有共同单词较多的关键帧进行相似度计算
 * 3. 将与关键帧相连（权值最高）的前十个关键帧归为一组，计算累计得分
 * 4. 只返回累计得分较高的组中分数最高的关键帧
 * @param pKF      需要闭环的关键帧
 * @param minScore 相似性分数最低要求
 * @return         可能闭环的关键帧
 * @see III-E Bags of Words Place Recognition
 */
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
    // 提出所有与该pKF相连的KeyFrame，这些相连Keyframe都是局部相连，在闭环检测的时候将被剔除
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    list<KeyFrame*> lKFsSharingWords;// 用于保存可能与pKF形成回环的候选帧（只要有相同的word，且不属于局部相连帧）

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    // 步骤1：找出和当前帧具有公共单词的所有关键帧（不包括与当前帧链接的关键帧）
    {
        unique_lock<mutex> lock(mMutex);

        // words是检测图像是否匹配的枢纽，遍历该pKF的每一个word
        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            // 提取所有包含该word的KeyFrame
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnLoopQuery!=pKF->mnId)// pKFi还没有标记为pKF的候选帧
                {
                    pKFi->mnLoopWords=0;
                    if(!spConnectedKeyFrames.count(pKFi))// 与pKF局部链接的关键帧不进入闭环候选帧
                    {
                        pKFi->mnLoopQuery=pKF->mnId;// pKFi标记为pKF的候选帧，之后直接跳过判断
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->mnLoopWords++;// 记录pKFi与pKF具有相同word的个数
            }
        }
    }

    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    // 步骤2：统计所有闭环候选帧中与pKF具有共同单词最多的单词数
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnLoopWords>maxCommonWords)
            maxCommonWords=(*lit)->mnLoopWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    int nscores=0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    // 步骤3：遍历所有闭环候选帧，挑选出共有单词数大于minCommonWords且单词匹配度大于minScore存入lScoreAndMatch
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        // pKF只和具有共同单词较多的关键帧进行比较，需要大于minCommonWords
        if(pKFi->mnLoopWords>minCommonWords)
        {
            nscores++;// 这个变量后面没有用到

            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

            pKFi->mLoopScore = si;
            if(si>=minScore)
                lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    // 单单计算当前帧和某一关键帧的相似性是不够的，这里将与关键帧相连（权值最高，共视程度最高）的前十个关键帧归为一组，计算累计得分
    // 步骤4：具体而言：lScoreAndMatch中每一个KeyFrame都把与自己共视程度较高的帧归为一组，每一组会计算组得分并记录该组分数最高的KeyFrame，记录于lAccScoreAndMatch
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first; // 该组最高分数
        float accScore = it->first;  // 该组累计得分
        KeyFrame* pBestKF = pKFi;    // 该组最高分数对应的关键帧
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
            {
                accScore+=pKF2->mLoopScore;// 因为pKF2->mnLoopQuery==pKF->mnId，所以只有pKF2也在闭环候选帧中，才能贡献分数
                if(pKF2->mLoopScore>bestScore)// 统计得到组里分数最高的KeyFrame
                {
                    pBestKF=pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)// 记录所有组中组得分最高的组
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    // 步骤5：得到组得分大于minScoreToRetain的组，得到组中分数最高的关键帧 0.75*bestScore
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(it->first>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))// 判断该pKFi是否已经在队列中了
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpLoopCandidates;
}
```
* std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* F);// Relocalization
    * 和DetectLoopCandidates类似。
    * 1. 找出和当前帧具有公共单词的所有关键帧
    * 2. 只和具有共同单词较多的关键帧进行相似度计算
    * 3. 将与关键帧相连（权值最高）的前十个关键帧归为一组，计算累计得分
    * 4. 只返回累计得分较高的组中分数最高的那几个关键帧

#### KeyFrameDatabase公有成员变量
#### KeyFrameDatabase私有成员函数
* const ORBVocabulary* mpVoc; ///< 预先训练好的词典
#### KeyFrameDatabase私有成员变量
* std::mutex mMutex;
* std::vector<list<KeyFrame*> > mvInvertedFile; ///< 倒排索引，mvInvertedFile[i]表示包含了第i个word id的所有关键帧
### KeyFrameDatabase源码分析
见上。

## KeyFrame
需要前向声明：
```c
class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;
```
关键帧，和普通的Frame不一样，但是可以由Frame来构造，许多数据会被三个线程同时访问，所以用锁的地方很普遍。
### KeyFrame方法与函数接口

#### KeyFrame公有成员函数
* KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);  
* Pose functions(这里的set,get需要用到锁)
    * void SetPose(const cv::Mat &Tcw);
    * cv::Mat GetPose();
    * cv::Mat GetPoseInverse();
    * cv::Mat GetCameraCenter();//返回Ow.clone();相机中心
    * cv::Mat GetStereoCenter();//return Cw.clone();Cw = Twc*center
    * cv::Mat GetRotation();
    * cv::Mat GetTranslation();
* void ComputeBoW();//词袋
* Covisibility graph functions
    * void AddConnection(KeyFrame* pKF, const int &weight);
        * @param pKF    关键帧
        * @param weight 权重，该关键帧与pKF共同观测到的3d点数量
        * 添加关键帧，并且更新权重UpdateBestCovisibles()获得排序后的mvpOrderedConnectedKeyFrames、mvOrderedWeights
    * void EraseConnection(KeyFrame* pKF);
    * void UpdateConnections();
    * void UpdateBestCovisibles();
    * std::set<KeyFrame *> GetConnectedKeyFrames();//得到与该关键帧连接的关键帧,保存在set<KeyFrame*> s里
    * std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();//得到与该关键帧连接的关键帧(已按权值排序)
        * return mvpOrderedConnectedKeyFrames;根据权重排好序的
    * std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);//得到与该关键帧连接的前N个关键帧(已按权值排序)，如果连接的关键帧少于N，则返回所有连接的关键帧
        * return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);
    * std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);//得到与该关键帧连接的权重大于等于给定值w的关键帧（设定的共视帧）用到了[upper_bound](http://www.cnblogs.com/cobbliu/archive/2012/05/21/2512249.html)
        * return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);   
    * int GetWeight(KeyFrame* pKF);//得到该关键帧的权重
* Spanning tree functions最小生成树
    * void AddChild(KeyFrame* pKF);
    * void EraseChild(KeyFrame* pKF);
    * void ChangeParent(KeyFrame* pKF);
    * std::set<KeyFrame*> GetChilds();
    * KeyFrame* GetParent();
    * bool hasChild(KeyFrame* pKF);
* Loop Edges
    * void AddLoopEdge(KeyFrame* pKF);
    * std::set<KeyFrame*> GetLoopEdges();
* MapPoint observation functions
    * void AddMapPoint(MapPoint* pMP, const size_t &idx);
        * @param idx MapPoint在KeyFrame中的索引
        * mvpMapPoints[idx]=pMP;
    * void EraseMapPointMatch(const size_t &idx);
        * mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
    * void EraseMapPointMatch(MapPoint* pMP);
    * void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);//用新点代替原先的地图点
    * std::set<MapPoint*> GetMapPoints();
        * return set<MapPoint*> s;//地图中所有挑选过（优）点
    * std::vector<MapPoint*> GetMapPointMatches();//获取该关键帧的MapPoints
        * return mvpMapPoints;
    * int TrackedMapPoints(const int &minObs);//关键帧中，大于等于minObs的MapPoints的数量，minObs就是一个阈值，大于minObs就表示该MapPoint是一个高质量的MapPoint，一个高质量的MapPoint会被多个KeyFrame观测到。
        * return nPoints;//该关键帧中 优质点个数
    * MapPoint* GetMapPoint(const size_t &idx);
        * return mvpMapPoints;
* KeyPoint functions
    * std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
    * cv::Mat UnprojectStereo(int i);
* bool IsInImage(const float &x, const float &y) const;
* Enable/Disable bad flag changes
    * void SetNotErase();
    * void SetErase();
* Set/check bad flag
    * void SetBadFlag();
    * bool isBad();
* Compute Scene Depth (q=2 median). Used in monocular.
    * float ComputeSceneMedianDepth(const int q);
    * static bool weightComp( int a, int b);
    * static bool lId(KeyFrame* pKF1, KeyFrame* pKF2);
	
#### KeyFrame公有成员变量
* The following variables are accesed from only 1 thread or never change (no mutex needed).
    * static long unsigned int nNextId;// nNextID名字改为nLastID更合适，表示上一个KeyFrame的ID号
    * long unsigned int mnId;// 在nNextID的基础上加1就得到了mnID，为当前KeyFrame的ID号
    * const long unsigned int mnFrameId;// mnFrameId记录了该KeyFrame是由哪个Frame初始化的
    * const double mTimeStamp;
* Grid (to speed up feature matching)和Frame类中的定义相同
    * const int mnGridCols;
    * const int mnGridRows;
    * const float mfGridElementWidthInv;
    * const float mfGridElementHeightInv;
* Variables used by the tracking
    * long unsigned int mnTrackReferenceForFrame;
    * long unsigned int mnFuseTargetForKF;
* Variables used by the local mapping
    * long unsigned int mnBALocalForKF;
    * long unsigned int mnBAFixedForKF;
* Variables used by the keyframe database
    * long unsigned int mnLoopQuery;
    * int mnLoopWords;
    * float mLoopScore;
    * long unsigned int mnRelocQuery;
    * int mnRelocWords;
    * float mRelocScore;
* Variables used by loop closing
    * cv::Mat mTcwGBA;
    * cv::Mat mTcwBefGBA;
    * long unsigned int mnBAGlobalForKF;
* Calibration parameters
    * const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;
* Number of KeyPoints
    * const int N;
* KeyPoints, stereo coordinate and descriptors (all associated by an index)(和Frame类中的定义相同)
    * const std::vector<cv::KeyPoint> mvKeys;
    * const std::vector<cv::KeyPoint> mvKeysUn;
    * const std::vector<float> mvuRight; // negative value for monocular points
    * const std::vector<float> mvDepth; // negative value for monocular points
    * const cv::Mat mDescriptors;

* BoW
    * DBoW2::BowVector mBowVec; ///< Vector of words to represent images
    * DBoW2::FeatureVector mFeatVec; ///< Vector of nodes with indexes of local features

* Pose relative to parent (this is computed when bad flag is activated)
    * cv::Mat mTcp;

* Scale
    * const int mnScaleLevels;
    * const float mfScaleFactor;
    * const float mfLogScaleFactor;
    * const std::vector<float> mvScaleFactors;// 尺度因子，scale^n，scale=1.2，n为层数
    * const std::vector<float> mvLevelSigma2;// 尺度因子的平方
    * const std::vector<float> mvInvLevelSigma2;

* Image bounds and calibration
    * const int mnMinX;
    * const int mnMinY;
    * const int mnMaxX;
    * const int mnMaxY;
    * const cv::Mat mK;

#### KeyFrame私有成员函数

#### KeyFrame私有成员变量
The following variables need to be accessed trough a mutex to be thread safe.
* SE3 Pose and camera center
    * cv::Mat Tcw;
    * cv::Mat Twc;
    * cv::Mat Ow;//相机中心
    * cv::Mat Cw; // Stereo middel point. Only for visualization
* MapPoints associated to keypoints
    * std::vector<MapPoint*> mvpMapPoints;
* BoW
    * KeyFrameDatabase* mpKeyFrameDB;
    * ORBVocabulary* mpORBvocabulary;
* Grid over the image to speed up feature matching
    * std::vector< std::vector <std::vector<size_t> > > mGrid;
* Covisibility Graph
    * std::map<KeyFrame*,int> mConnectedKeyFrameWeights; ///< 与该关键帧连接的关键帧与权重
    * std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames; ///< 排序后的关键帧
    * std::vector<int> mvOrderedWeights; ///< 排序后的权重(从大到小)
* Spanning Tree and Loop Edges std::set是集合，相比vector，进行插入数据这样的操作时会自动排序
    * bool mbFirstConnection;
    * KeyFrame* mpParent;
    * std::set<KeyFrame*> mspChildrens;
    * std::set<KeyFrame*> mspLoopEdges;
* Bad flags
    * bool mbNotErase;
    * bool mbToBeErased;
    * bool mbBad;    
    * float mHalfBaseline; // Only for visualization
    * Map* mpMap;
* 线程
    * std::mutex **mMutexPose**;
    * std::mutex **mMutexConnections**;
    * std::mutex **mMutexFeatures**;

### KeyFrame源码分析
* 构造函数

#### KeyFrame Tips：
*  std::map::count函数只可能返回0或1两种情况
*  [upper_bound与lower_bound的用法](http://www.cnblogs.com/cobbliu/archive/2012/05/21/2512249.html)
#### KeyFrame.cpp
```c
long unsigned int KeyFrame::nNextId=0;
```

## Map
地图主要由关键帧和地图点组成。  
前向声明： "MapPoint.h"、"KeyFrame.h"  
主要内容包括：添加/剔除关键帧；添加/剔除地图点；获取所有关键帧/地图点等等
### Map方法与函数接口

#### 公有成员函数
* Map();  
`Map::Map():mnMaxKFid(0){}`
* void AddKeyFrame(KeyFrame* pKF);//添加关键帧
```c
void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);//锁住mMutexMap这个互斥元，如果已经被其他unique_lock锁住，暂时阻塞
    mspKeyFrames.insert(pKF);
    if(pKF->mnId>mnMaxKFid)
        mnMaxKFid=pKF->mnId;//更新地图最新关键帧的id
}
```
* void AddMapPoint(MapPoint* pMP);//添加地图点
```c
    unique_lock<mutex> lock(mMutexMap);//锁住mMutexMap
    mspMapPoints.insert(pMP);
```
* void EraseMapPoint(MapPoint* pMP);//剔除地图点
```c
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);
```
* void EraseKeyFrame(KeyFrame* pKF);//剔除关键帧
```c
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);
```
* void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);//设置参考MapPoints，将用于DrawMapPoints函数画图
* std::vector<KeyFrame*> GetAllKeyFrames();//获取所有关键帧
    * return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
* std::vector<MapPoint*> GetAllMapPoints();//获取所有地图点
    * return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
* std::vector<MapPoint*> GetReferenceMapPoints();//获取参考地图点
    * return mvpReferenceMapPoints;
* long unsigned int MapPointsInMap();//返回地图中地图点的个数
    * return mspMapPoints.size();
* long unsigned  KeyFramesInMap();//返回地图中关键帧的个数
    * return mspKeyFrames.size();
* long unsigned int GetMaxKFid();//return mnMaxKFid; 地图中目前最新关键帧的id
    * return mnMaxKFid;
* void clear();//清除地图
#### 公有成员变量
* vector<KeyFrame*> mvpKeyFrameOrigins;
* **std::mutex mMutexMapUpdate**;
* **std::mutex mMutexPointCreation**;/// This avoid that two points are created simultaneously in separate threads (id conflict)

#### 私有函数

#### 私有变量
* std::set<MapPoint*> mspMapPoints; ///< MapPoints
* std::set<KeyFrame*> mspKeyFrames; ///< Keyframs
* std::vector<MapPoint*> mvpReferenceMapPoints;
* long unsigned int mnMaxKFid;
* **std::mutex mMutexMap**;

### Map源码分析

#### Map Tips:

* [新创建的 unique_lock 对象管理 Mutex 对象 m，并尝试调用 m.lock() 对 Mutex 对象进行上锁，如果此时另外某个 unique_lock 对象已经管理了该 Mutex 对象 m，则当前线程将会被阻塞。](http://www.cnblogs.com/haippy/p/3346477.html)   
* [unique_lock与std::lock_guard区别: unique_lock 不一定要拥有 mutex，所以可以透过 default constructor 建立出一个空的 unique_lock。](http://blog.csdn.net/liuxuejiang158blog/article/details/17263353)

**std::unique_lock构造函数**

default(1)  | unique_lock() noexcept;                                 |
------------|---------------------------------------------------------|
locking(2)  | explicit unique_lock(mutex_type& m);                    |
try-locking | unique_lock(mutex_type& m, try_to_lock_t tag);          |
deferred (4)| unique_lock(mutex_type& m, defer_lock_t tag) noexcept;  |
adopting (5)|unique_lock(mutex_type& m, adopt_lock_t tag);            |

    1. 新创建的 unique_lock 对象不管理任何 Mutex 对象。
    2. 新创建的 unique_lock 对象管理 Mutex 对象 m，并尝试调用 m.lock() 对 Mutex 对象进行上锁，如果此时另外某个 unique_lock 对象已经管理了该 Mutex 对象 m，则当前线程将会被阻塞。
    3. 新创建的 unique_lock 对象管理 Mutex 对象 m，并尝试调用 m.try_lock() 对 Mutex 对象进行上锁，但如果上锁不成功，并不会阻塞当前线程。
    4. 新创建的 unique_lock 对象管理 Mutex 对象 m，但是在初始化的时候并不锁住 Mutex 对象。 m 应该是一个没有当前线程锁住的 Mutex 对象。
    5. 新创建的 unique_lock 对象管理 Mutex 对象 m， m 应该是一个已经被当前线程锁住的 Mutex 对象。(并且当前新创建的 unique_lock 对象拥有对锁(Lock)的所有权)。
**std::lock_guard 构造函数**
locking (1)	|explicit lock_guard (mutex_type& m);
-------------|------------------------------------
adopting (2)	|lock_guard (mutex_type& m, adopt_lock_t tag);
copy _[deleted]_(3)	|lock_guard (const lock_guard&) = delete;

     1.lock_guard 对象管理 Mutex 对象 m，并在构造时对 m 进行上锁（调用 m.lock()）。
     2. lock_guard 对象管理 Mutex 对象 m，与 locking 初始化(1) 不同的是， Mutex 对象 m 已被当前线程锁住。
     3. lock_guard 对象的拷贝构造和移动构造(move construction)均被禁用，因此 lock_guard 对象不可被拷贝构造或移动构造。
      
```c
void Map::AddKeyFrame(KeyFrame *pKF)//locking
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if(pKF->mnId>mnMaxKFid)
        mnMaxKFid=pKF->mnId;//更新地图最新关键帧的id
}
```

## MapPoint
1. 需要前向定义Frame/KeyFrame/Map
1. 用到互斥元mutex

### MapPoint方法与函数接口
最关键：
* std::map<KeyFrame*,size_t> mObservations; ///< 观测到该MapPoint的KF和该MapPoint在KF中的索引
#### 公有成员函数
* MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
    * 给定坐标与**关键帧keyframe**构造MapPoint
    * Pos      MapPoint的坐标（wrt世界坐标系)
    * pRefKF   KeyFrame
    * pMap     Map
```c
MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    mNormalVector = cv::Mat::zeros(3,1,CV_32F);// 该MapPoint平均观测方向
    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}
```
* MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF);
    * 给定坐标与**普通frame**构造MapPoint,普通frame是如何添加地图点的？
    * @param idxF   MapPoint在Frame中的索引，即对应的特征点的编号
```c
MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    cv::Mat Ow = pFrame->GetCameraCenter();
    mNormalVector = mWorldPos - Ow;// 世界坐标系下相机到3D点的向量
    mNormalVector = mNormalVector/cv::norm(mNormalVector);// 世界坐标系下相机到3D点的单位向量

    cv::Mat PC = Pos - Ow;
    const float dist = cv::norm(PC);//NORM_L2 距离
    const int level = pFrame->mvKeysUn[idxF].octave;//关键点的层数
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];//获得相应尺度因子
    const int nLevels = pFrame->mnScaleLevels;

    // 另见PredictScale函数前的注释
    mfMaxDistance = dist*levelScaleFactor;//当前层
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];//下一层

    // 见mDescriptor在MapPoint.h中的注释
    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);//将这个3D点在frame上的索引，找到该描述子，赋值给mDescriptor

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;//Global ID for MapPoint
}
```
* void SetWorldPos(const cv::Mat &Pos);
```c
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}
```
* cv::Mat GetWorldPos();//获取地图点的世界坐标系的坐标
```c
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
```
* cv::Mat GetNormal();//获取该点的平均观测方向
```c
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
```
* KeyFrame* GetReferenceKeyFrame();//获取参考关键帧？
```c
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
```
* void AddObservation(KeyFrame* pKF,size_t idx);
    * 记录哪些KeyFrame的那个特征点能观测到该MapPoint,并增加观测的相机数目nObs，单目+1，双目或者grbd+2,这个函数是建立关键帧共视关系的核心函数，能共同观测到某些MapPoints的关键帧是共视关键帧。
    * @param idx MapPoint在KeyFrame中的索引
    * @param pKF KeyFrame
```c
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return;
    // 记录下能观测到该MapPoint的KF和该MapPoint在KF中的索引
    //std::map<KeyFrame*,size_t> mObservations; ///< 观测到该MapPoint的KF和该MapPoint在KF中的索引
    mObservations[pKF]=idx;

    if(pKF->mvuRight[idx]>=0)//检测一下这个特征点是不是存在双目
        nObs+=2; // 双目或者rgbd  观测到的次数
    else
        nObs++; // 单目
}
```
* void EraseObservation(KeyFrame* pKF);//在这个关键帧中删除有关这个点观测数据
```c
void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            int idx = mObservations[pKF];
            if(pKF->mvuRight[idx]>=0)
                nObs-=2;
            else
                nObs--;

            mObservations.erase(pKF);

            // 如果该keyFrame是参考帧，该Frame被删除后重新指定RefFrame
            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            // 如果删除这个帧后，这个点表示观测到该点的相机数目少于2时，也就是说有可能再也没有人有你的记录了，那你也就GG了。
            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)
        SetBadFlag();
}
```
* void SetBadFlag();//宣布该点已作废。告知可以观测到该MapPoint的Frame，该MapPoint已被删除
```c
void MapPoint::SetBadFlag()
{
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;// 把mObservations转存到obs，obs和mObservations里存的是指针，赋值过程为浅拷贝
        mObservations.clear();// 把mObservations指向的内存释放，obs作为局部变量之后自动删除
    }
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapPointMatch(mit->second);// 告诉可以观测到该MapPoint的KeyFrame，该MapPoint被删了
    }

    mpMap->EraseMapPoint(this);// 擦除该MapPoint申请的内存
}
```
* std::map<KeyFrame*,size_t> GetObservations();
```c
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
```
* int Observations();//你被关键帧看到的次数。
 ```c
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
 ```

* int GetIndexInKeyFrame(KeyFrame* pKF);
    * return mObservations[pKF];
```c
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
```
* bool isBad();
* bool IsInKeyFrame(KeyFrame* pKF);
```c
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
```
```c
bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}
```
* void Replace(MapPoint* pMP);
```c
void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,size_t> obs;// 这一段和SetBadFlag函数相同
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();//释放
        mbBad=true;
        nvisible = mnVisible;//得到被看到的次数
        nfound = mnFound;
        mpReplaced = pMP;//你变成了“被替代的”
    }

    // 所有能观测到该MapPoint的keyframe都要替换
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;

        if(!pMP->IsInKeyFrame(pKF))//如果这个pKF没有看到pMP,那就让他看到。
            //具体的就是，地图点（旧点）在pKF中的映射索引对象换成新点
        {
            pKF->ReplaceMapPointMatch(mit->second, pMP);// 让KeyFrame用pMP替换掉原来的MapPoint
            pMP->AddObservation(pKF,mit->second);// 让MapPoint替换掉对应的KeyFrame
        }
        else
        {
            // 产生冲突，即pKF中有两个特征点a,b（这两个特征点的描述子是近似相同的），这两个特征点对应两个MapPoint为this,pMP
            // 然而在fuse的过程中pMP的观测更多，需要替换this，因此保留b与pMP的联系，去掉a与this的联系
            pKF->EraseMapPointMatch(mit->second);
        }
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this);
}
```
* MapPoint* GetReplaced();
```c
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;// (we do not currently erase MapPoint from memory)
```
* void IncreaseVisible(int n);
    * 该MapPoint在某些帧的视野范围内，通过Frame::isInFrustum()函数判断
```c
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
```
* void IncreaseFound(int n);
    * 能找到该点的帧数+n，n默认为1
```c
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
```
* float GetFoundRatio();
```c
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
```
* inline int GetFound()
* void ComputeDistinctiveDescriptors();//计算具有代表的描述子
    * 由于一个MapPoint会被许多相机观测到，因此在插入关键帧后，需要判断是否更新当前点的最适合的描述子,先获得当前点的所有描述子，然后计算描述子之间的两两距离，最好的描述子与其他描述子应该具有最小的距离中值
    * mDescriptor = vDescriptors[BestIdx].clone();//最好的描述子
```c
void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    map<KeyFrame*,size_t> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations=mObservations;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    // 遍历观测到3d点的所有关键帧，获得orb描述子，并插入到vDescriptors中
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad())
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them
    // 获得这些描述子两两之间的距离
    const size_t N = vDescriptors.size();
	
    //float Distances[N][N];
	std::vector<std::vector<float> > Distances;
	Distances.resize(N, vector<float>(N, 0));
	for (size_t i = 0; i<N; i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        // 第i个描述子到其它所有所有描述子之间的距离
        //vector<int> vDists(Distances[i],Distances[i]+N);
		vector<int> vDists(Distances[i].begin(), Distances[i].end());
		sort(vDists.begin(), vDists.end());

        // 获得中值
        int median = vDists[0.5*(N-1)];
        
        // 寻找最小的中值
        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        
        // 最好的描述子，该描述子相对于其他描述子有最小的距离中值
        // 简化来讲，中值代表了这个描述子到其它描述子的平均距离
        // 最好的描述子就是和其它描述子的平均距离最小
        mDescriptor = vDescriptors[BestIdx].clone();       
    }
}
```
* cv::Mat GetDescriptor();
    * return mDescriptor.clone();//最好的描述子
```c
cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();//最好的描述子
}
```
* void UpdateNormalAndDepth();//更新平均观测方向(看到此点的关键帧们)以及观测距离范围
```c
void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    cv::Mat Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;

        observations=mObservations; // 获得观测到该3d点的所有关键帧
        pRefKF=mpRefKF;             // 观测到该点的参考关键帧
        Pos = mWorldPos.clone();    // 3d点在世界坐标系中的位置
    }

    if(observations.empty())
        return;

    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
    int n=0;
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        cv::Mat Owi = pKF->GetCameraCenter();
        cv::Mat normali = mWorldPos - Owi;
        normal = normal + normali/cv::norm(normali); // 对所有关键帧对该点的观测方向归一化为单位向量进行求和
        n++;
    } 

    cv::Mat PC = Pos - pRefKF->GetCameraCenter(); // 参考关键帧相机指向3D点的向量（在世界坐标系下的表示）
    const float dist = cv::norm(PC); // 该点到参考关键帧相机的距离
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels; // 金字塔层数

    {
        unique_lock<mutex> lock3(mMutexPos);
        // 另见PredictScale函数前的注释
        mfMaxDistance = dist*levelScaleFactor;                           // 观测到该点的距离下限
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1]; // 观测到该点的距离上限
        mNormalVector = normal/n;                                        // 获得平均的观测方向
    }
}
```
* float GetMinDistanceInvariance();
```c
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
```
* float GetMaxDistanceInvariance();
```c
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
```
* int PredictScale(const float &currentDist, KeyFrame*pKF);//给定距离预测尺度
    * currentDist  获取的空间距离
    * return 该点的尺度

```c
//              ____
// Nearer      /____\     level:n-1 --> dmin
//            /______\                       d/dmin = 1.2^(n-1-m)
//           /________\   level:m   --> d
//          /__________\                     dmax/d = 1.2^m
// Farther /____________\ level:0   --> dmax
//
//           log(dmax/d)
// m = ceil(------------)返回大于或者等于指定表达式的最小整数
//            log(1.2)
int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        // mfMaxDistance = ref_dist*levelScaleFactor为参考帧考虑上尺度后的距离
        // ratio = mfMaxDistance/currentDist = ref_dist/cur_dist
        ratio = mfMaxDistance/currentDist;
    }

    // 同时取log线性化
    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}
```
* int PredictScale(const float &currentDist, Frame* pF);//普通帧
#### 公有成员变量
* long unsigned int mnId; ///< Global ID for MapPoint
* static long unsigned int nNextId;
* const long int mnFirstKFid; ///< 创建该MapPoint的关键帧ID
* const long int mnFirstFrame; ///< 创建该MapPoint的帧ID（即每一关键帧有一个帧ID）
* int nObs;
* Variables used by the tracking
    * float mTrackProjX;
    * float mTrackProjY;
    * float mTrackProjXR;
    * int mnTrackScaleLevel;
    * float mTrackViewCos;
    * bool mbTrackInView;
    * long unsigned int mnTrackReferenceForFrame;
    * long unsigned int mnLastFrameSeen;
* Variables used by local mapping
    * long unsigned int mnBALocalForKF;
    * long unsigned int mnFuseCandidateForKF;
* Variables used by loop closing
    * long unsigned int mnLoopPointForKF;
    * long unsigned int mnCorrectedByKF;
    * long unsigned int mnCorrectedReference;
    * cv::Mat mPosGBA;
    * long unsigned int mnBAGlobalForKF;
    * static std::mutex mGlobalMutex;
#### 私有函数

#### 私有变量
* cv::Mat mWorldPos; ///< MapPoint在世界坐标系下的坐标
* std::map<KeyFrame*,size_t> mObservations; ///< 观测到该MapPoint的KF和该MapPoint在KF中的索引
* cv::Mat mNormalVector;// 该MapPoint平均观测方向
* cv::Mat mDescriptor; ///< 通过 ComputeDistinctiveDescriptors() 得到的最优描述子
* KeyFrame* mpRefKF;// Reference KeyFrame
* int mnVisible;// Tracking counters
* int mnFound;
* bool mbBad;// Bad flag (we do not currently erase MapPoint from memory)
* MapPoint* mpReplaced;
* float mfMinDistance;// Scale invariance distances
* float mfMaxDistance;
* Map* mpMap;
* std::mutex mMutexPos;
* std::mutex mMutexFeatures;

### MapPoint源码分析
#### MapPoint.h

```c
#ifndef MAPPOINT_H
#define MAPPOINT_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"

#include<opencv2/core/core.hpp>
#include<mutex>

namespace ORB_SLAM2
{

class KeyFrame;///前向声明
class Map;///前向声明
class Frame;///前向声明

/**
 * @brief MapPoint是一个地图点
 */
class MapPoint
{
public:
    MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
    MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF);

    void SetWorldPos(const cv::Mat &Pos);
    cv::Mat GetWorldPos();

    cv::Mat GetNormal();
    KeyFrame* GetReferenceKeyFrame();

    std::map<KeyFrame*,size_t> GetObservations();
    int Observations();

    void AddObservation(KeyFrame* pKF,size_t idx);
    void EraseObservation(KeyFrame* pKF);

    int GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapPoint* pMP);
    MapPoint* GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, KeyFrame*pKF);
    int PredictScale(const float &currentDist, Frame* pF);

public:
    long unsigned int mnId; ///< Global ID for MapPoint
    static long unsigned int nNextId;
    const long int mnFirstKFid; ///< 创建该MapPoint的关键帧ID
    const long int mnFirstFrame; ///< 创建该MapPoint的帧ID（即每一关键帧有一个帧ID）
    int nObs;

    // Variables used by the tracking
    float mTrackProjX;
    float mTrackProjY;
    float mTrackProjXR;
    int mnTrackScaleLevel;
    float mTrackViewCos;
    // TrackLocalMap - SearchByProjection中决定是否对该点进行投影的变量
    // mbTrackInView==false的点有几种：
    // a 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
    // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影
    // c 不在当前相机视野中的点（即未通过isInFrustum判断）
    bool mbTrackInView;
    // TrackLocalMap - UpdateLocalPoints中防止将MapPoints重复添加至mvpLocalMapPoints的标记
    long unsigned int mnTrackReferenceForFrame;
    // TrackLocalMap - SearchLocalPoints中决定是否进行isInFrustum判断的变量
    // mnLastFrameSeen==mCurrentFrame.mnId的点有几种：
    // a 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
    // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影
    long unsigned int mnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;


    static std::mutex mGlobalMutex;

protected:

    // Position in absolute coordinates
    cv::Mat mWorldPos; ///< MapPoint在世界坐标系下的坐标

    // Keyframes observing the point and associated index in keyframe
    std::map<KeyFrame*,size_t> mObservations; ///< 观测到该MapPoint的KF和该MapPoint在KF中的索引

    // Mean viewing direction
    // 该MapPoint平均观测方向
    cv::Mat mNormalVector;

    // Best descriptor to fast matching
    // 每个3D点也有一个descriptor
    // 如果MapPoint与很多帧图像特征点对应（由keyframe来构造时），那么距离其它描述子的平均距离最小的描述子是最佳描述子
    // MapPoint只与一帧的图像特征点对应（由frame来构造时），那么这个特征点的描述子就是该3D点的描述子
    cv::Mat mDescriptor; ///< 通过 ComputeDistinctiveDescriptors() 得到的最优描述子

    // Reference KeyFrame
    KeyFrame* mpRefKF;

    // Tracking counters
    int mnVisible;
    int mnFound;

    // Bad flag (we do not currently erase MapPoint from memory)
    bool mbBad;
    MapPoint* mpReplaced;

    // Scale invariance distances
    float mfMinDistance;
    float mfMaxDistance;

    Map* mpMap;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
```

## ORBmatcher

### ORBmatcher方法与函数接口

#### 公有成员函数
* ORBmatcher(float nnratio=0.6, bool checkOri=true);//构造函数
    * nnratio: ratio of the best and the second score
    * checkOri: check orientation
* static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);//*静态方法*比较a，b之间的 Hamming距离
```c
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}
```
* int SearchByProjection(Frame &F, const std::vector<MapPoint*> &vpMapPoints, const float th=3);//**该帧的特征点和Local MapPoint的投影**
    * @param  F           当前帧
    * @param  vpMapPoints Local MapPoints
    * @param  th          阈值
    * @return             成功匹配的数量 
    * 每个MapPoints都有一个自己局部地图里最佳的描述子（在对应关键帧中），我们就是那当前帧匹配到的描述子群和这个最佳的比较，在这些个\n
    * 描述子群里选出最好的（当然符合一系列条件），这样这个地图点就成功匹配到图像中了。
    * 然后重复遍历局部地图里所有的点，都去在这张图里找。
    * 最后看看能找到几个符合条件的，这个匹配数量是决定该帧能不能晋升为关键帧的必要条件之一。
```c
int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th)
{
    int nmatches=0;

    const bool bFactor = th!=1.0;

    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
    {
        MapPoint* pMP = vpMapPoints[iMP];

        // 判断该点是否要投影
        // mbTrackInView==false的点有几种：
        // a 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
        // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影
        // c 不在当前相机视野中的点（即未通过isInFrustum判断）
        if(!pMP->mbTrackInView)
            continue;

        if(pMP->isBad())
            continue;
            
        // 通过距离预测的金字塔层数，该层数相对于当前的帧
        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction
        // 搜索窗口的大小取决于视角, 若当前视角和平均视角夹角接近0度时, r取一个较小的值
        float r = RadiusByViewingCos(pMP->mTrackViewCos);//接近0就取2.5，否则就取4
        
        // 如果需要进行更粗糙的搜索，则增大范围
        if(bFactor)//只是一个参数，为了调节搜索范围
            r*=th;

        // 通过投影点(投影到当前帧,见isInFrustum())以及搜索窗口和预测的尺度进行搜索, 找出附近的兴趣点
        const vector<size_t> vIndices =
                //找到在 以（pMP->mTrackProjX,pMP->mTrackProjY）为中心,边长为2r的方形内且在层数[minLevel, maxLevel]的特征点
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);
        if(vIndices.empty())
            continue;

        const cv::Mat MPdescriptor = pMP->GetDescriptor(); ///< 通过 ComputeDistinctiveDescriptors() 得到的最优描述子

        int bestDist=256;
        int bestLevel= -1;
        int bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        // Get best and second matches with near keypoints
        //通过遍历投影点附近区域所有的特征点序列号
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            // 如果Frame中的该兴趣点已经有对应的MapPoint了,则退出该次循环
            if(F.mvpMapPoints[idx])
                if(F.mvpMapPoints[idx]->Observations()>0)
                    continue;

            if(F.mvuRight[idx]>0)
            {
                const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);//fabs：求浮点型绝对值  F.mvuRight[idx]：右横坐标
                if(er>r*F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

            const cv::Mat &d = F.mDescriptors.row(idx);

            const int dist = DescriptorDistance(MPdescriptor,d);
            
            // 根据描述子寻找描述子距离最小和次小的特征点
            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx=idx;
            }
            else if(dist<bestDist2)
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2=dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if(bestDist<=TH_HIGH)
        {
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                continue;

            F.mvpMapPoints[bestIdx]=pMP; // 为Frame中的兴趣点增加对应的MapPoint
            nmatches++;
        }
    }

    return nmatches;
}
```

* int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);//**上一帧与当前帧匹配**
    * @param  bMono        是否为单目 
```c


```
* int SearchByProjection(Frame &CurrentFrame, KeyFrame* pKF, const std::set<MapPoint*> &sAlreadyFound, const float th, const int ORBdist);//**当前帧与关键帧匹配**  用于relocalisation(Tracking)
* int SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, std::vector<MapPoint*> &vpMatched, int th);//**Project MapPoints using a Similarity Transformation and search matches**  (Loop Closing)
KeyFrame中包含了MapPoints，对这些MapPoints进行tracking；由于每一个MapPoint对应有描述子，因此可以通过描述子距离进行跟踪；为了加速匹配过程，将关键帧和当前帧的描述子划分到特定层的nodes中；对属于同一node的描述子计算距离进行匹配；通过距离阈值、比例阈值和角度投票进行剔除误匹配。
* int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);//通过词包，对关键帧的特征点进行跟踪
    * @param  vpMapPointMatches F中MapPoints对应的匹配，NULL表示未匹配
    * @return             成功匹配的数量 
```c
/**
 * @brief 通过词包，对关键帧的特征点进行跟踪
 * 
 * 通过bow对pKF和F中的特征点进行快速匹配（不属于同一node的特征点直接跳过匹配） \n
 * 对属于同一node的特征点通过描述子距离进行匹配 \n
 * 通过关键帧找到的地图点，关键帧的特征点又和frame一一匹配
 * 根据匹配，用pKF中特征点对应的MapPoint更新F中特征点对应的MapPoints \n
 * 每个特征点都对应一个MapPoint，因此pKF中每个特征点的MapPoint也就是F中对应点的MapPoint \n
 * 通过距离阈值、比例阈值和角度投票进行剔除误匹配
 * @param  pKF               KeyFrame
 * @param  F                 Current Frame
 * @param  vpMapPointMatches F中MapPoints对应的匹配，NULL表示未匹配
 * @return                   成功匹配的数量
 */
int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));//F中MapPoints对应的匹配，NULL表示未匹配,初始化

    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;//DBoW2::FeatureVector std::map<NodeId, std::vector<unsigned int> >//不同层

    int nmatches=0;

    vector<int> rotHist[HISTO_LENGTH];//直方图的长度
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = HISTO_LENGTH/360.0f;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    // 将属于同一节点(特定层)的ORB特征进行匹配
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();//关键帧的第一层
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();//F的第一层
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    while(KFit != KFend && Fit != Fend)
    {
        if(KFit->first == Fit->first) //步骤1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)//同一层的特征点
        {
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF = Fit->second;

            // 步骤2：遍历KF中属于该层的特征点
            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)//这个size是每一层的size,而不是总的特征点数量
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];

                ///补充一句，目前只有keyframe才有资格这样做，根据vpMapPointsKF[],找出对应的地图点
                MapPoint* pMP = vpMapPointsKF[realIdxKF]; // 取出KF中该特征对应的MapPoint，为了以后更新F

                if(!pMP)
                    continue;

                if(pMP->isBad())
                    continue;

                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF); // 取出KF中该特征对应的描述子

                int bestDist1=256; // 最好的距离（最小距离）
                int bestIdxF =-1 ;
                int bestDist2=256; // 倒数第二好距离（倒数第二小距离）

                // 步骤3：遍历F中属于该node的特征点，找到最佳匹配点
                for(size_t iF=0; iF<vIndicesF.size(); iF++)
                {
                    const unsigned int realIdxF = vIndicesF[iF];

                    if(vpMapPointMatches[realIdxF])// 表明这个点已经被匹配过了，不再匹配，加快速度
                        continue;

                    const cv::Mat &dF = F.mDescriptors.row(realIdxF); // 取出F中该特征对应的描述子

                    const int dist =  DescriptorDistance(dKF,dF); // 求描述子的距离

                    if(dist<bestDist1)// dist < bestDist1 < bestDist2，更新bestDist1 bestDist2
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdxF=realIdxF;
                    }
                    else if(dist<bestDist2)// bestDist1 < dist < bestDist2，更新bestDist2
                    {
                        bestDist2=dist;
                    }
                }

                // 步骤4：根据阈值 和 角度投票剔除误匹配
                if(bestDist1<=TH_LOW) // 匹配距离（误差）小于阈值
                {
                    // trick!
                    /// 最佳匹配比次佳匹配明显要好，那么最佳匹配才真正靠谱
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        // 步骤5：更新F中特征点的MapPoint
                        vpMapPointMatches[bestIdxF]=pMP;

                        const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

                        if(mbCheckOrientation)
                        {
                            // trick!
                            // angle：每个特征点在提取描述子时的旋转主方向角度，如果图像旋转了，这个角度将发生改变
                            // 所有的特征点的角度变化应该是一致的，通过直方图统计得到最准确的角度变化值
                            float rot = kp.angle-F.mvKeys[bestIdxF].angle;// 该特征点的角度变化值
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);// 将rot分配到箱子  const float factor = HISTO_LENGTH/360.0f; 分成360份，按度数大小排列
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);//每一层的特征点KF与F之间最佳匹配点（一对），检查他们的描述子角度之差，做个统计。
                        }
                        nmatches++;//这个时候F中好的特征点已经找到mappoint了。依次计数。
                    }
                }

            }

            KFit++;//接着下一层
            Fit++;
        }
        else if(KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }

    // 根据方向剔除误匹配的点
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        // 计算rotHist中最大的三个的index
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);//排名最高的3个角度

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            // 如果特征点的旋转角度变化量属于这三个组，则保留
            if(i==ind1 || i==ind2 || i==ind3)
                continue;

            // 将除了ind1 ind2 ind3以外的匹配点去掉
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}
```
* int SearchByBoW(KeyFrame *pKF1, KeyFrame* pKF2, std::vector<MapPoint*> &vpMatches12);//主要用于闭环检测时两个关键帧匹配。程序方法与上一个类似
```c
/**
 * @brief 通过词包，对关键帧的特征点进行跟踪，该函数用于闭环检测时两个关键帧间的特征点匹配
 * 
 * 通过bow对pKF1和pKF2中的特征点进行快速匹配（不属于同一node的特征点直接跳过匹配） \n
 * 对属于同一node的特征点通过描述子距离进行匹配 \n
 * 根据匹配，更新vpMatches12 \n
 * 通过距离阈值、比例阈值和角度投票进行剔除误匹配
 * @param  pKF1               KeyFrame1
 * @param  pKF2               KeyFrame2
 * @param  vpMatches12        pKF2中与pKF1匹配的MapPoint，null表示没有匹配
 * @return                    成功匹配的数量
 */
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    // 详细注释可参见：SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)

    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = HISTO_LENGTH/360.0f;

    int nmatches = 0;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
        if(f1it->first == f2it->first)//步骤1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
        {
            // 步骤2：遍历KF中属于该node的特征点
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint* pMP1 = vpMapPoints1[idx1];
                if(!pMP1)
                    continue;
                if(pMP1->isBad())
                    continue;

                const cv::Mat &d1 = Descriptors1.row(idx1);

                int bestDist1=256;
                int bestIdx2 =-1 ;
                int bestDist2=256;

                // 步骤3：遍历F中属于该node的特征点，找到了最佳匹配点
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    if(vbMatched2[idx2] || !pMP2)//
                        continue;

                    if(pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);

                    int dist = DescriptorDistance(d1,d2);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                // 步骤4：根据阈值 和 角度投票剔除误匹配
                // 详见SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)函数步骤4
                if(bestDist1<TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation)
                        {
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
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
                vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
} 
```
* int SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,std::vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo);//三角化，对极约束，双目。//利用基本矩阵F12，在两个关键帧之间未匹配的特征点中产生新的3d点
```c
/**
 * @brief 利用基本矩阵F12，在两个关键帧之间未匹配的特征点中产生新的3d点
 * 
 * @param pKF1          关键帧1
 * @param pKF2          关键帧2
 * @param F12           基础矩阵
 * @param vMatchedPairs 存储匹配特征点对，特征点用其在关键帧中的索引表示
 * @param bOnlyStereo   在双目和rgbd情况下，要求特征点在右图存在匹配
 * @return              成功匹配的数量
 */
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo)
{
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;//std::map<NodeId, std::vector<unsigned int> >
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    // Compute epipole in second image
    // 计算KF1的相机中心在KF2图像平面的坐标，即极点坐标
    cv::Mat Cw = pKF1->GetCameraCenter(); // twc1 相机1光心O1
    cv::Mat R2w = pKF2->GetRotation();    // Rc2w
    cv::Mat t2w = pKF2->GetTranslation(); // tc2w
    cv::Mat C2 = R2w*Cw+t2w; // tc2c1 相机2光心O2
    const float invz = 1.0f/C2.at<float>(2);
    // 步骤0：得到KF1的相机光心在KF2中的坐标（极点坐标）像素坐标
    const float ex =pKF2->fx*C2.at<float>(0)*invz+pKF2->cx;//e2.x
    const float ey =pKF2->fy*C2.at<float>(1)*invz+pKF2->cy;//e2.y

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false);//记录pKF2匹配情况
    vector<int> vMatches12(pKF1->N,-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = HISTO_LENGTH/360.0f;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    // 将属于同一节点(特定层)的ORB特征进行匹配
    // FeatureVector的数据结构类似于：{(node1,feature_vector1) (node2,feature_vector2)...}
    // f1it->first对应node编号，f1it->second对应属于该node的所有特特征点编号
    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    // 步骤1：遍历pKF1和pKF2中的node节点
    while(f1it!=f1end && f2it!=f2end)
    {
        // 如果f1it和f2it属于同一个node节点
        if(f1it->first == f2it->first)
        {
            // 步骤2：遍历该node节点下(f1it->first)的所有特征点
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                // 获取pKF1中属于该node节点的所有特征点索引
                const size_t idx1 = f1it->second[i1];
                
                // 步骤2.1：通过特征点索引idx1在pKF1中取出对应的MapPoint
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);
                
                // If there is already a MapPoint skip
                /// 由于寻找的是未匹配的特征点，所以pMP1应该为NULL
                if(pMP1)
                    continue;

                /// 如果mvuRight中的值大于0，表示是双目，且该特征点有深度值
                const bool bStereo1 = pKF1->mvuRight[idx1]>=0;

                if(bOnlyStereo)
                    if(!bStereo1)
                        continue;
                
                // 步骤2.2：通过特征点索引idx1在pKF1中取出对应的特征点
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
                
                // 步骤2.3：通过特征点索引idx1在pKF1中取出对应的特征点的描述子
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
                
                int bestDist = TH_LOW;
                int bestIdx2 = -1;
                
                // 步骤3：遍历该node节点下(f2it->first)的所有特征点
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    // 获取pKF2中属于该node节点的所有特征点索引
                    size_t idx2 = f2it->second[i2];
                    
                    // 步骤3.1：通过特征点索引idx2在pKF2中取出对应的MapPoint
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);
                    
                    // If we have already matched or there is a MapPoint skip
                    // 如果pKF2当前特征点索引idx2已经被匹配过或者对应的3d点非空
                    // 那么这个索引idx2就不能被考虑
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    const bool bStereo2 = pKF2->mvuRight[idx2]>=0;

                    if(bOnlyStereo)
                        if(!bStereo2)
                            continue;
                    
                    // 步骤3.2：通过特征点索引idx2在pKF2中取出对应的特征点的描述子
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
                    
                    // 计算idx1与idx2在两个关键帧中对应特征点的描述子距离
                    const int dist = DescriptorDistance(d1,d2);
                    
                    if(dist>TH_LOW || dist>bestDist)
                        continue;

                    // 步骤3.3：通过特征点索引idx2在pKF2中取出对应的特征点
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

                    if(!bStereo1 && !bStereo2)
                    {
                        const float distex = ex-kp2.pt.x;
                        const float distey = ey-kp2.pt.y;
                        // 该特征点距离极点太近，表明kp2对应的MapPoint距离pKF1相机太近(对极几何)
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                            continue;
                    }

                    // 步骤4：满足对极约束吗？
                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))//
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }

                // 步骤1、2、3、4总结下来就是：将左图像的每个特征点与右图像同一node节点的所有特征点（未匹配mappoint）
                // 依次检测，判断是否满足对极几何约束，满足约束就是匹配的特征点
                
                // 详见SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)函数步骤4
                if(bestIdx2>=0)
                {
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
                    vMatches12[idx1]=bestIdx2;
                    vbMatched2[bestIdx2]=true;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
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
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }

    return nmatches;
}
```
* int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize=10);// Matching for the Map 初始化 (只用在单目初始化)
```c

```

* int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th);//相似变换 Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
```c
// 通过Sim3变换，确定pKF1的特征点在pKF2中的大致区域
// 同理，确定pKF2的特征点在pKF1中的大致区域
// 在该区域内通过描述子进行匹配捕获pKF1和pKF2之前漏匹配的特征点，更新vpMatches12（之前使用SearchByBoW进行特征点匹配时会有漏匹配）
///1. 2个关键帧都有自己各自的地图点。2. 两者如何快速匹配？ 3. 互相找到各自特征点在对方中的大致区域  4.各自找到最佳 5. 评判两者是不是都选择对方
int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    // 步骤1：变量初始化
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    // 从world到camera的变换
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);// 用于记录该特征点是否被处理过
    vector<bool> vbAlreadyMatched2(N2,false);// 用于记录该特征点是否在pKF1中有匹配

    // 步骤2：用vpMatches12更新vbAlreadyMatched1和vbAlreadyMatched2
    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;// 该特征点已经判断过
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;// 该特征点在pKF1中有匹配
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    /// Transform from KF1 to KF2 and search
    // 步骤3.1：通过Sim变换，确定pKF1的特征点在pKF2中的大致区域，
    //         在该区域内通过描述子进行匹配捕获pKF1和pKF2之前漏匹配的特征点，更新vpMatches12
    //         （之前使用SearchByBoW进行特征点匹配时会有漏匹配）
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];///①KF1的该地图点存在

        if(!pMP || vbAlreadyMatched1[i1])/// ②该特征点已经有匹配点了，直接跳过
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();///③找到该地图点的世界坐标
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;/// ④把pKF1系下的MapPoint从world坐标系变换到camera1坐标系
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;// 再通过Sim3将该MapPoint从camera1变换到camera2坐标系

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0)
            continue;

        /// ⑤投影到camera2图像平面
        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;
        
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        // 预测该MapPoint对应的特征点在图像金字塔哪一层
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        // 计算特征点搜索半径
        /// ⑥这里就会产生一片区域（多个特征点群）
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        // 取出该区域内的所有特征点
        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();///还是第一帧那个地图点，拿出它的最佳描述子和⑥（特征点群）PK

        int bestDist = INT_MAX;
        int bestIdx = -1;
        // 遍历搜索区域内的所有特征点，与pMP进行描述子匹配
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;///⑦找到pMP在KF2里最佳描述子
        }
    }

    /// Transform from KF2 to KF1 and search
    // 步骤3.2：通过Sim变换，确定pKF2的特征点在pKF1中的大致区域，
    //         在该区域内通过描述子进行匹配捕获pKF1和pKF2之前漏匹配的特征点，更新vpMatches12
    //         （之前使用SearchByBoW进行特征点匹配时会有漏匹配）
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }
/// 到这里，大家都找到各自在对方中最佳匹配点了。确实很快！
/// 不过要验证一下，确实啊，比如5对5，大家各自找找自己最喜欢的人，
/// 最后评委看看，是不是你俩选的一样啊！1号选的5号，那么5号也要选1号，那才算成功匹配，哈哈哈～～～
    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}
```
* int Fuse(KeyFrame* pKF, const vector<MapPoint *> &vpMapPoints, const float th=3.0);//地图点投影到当前关键帧里，且寻找冗余点
* int Fuse(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint);//Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
#### 公有成员变量

* static const int TH_LOW;//50
* static const int TH_HIGH;//100
* static const int HISTO_LENGTH;//30

#### 私有函数
* bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF);//检查极线约束
```c
bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    /// 求出kp1在pKF2上对应的极线 x1'*F12*x2=0, 因为x2都在（只在）这个方程上，所以极线l2就是有这些点组成。
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    // 计算kp2特征点到极线的距离：（接近0就是满足极线约束，kp1和kp2）
    /// 极线l：ax + by + c = 0
    // (u,v)到l的距离为： |au+bv+c| / sqrt(a^2+b^2)

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    // 尺度越大，范围应该越大。
    // 金字塔最底层一个像素就占一个像素，在倒数第二层，一个像素等于最底层1.2个像素（假设金字塔尺度为1.2）
    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
}
```
* float RadiusByViewingCos(const float &viewCos);
```c
float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}
```
* void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
#### 私有变量
* float mfNNratio;
* bool mbCheckOrientation;
好了介绍这么多，真的好累。。。。终于可以撸起袖子。。。。
## 主要线程
### system入口
#### 主要函数
系统初始化入口
* **默认构造函数 System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
               const bool bUseViewer):mSensor(sensor),mbReset(false),mbActivateLocalizationMode(false),
               mbDeactivateLocalizationMode(false)**  
    当调用这个system时，初始化：  
    * 加载词典包Load ORB Vocabulary
    * Create关键帧数据库 KeyFrame Database
    * Create 地图 the Map
    * Create 绘图 Drawers. These are used by the Viewer
    * 初始化 跟踪 the Tracking thread(// The Tracking thread "lives" in the main execution thread that creates the System object.)
    ```c
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::STEREO,true);
    ```
    * 初始化 局部地图线程 the Local Mapping thread and launch
    * 初始化 闭环检测线程 the Loop Closing thread and launch
    * 初始化 Viewer线程 the Viewer thread and launch
```c
System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
               const bool bUseViewer):mSensor(sensor),mbReset(false),mbActivateLocalizationMode(false),
               mbDeactivateLocalizationMode(false)
{
    // Output welcome message
    cout << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";

    if(mSensor==MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }


    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;
    
    mpVocabulary = new ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;
/*
    mpVocabulary = new ORBVocabulary();
    bool bVocLoad = false; // chose loading method based on file extension
    if (has_suffix(strVocFile, ".txt"))
	  bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
	else if(has_suffix(strVocFile, ".bin"))
	  bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);
	else
	  bVocLoad = false;
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Failed to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;
*/
    //Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    //Create the Map
    mpMap = new Map();

    //Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    //初始化 the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

    //初始化 the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);
    mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run,mpLocalMapper);

    //初始化 the Loop Closing thread and launch
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);
    mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);

    //初始化 the Viewer thread and launch
    mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
    if(bUseViewer)
        mptViewer = new thread(&Viewer::Run, mpViewer);

    mpTracker->SetViewer(mpViewer);

    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}
```
* cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
    * Returns the camera pose

### Tracking 线程
* 初始化步骤：  
    1. 加载相机参数文件
    1. 加载ORB配置参数
        1. 每一帧提取的特征点数 1000 `nFeatures`
        1. 图像建立金字塔时的变化尺度 1.2 `fScaleFactor`
        1. 尺度金字塔的层数 8 `nLevels`
        1. 提取fast特征点的默认阈值 20，`fIniThFAST`
        1. 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8 `fIniThFAST`
    1. tracking过程都会用到mpORBextractorLeft作为特征点提取器
        * mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
            * 如果是双目，tracking过程中还会用用到mpORBextractorRight作为右目特征点提取器
            * 在单目初始化的时候，会用mpIniORBextractor来作为特征点提取器//是上面的2倍特征点数量
    1. 如果是双目/RBGD还要判断一个3D点远/近的阈值 `mThDepth`
        * `mThDepth = mbf*(float)fSettings["ThDepth"]/fx`;
        * 如果`sensor==System::RGBD`,`mDepthMapFactor`深度相机disparity转化为depth时的因子
* 抓取图片函数
    1. 将RGB或RGBA图像转为灰度图像
        * 主要是对不同类型图像的处理`cvtColor`
            * `if(mImGray.channels()==3)`
            * `else if(mImGray.channels()==4)`
    2. 构造Frame
        1. 双目//使用`Frame`类里双目构造函数
            * `mCurrentFrame` = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
        1. RGBD//使用`Frame`类里RGBD构造函数
            * mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
        1. 单目
            1. 判断初始化状态`if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)`
                * mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);//2倍的数量
                * mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    3. **跟踪（下面重点讲）**
    4. 返回当前帧输出世界坐标系到该帧相机坐标系的变换矩阵
        * return mCurrentFrame.mTcw.clone();
* 跟踪 void Tracking::Track()  
track包含两部分：估计运动、跟踪局部地图      
    *  Get Map Mutex -> Map cannot be changed
        * `unique_lock<mutex> lock(mpMap->mMutexMapUpdate);`
    *  判断状态`mState==NOT_INITIALIZED`  
    如果图像复位过、或者第一次运行，则为`NO_IMAGE_YET`状态，如果是`NO_IMAGE_YET`，就设置为`NOT_INITIALIZED`
        1. **初始化**
            1. 双目初始化`StereoInitialization()`
                1. 检测特征点数量是否大于500
                1. 设定初始位姿eye(4,4),并将当前帧构造为初始关键帧
                    * KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);
                1. 在地图中添加该初始关键帧
                    * mpMap->AddKeyFrame(pKFini);
                1. 为每个特征点构造MapPoint
                    1. 通过反投影得到该特征点的3D坐标//cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    1. 将3D点构造为MapPoint//MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                    1. 为该MapPoint添加属性
                        1. 观测到该MapPoint的关键帧//pNewMP->AddObservation(pKFini,i);
                        1. 从众多观测到该MapPoint的特征点中挑选区分读最高的描述子//pNewMP->ComputeDistinctiveDescriptors();
                        1. 该MapPoint的平均观测方向和深度范围//pNewMP->UpdateNormalAndDepth();
                        1. 在地图中添加该MapPoint//mpMap->AddMapPoint(pNewMP);
                        1. 表示该KeyFrame的哪个特征点可以观测到哪个3D点//pKFini->AddMapPoint(pNewMP,i);
                        1. 将该MapPoint添加到当前帧的mvpMapPoints中//mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    1. 在局部地图中添加该初始关键帧//mpLocalMapper->InsertKeyFrame(pKFini);
                        1. 将当前帧赋值给上一帧//mLastFrame = Frame(mCurrentFrame);
                        1. id;//mnLastKeyFrameId=mCurrentFrame.mnId;
                        1. 关键帧//mpLastKeyFrame = pKFini;
                        1. 把当前帧添加进局部关键帧//mvpLocalKeyFrames.push_back(pKFini);
                        1. 局部地图点//mvpLocalMapPoints=mpMap->GetAllMapPoints();
                        1. 参考帧设为该帧//mpReferenceKF = pKFini;
                        1. 当前帧的参考帧设为自己//mCurrentFrame.mpReferenceKF = pKFini;
                        1. 把当前（最新的）局部MapPoints作为ReferenceMapPoints//mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
                        1. 画图需要关键帧的Ｔ//mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
                        1. mState=OK;
            1. 单目初始化`MonocularInitialization()`
                * 如果单目初始器没有初始化，则需要创建单目初始器。
                    1. 单目初始帧的特征点数必须大于100
                    1. 得到用于初始化的第一帧，初始化需要两帧//mInitialFrame = Frame(mCurrentFrame);
                    1. mLastFrame = Frame(mCurrentFrame);//记录最近的一帧
                    1. 由当前帧构造初始器 sigma:1.0 iterations:200//mpInitializer =  new Initializer(mCurrentFrame,1.0,200);
                * 有了单目初始器，插入第二帧。如果当前帧特征点太少，重新构造初始器
                    * 在mInitialFrame与mCurrentFrame中找匹配的特征点对//int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);
                * 如果初始化的两帧之间的匹配点太少，重新初始化
                * **通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints//if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))**
                    * Initialize函数会得到mvIniP3D，
                    * mvIniP3D是cv::Point3f类型的一个容器，是个存放3D点的临时变量
                * 删除那些无法进行三角化的匹配点
                * 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵//mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
                * 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到该帧的变换矩阵
                * 三角化生成3D点；包装成MapPoints//CreateInitialMapMonocular();
                    1. 先将那两帧生成关键帧`mInitialFrame/mCurrentFrame`
                    1. 将初始关键帧和当前帧的描述子转为BoW//pKFini->ComputeBoW();pKFcur->ComputeBoW();
                    1. 将关键帧（他们）插入到地图，凡是关键帧，都要插入地图//mpMap->AddKeyFrame(pKFini);mpMap->AddKeyFrame(pKFcur);
                    1. 将3D点包装成MapPoints
                        1. 遍历他们匹配的关键点，依次生成空间点worldPos//cv::Mat worldPos(mvIniP3D[i]);
                        1. 用3D点构造MapPoint//MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);
                        1. 添加mappoint//pKFini->AddMapPoint(pMP,i);pKFcur->AddMapPoint(pMP,mvIniMatches[i]);  
                        添加属性：
                            1. 表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到//pMP->AddObservation(pKFini,i);
                            1. 该MapPoint的描述子//pMP->ComputeDistinctiveDescriptors();
                            1. 该MapPoint的平均观测方向和深度范围//pMP->UpdateNormalAndDepth();
                        1. mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
                        1. mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;
                        1. 在地图中添加该MapPoint// mpMap->AddMapPoint(pMP);
                    1. 更新关键帧间的连接关系，在3D点和关键帧之间建立边，每个边有一个权重，边的权重是该关键帧与当前帧公共3D点的个数//pKFini->UpdateConnections();pKFcur->UpdateConnections();
                    1. BA优化
                    1. 将MapPoints的中值深度归一化到1，并归一化两帧之间变换
                        1. 评估关键帧场景深度，q=2表示中值//medianDepth = pKFini->ComputeSceneMedianDepth(2);float invMedianDepth = 1.0f/medianDepth;
                        1. 位姿的z归一//Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
                        1. 把所有的3D点的尺度也归一化到1//pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
                    1. 在局部地图中添加该初始关键帧//mpLocalMapper->InsertKeyFrame(pKFini);mpLocalMapper->InsertKeyFrame(pKFcur);//这部分和SteroInitialization()相似
                        1. mCurrentFrame.SetPose(pKFcur->GetPose());
                        1. 将当前帧赋值给上一帧//mLastFrame = Frame(mCurrentFrame);
                        1. id;//mnLastKeyFrameId=mCurrentFrame.mnId;
                        1. 关键帧//mpLastKeyFrame = pKFcur;
                        1. 把当前帧添加进局部关键帧//mvpLocalKeyFrames.push_back(pKFini);mvpLocalKeyFrames.push_back(pKFcur);
                        1. 局部地图点//mvpLocalMapPoints=mpMap->GetAllMapPoints();
                        1. 参考帧设为该帧//mpReferenceKF = pKFcur;
                        1. 当前帧的参考帧设为自己//mCurrentFrame.mpReferenceKF = pKFini;
                        1. 把当前（最新的）局部MapPoints作为ReferenceMapPoints//mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
                        1. 画图需要关键帧的Ｔ//mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());
                        1. mState=OK;// 初始化成功，至此，初始化过程完成

                
        2. **跟踪相机位姿**  
        2种模式：TrackWithMotionModel()和TrackReferenceKeyFrame()，主要区别就在于匹配的方式不一样  
            1. SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);///mCurrentFrame的姿态是估计出来的。速度模型所用的。这个可以极大缩小匹配范围  
            2. SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);参考帧模型所用的，和常规的差不太多。
            * mbOnlyTracking(false)：（默认）同时跟踪与定位，Local Mapping 被激活！
                * 状态：`OK`//过了初始化部分。
                    + 检查并更新上一帧被替换的MapPoints。`CheckReplacedInLastFrame();`
                    + 跟踪上一帧或者参考帧或者重定位.
                        *  检查 运动模型是空的或刚完成重定位
                            * 是：TrackReferenceKeyFrame();
                                1. ORBmatcher matcher(0.7,true);
                                1. 将当前帧的描述子转化为BoW向量// mCurrentFrame.ComputeBoW();
                                1. 通过特征点的BoW加快当前帧与参考帧之间的特征点匹配//int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
                                    * if(nmatches<15) return false;退出函数
                                1. 将上一帧的位姿作为当前帧的初始位姿
                                    1. mCurrentFrame.mvpMapPoints = vpMapPointMatches;
                                    1. mCurrentFrame.SetPose(mLastFrame.mTcw); // 用上一次的Tcw设置初值，在PoseOptimization可以收敛快一些
                                1. 通过优化3D-2D的重投影误差来获得位姿//Optimizer::PoseOptimization(&mCurrentFrame);
                                1. 剔除优化后的outlier匹配点（MapPoints）
                                1. return nmatchesMap>=10;
                            * 否：TrackWithMotionModel();
                                1. 对于双目或rgbd摄像头，根据深度值为上一关键帧补充生成新的MapPoints（由特征点构成）//UpdateLastFrame();
                                    1. 更新最近一帧的位姿
                                    1. 对于双目或rgbd摄像头，为上一帧临时生成新的MapPoints
                                        1. 得到上一帧有深度值的特征点// `vDepthIdx.push_back(make_pair(z,i));` 
                                        1. 按照深度从小到大排序//sort(vDepthIdx.begin(),vDepthIdx.end());
                                        1. 将距离比较近的点包装成MapPoints（前100个），这些点都不会添加属性，仅仅为了提高双目的跟踪率
                                            1. 不是地图点//if(!pMP)，bCreateNew = false;
                                            1. 没有被记录的地图点（没有属性的点）pMP->Observations()<1
                                            1. 把这些点标记成临时 //mlpTemporalPoints.push_back(pNewMP);
                                1. 根据恒速模型设定当前帧的初始位姿，认为这两帧之间的相对运动和之前两帧间相对运动相同；mVelocity：其实就是2帧之间的变换矩阵，这样通过乘以这样一个变换矩阵“预测”mCurrentFrame的姿态；因为新来的帧还没有匹配就不知道姿态，但要缩小匹配范围就要事先“知道”姿态变化！//mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);
                                1. 通过投影的方式在参考帧中找当前帧特征点的匹配点（难点）//SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR)；//这个函数需要两个帧`frame`,所以需要他们各自的参数`CurrentFrame.mTcw`。所以要用这个函数必须事先给定，哪怕估计出来的`CurrentFrame.mTcw`。
                                    * 如果跟踪的点少，则扩大搜索半径再来一次.第二次失败，直接跳出函数`TrackWithMotionModel()`
                                1. 优化每个特征点所对应3D点的投影误差即可得到位姿//PoseOptimization(&mCurrentFrame);
                                1. 优化位姿后剔除outlier的mvpMapPoints
                                1. return nmatchesMap>=10;
                                    * 如果为false: bOK = TrackReferenceKeyFrame();
                * 状态非`OK`
                    * Relocalization();//BOW搜索，PnP求解位姿
                        1. 计算当前帧特征点的Bow映射
                        1. 找到与当前帧相似的候选关键帧(vector)
                            * `vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);`
                        1. 通过BoW对这些候选关键帧与当前帧进行匹配
                        ！！！开始循环遍历候选帧！！！
                            * SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
                            * if(nmatches<15) continue;
                            * if(nmatches>15) 初始化PnPsolver
                                * `PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);`
                                * `pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);`
                        1. 通过EPnP算法估计姿态
                            1. PnPsolver* pSolver = vpPnPsolvers[i];
                            1. cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);
                                1. If Ransac reachs max. iterations discard keyframe
                                1. 如果相机位姿算出来了，优化// Tcw.copyTo(mCurrentFrame.mTcw);
                                1. 给mCurrentFrame挑出匹配的地图点，好进行BA优化
                        1. 通过PoseOptimization对姿态进行优化求解//int nGood = Optimizer::PoseOptimization(&mCurrentFrame)；//返回内点数量
                        1. 如果内点较少（<50），则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
                            1. `int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);`//增加点数
                            * if(nadditional+nGood>=50)//如果这次满足优化的点数了
                                *  nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                                *  如果这样操作后还是少，比如在（30-50）
                                    * 该帧的发现点数变成操作之后（后添加的）的所有点//sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                    * 再找一次，不过in a narrower window。`nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);`
                                    * 如果返回值大于50，BA优化，标记下mvbOutlier。....
                        1. 如果内点较少（>50）
                            1. bMatch = true;//标志位！结束循环标志位！
                            1. break; //退出循环，这么多候选帧里就你了！重定位成功！！！
                        ！！！退出循环！！！
                        1. 如果循环完后发现还是不行bMatch还是false.//return false;
                        1. 如果循环完后，bMatch变色了！
                            * mnLastRelocFrameId = mCurrentFrame.mnId;//当前帧的id给他
                            * return true;
            * mbOnlyTracking(true)://Localization Mode: Local Mapping is deactivated，只进行跟踪tracking，局部地图不工作，不插入关键帧。
                * 状态：`LOST`
                    * Relocalization();（见上展开）
                        1. 计算当前帧特征点的Bow映射
                        1. 找到与当前帧相似的候选关键帧
                        1. 通过BoW进行匹配
                        1. 通过EPnP算法估计姿态
                        1. 通过PoseOptimization对姿态进行优化求解
                        1. 如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
                * 状态：非`LOST`
                    * `mbVO==0`： mbVO为0则表明此帧匹配了很多的3D map点，非常好
                        * 判断`!mVelocity.empty()`
                            * 有运动模型：TrackWithMotionModel()
                            * 无运动模型：TrackReferenceKeyFrame()
                    * `mbVO==1`：mbVO为1，则表明此帧匹配了很少的3D map点，少于10个，要跪的节奏，既做跟踪又做定位
                        1. 判断`!mVelocity.empty()`
                            * 有运动模型：TrackWithMotionModel()
                        1. 没有，bOKReloc = Relocalization();//重定位
                        1. 如果重定位没有成功，但是如果跟踪成功
                            * 再次观察`mbVO`
                                * `mbVO==1`: 更新当前帧的MapPoints被观测程度
                        1. 如果重定位成功：**只要重定位成功整个跟踪过程正常进行（定位与跟踪，更相信重定位）**
                            * mbVO = false
                            * bOK = bOKReloc || bOKMM;  
        3. **局部地图跟踪**： 当前帧、当前帧的MapPoints、当前关键帧与其它关键帧共视关系

            在帧间匹配得到初始的姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
            local map:当前帧、当前帧的MapPoints、当前关键帧与其它关键帧共视关系
            在步骤2.1中主要是两两跟踪（恒速模型跟踪上一帧、跟踪参考帧），这里搜索局部关键帧后搜集所有局部MapPoints,然后将局部MapPoints和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化。
            
            * mCurrentFrame.mpReferenceKF = mpReferenceKF;//将最新的关键帧作为reference frame
            * 判断`!mbOnlyTracking`为真//判断是不是同时跟踪与定位
                * `bOK`: TrackLocalMap();
                    1. UpdateLocalMap()//更新局部关键帧mvpLocalKeyFrames和局部地图点mvpLocalMapPoints
                        +  mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
                        + **UpdateLocalKeyFrames();** 
                            +  遍历当前帧的MapPoints，记录所有能观测到当前帧MapPoints的关键帧
                            +  更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有三个策略
                                +  先清空局部关键帧
                                +  能观测到当前帧MapPoints的关键帧作为局部关键帧
                                +  记录最强关键帧
                                +  与策略1得到的局部关键帧共视程度很高的关键帧作为局部关键帧
                                    +  最佳共视的10帧
                                    +  自己的子关键帧
                                    +  自己的父关键帧
                        +  UpdateLocalPoints();
                            +  清空局部MapPoints
                            +  遍历局部关键帧mvpLocalKeyFrames
                            +  将局部关键帧的MapPoints添加到mvpLocalMapPoints
                    1. SearchLocalPoints();//在局部地图中查找与当前帧匹配的MapPoints
                        1. 遍历当前帧的mvpMapPoints，标记这些MapPoints不参与之后的搜索
                        1. 将所有局部MapPoints投影到当前帧，判断是否在视野范围内，然后进行投影匹配
                            1. 已经被当前帧观测到MapPoint不再判断是否能被当前帧观测到
                            1. 判断LocalMapPoints中的点是否在在视野内
                            1. 只有在视野范围内的MapPoints才参与之后的投影匹配
                            1. 对视野范围内的MapPoints通过投影进行特征点匹配//SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
                    1. 更新局部所有MapPoints后对位姿再次优化//Optimizer::PoseOptimization(&mCurrentFrame)
                    1. 更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
                    1. 决定是否跟踪成功
            * 判断`!mbOnlyTracking`为假：
                * 重定位成功　`if(bOK && !mbVO)`
                    * bOK = TrackLocalMap();
            * `if(bOK)` : `mState = OK;`否则` mState=LOST;`.
            * Update drawer
            * 再次判断`if(bOK)`//If tracking were good, check if we insert a keyframe
                * 是：Update motion model
                    * `if(!mLastFrame.mTcw.empty())`//插入新的关键帧
                        * 真：更新恒速运动模型TrackWithMotionModel中的mVelocity
                            * mVelocity = mCurrentFrame.mTcw*LastTwc;
                        * 否：mVelocity = cv::Mat();
                    * mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
                    * Clean VO matches。在当前帧中将这些MapPoints剔除
                    * 清除临时的MapPoints，这些MapPoints在TrackWithMotionModel的UpdateLastFrame函数里生成（仅双目和rgbd）从MapPoints数据库中删除；这里生成的仅仅是为了提高双目或rgbd摄像头的帧间跟踪效果，用完以后就扔了，没有添加到地图中。
                    * 检测并插入关键帧，对于双目会产生新的MapPoints
                        * if(NeedNewKeyFrame())
                            1. 如果用户在界面上选择重定位，那么将不插入关键帧
                            1. 如果局部地图被闭环检测使用，则不插入关键帧//if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
                            1. 判断是否距离上一次重定位的时间太短//`if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)`
                                * mCurrentFrame.mnId是当前帧的ID
                                * mnLastRelocFrameId是最近一次重定位帧的ID
                                * mMaxFrames等于图像输入的帧率
                                * 如果关键帧比较少，则考虑插入关键帧
                                * 或距离上一次重定位超过1s，则考虑插入关键帧
                                * nKFs:地图里的关键帧
                            1. 得到参考关键帧跟踪到的MapPoints数量
                            1. 查询局部地图管理器是否繁忙
                            1. 对于双目或RGBD摄像头，统计总的可以添加的MapPoints数量和跟踪到地图中的MapPoints数量
                            1. 决策是否需要插入关键帧
                                1. 设定inlier阈值，和之前帧特征点匹配的inlier比例。关键帧只有一帧，那么插入关键帧的阈值设置很低//thRefRatio = 0.4f
                                1. 如果是单目， thRefRatio = 0.9f;
                                1. MapPoints中和地图关联的比例阈值// float thMapRatio = 0.35f;
                                    1. 很长时间没有插入关键帧//const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
                                    1. localMapper处于空闲状态//const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
                                    1. 跟踪要跪的节奏，0.25和0.3是一个比较低的阈值//const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || ratioMap<0.3f) ;
                                1. 阈值比c1c要高，与之前参考帧（最近的一个关键帧）重复度不是太高//const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio || ratioMap<thMapRatio) && mnMatchesInliers>15);
                        * CreateNewKeyFrame();
                            1. 将当前帧构造成关键帧//KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);
                            1. 将当前关键帧设置为当前帧的参考关键帧
                            1. 对于双目或rgbd摄像头，为当前帧生成新的MapPoints
                                1. 根据Tcw计算mRcw、mtcw和mRwc、mOw//mCurrentFrame.UpdatePoseMatrices();
                                1. 得到当前帧深度小于阈值的特征点
                                1. 按照深度从小到大排序
                                1. 将距离比较近的点包装成MapPoints
                                1. 每次创建MapPoint后都要添加地图点属性
                                    1. pNewMP->AddObservation(pKF,i);
                                    1. pKF->AddMapPoint(pNewMP,i);
                                    1. pNewMP->ComputeDistinctiveDescriptors();
                                    1. pNewMP->UpdateNormalAndDepth();
                                    1. mpMap->AddMapPoint(pNewMP);
                                1. mCurrentFrame.mvpMapPoints[i]=pNewMP;
                                1. nPoints>100 退出！
                            1. 局部建图：mpLocalMapper->InsertKeyFrame(pKF);
                            1. mpLocalMapper->SetNotStop(false);
            * if(mState==LOST)
                * 判断地图里的关键帧数量`mpMap->KeyFramesInMap()<=5`
                * mpSystem->Reset();
                *  return;
    *  记录位姿信息，用于轨迹复现
        1.  成功：计算相对姿态T_currentFrame_referenceKeyFrame
        1.  失败：相对位姿使用上一次值
#### Tracking公有函数
* Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor);
```c
Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap,
                   KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);//config文件
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    //     |fx  0   cx|
    // K = |0   fy  cy|
    //     |0   0   1 |
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    // 图像矫正系数
    // [k1 k2 p1 p2 k3]
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    // 双目摄像头baseline * fx 50
    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    // 1:RGB 0:BGR
    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    // 每一帧提取的特征点数 1000
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    // 图像建立金字塔时的变化尺度 1.2
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    // 尺度金字塔的层数 8
    int nLevels = fSettings["ORBextractor.nLevels"];
    // 提取fast特征点的默认阈值 20
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    // tracking过程都会用到mpORBextractorLeft作为特征点提取器
    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    // 如果是双目，tracking过程中还会用用到mpORBextractorRight作为右目特征点提取器
    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    // 在单目初始化的时候，会用mpIniORBextractor来作为特征点提取器
    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        // 判断一个3D点远/近的阈值 mbf * 35 / fx
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        // 深度相机disparity转化为depth时的因子
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}
```
* cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp);
* cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp);
* cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);
* void SetLocalMapper(LocalMapping* pLocalMapper);
* void SetLoopClosing(LoopClosing* pLoopClosing);
* void SetViewer(Viewer* pViewer);
* void ChangeCalibration(const string &strSettingPath);
* void InformOnlyTracking(const bool &flag);
* void Reset();
#### Tracking公有成员函数
* 跟踪状态
```c
enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };
```
* eTrackingState mState;
* eTrackingState mLastProcessedState;
* int mSensor;
* Frame mCurrentFrame;//当前帧
* cv::Mat mImGray;
* `std::vector<int>` mvIniLastMatches;
* `std::vector<int>` mvIniMatches;// 跟踪初始化时前两帧之间的匹配
* `std::vector<cv::Point2f>` mvbPrevMatched;
* `std::vector<cv::Point3f>` mvIniP3D;
* Frame mInitialFrame;
* `list<cv::Mat>` mlRelativeFramePoses;
* `list<KeyFrame*>` mlpReferences;
* `list<double>` mlFrameTimes;
* `list<bool>` mlbLost;
* bool mbOnlyTracking;

#### Tracking私有成员函数
* void Track();// Main tracking function. It is independent of the input sensor.
* void StereoInitialization();// Map initialization for stereo and RGB-D
* void MonocularInitialization();// Map initialization for monocular
* void CreateInitialMapMonocular();
* void CheckReplacedInLastFrame();
* bool TrackReferenceKeyFrame();
* void UpdateLastFrame();
* bool TrackWithMotionModel();
* bool Relocalization();
* void UpdateLocalMap();
* void UpdateLocalPoints();
* void UpdateLocalKeyFrames();
* bool TrackLocalMap();
* void SearchLocalPoints();
* bool NeedNewKeyFrame();
* void CreateNewKeyFrame();  

#### Tracking私有变量

* bool mbVO;
* 其他 Thread Pointers
    * LocalMapping* mpLocalMapper;
    * LoopClosing* mpLoopClosing;
* ORB
    * ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    * ORBextractor* mpIniORBextractor;
* BoW
    * ORBVocabulary* mpORBVocabulary;
    * KeyFrameDatabase* mpKeyFrameDB;
* 单目初始器
    * Initializer* mpInitializer;
* Local Map
    * KeyFrame* mpReferenceKF;// 当前关键帧就是参考帧
    * std::vector<KeyFrame*> mvpLocalKeyFrames;
    * std::vector<MapPoint*> mvpLocalMapPoints;
* System
    * System* mpSystem;
* Drawers
    * Viewer* mpViewer;
    * FrameDrawer* mpFrameDrawer;
    * MapDrawer* mpMapDrawer;
* Map
    * Map* mpMap;
* Calibration matrix
    * cv::Mat mK;
    * cv::Mat mDistCoef;
    * float mbf;
* New KeyFrame rules (according to fps)
    * int mMinFrames，int mMaxFrames;
* Threshold close/far points
    * float mThDepth;
* float mDepthMapFactor;
* Current matches in frame
    * int mnMatchesInliers;
* Last Frame, KeyFrame and Relocalisation Info
    * KeyFrame* mpLastKeyFrame;
    * Frame mLastFrame;
    * unsigned int mnLastKeyFrameId;
    * unsigned int mnLastRelocFrameId;
* Motion Model
    * cv::Mat mVelocity;
* Color order (true RGB, false BGR, ignored if grayscale)
    * bool mbRGB;
* list<MapPoint*> mlpTemporalPoints;

### LocalMapping线程

#### 步骤:  

* void LocalMapping::Run()
    * SetAcceptKeyFrames(false);//告诉Tracking，LocalMapping正处于繁忙状态,LocalMapping线程处理的关键帧都是Tracking线程发过的,在LocalMapping线程还没有处理完关键帧之前Tracking线程最好不要发送太快。
    ```c
    void LocalMapping::SetAcceptKeyFrames(bool flag)
    {
        unique_lock<mutex> lock(mMutexAccept);
        mbAcceptKeyFrames=flag;
    }
    ```
    * 判断等待处理的关键帧列表不为空//CheckNewKeyFrames()//每次执行ProcessNewKeyFrame(),弹出一个
        * if(CheckNewKeyFrames())//如果等待处理的关键帧列表不为空!!!开始循环!!!
            * std::list<KeyFrame*> mlNewKeyFrames; ///< 等待处理的关键帧列表,Tracking线程向LocalMapping中插入关键帧是先插入到该队列中
            * **计算关键帧特征点的BoW映射，将关键帧插入地图//ProcessNewKeyFrame();**
                1. 从缓冲队列中弹出一帧关键帧
                    ```c
                    unique_lock<mutex> lock(mMutexNewKFs);
                    // 从列表中获得一个等待被插入的关键帧
                    mpCurrentKeyFrame = mlNewKeyFrames.front();
                    mlNewKeyFrames.pop_front();
                    ```
                1. 计算该关键帧特征点的Bow映射关系//mpCurrentKeyFrame->ComputeBoW();
                1. 跟踪局部地图过程中新匹配上的MapPoints和当前关键帧绑定  
                在TrackLocalMap函数中将局部地图中的MapPoints与当前帧进行了匹配，但没有对这些匹配上的MapPoints与当前帧进行关联.
                    * 获取当前帧的mvpMapPoints.(与keypoints匹配上的地图点)
                        * `!pMP->IsInKeyFrame(mpCurrentKeyFrame)`//不在当前帧里,没有被赋予观察属性
                            * 添加地图点的属性
                        * `pMP->IsInKeyFrame(mpCurrentKeyFrame)`//this can only happen for new stereo points inserted by the Tracking
                            * mlpRecentAddedMapPoints.push_back(pMP);
                                * 当前帧生成的MapPoints
                                * 将双目或RGBD跟踪过程中新插入的MapPoints放入mlpRecentAddedMapPoints，等待检查
                                * CreateNewMapPoints函数中通过三角化也会生成MapPoints
                                * 这些MapPoints都会经过MapPointCulling函数的检验
                
                1. 更新关键帧间的连接关系，Covisibility图和Essential图(tree)//mpCurrentKeyFrame->UpdateConnections();
                1. 将该关键帧插入到地图中//mpMap->AddKeyFrame(mpCurrentKeyFrame);
            * 剔除ProcessNewKeyFrame函数中引入的不合格MapPoints//MapPointCulling();
                * 检查那些当前帧生成的MapPoints(双目),将双目或RGBD跟踪过程中新插入的MapPoints放入mlpRecentAddedMapPoints，等待检查.//list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
                    * 已经是坏点的MapPoints直接从检查链表中删除//lit = mlpRecentAddedMapPoints.erase(lit);
                        * pMP->SetBadFlag();
                    * 将不满足GetFoundRatio()<0.25f剔除//(mnFound)/mnVisible
                        * 跟踪到该MapPoint的Frame数相比预计可观测到该MapPoint的Frame数的比例需大于25%,注意不一定是关键帧.
                        * pMP->SetBadFlag();
                    * 从该点建立开始，到现在已经过了不小于2个关键帧,但是观测到该点的关键帧数却不超过cnThObs帧，那么该点检验不合格
                        * 单目是2,双目是3
                        * pMP->SetBadFlag();
                    * 从建立该点开始，已经过了3个关键帧而没有被剔除，则认为是质量高的点
                        * 因此没有SetBadFlag()，仅从队列中删除，放弃继续对该MapPoint的检测
            * **相机运动过程中与相邻关键帧通过三角化恢复出一些MapPoints//CreateNewMapPoints();**
                1. 在当前关键帧的共视关键帧中找到共视程度最高的nn帧相邻帧vpNeighKFs//单目20 ,双目10
                1. 得到当前关键帧在世界坐标系中的坐标//cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();
                1. 遍历相邻关键帧vpNeighKFs
                    1. 邻接的关键帧在世界坐标系中的坐标//cv::Mat Ow2 = pKF2->GetCameraCenter();
                    1. 基线向量，两个关键帧间的相机位移//cv::Mat vBaseline = Ow2-Ow1;
                    1. 基线长度//const float baseline = cv::norm(vBaseline);
                    1. 判断相机运动的基线是不是足够长
                        * 如果是立体相机，关键帧间距太小时不生成3D点//if(baseline<pKF2->mb),continue;
                        * 如果单目
                            * 评估遍历的当前关键帧场景深度，q=2表示中值//medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
                            * baseline与景深的比例//ratioBaselineDepth = baseline/medianDepthKF2;
                            * if(ratioBaselineDepth<0.01);continue;
                    1. 根据两个关键帧的位姿计算它们之间的基本矩阵//cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);
                    1. 通过极线约束限制匹配时的搜索范围，进行特征点匹配//matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);
                    1. 对每对匹配通过三角化生成3D点
                        1. 取出匹配特征点对(一对)
                        1. 利用匹配点反投影得到视差角
                            * 特征点反投影(x,y,1)到相机坐标系//cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
                            * 由相机坐标系转到世界坐标系，得到视差角余弦值 
                                * cv::Mat ray1 = Rwc1*xn1;
                                * cv::Mat ray2 = Rwc2*xn2;
                                * cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));
                            * 对于双目，利用双目得到视差角
                                * 当前关键帧是双目的话,atan2(b/2,z)就是视差//cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
                                * 如果匹配的候选的关键帧是双目.//cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));
                                * 取小,也就是角度大的//cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);
                            * 三角化恢复3D点
                                *  视差角度小时用三角法恢复3D点，视差角大时用双目恢复3D点（双目以及深度有效）
                                *  `cosParallaxRays<cosParallaxStereo`:三角法算深度
                                    *  SVD分解
                                *  bStereo1 && `cosParallaxStereo1<cosParallaxStereo2`//当前帧双目,且角度比匹配帧大
                                    *  x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);
                                * bStereo2 && cosParallaxStereo2<cosParallaxStereo1//匹配帧双目,且角度比当前帧大
                                    * x3D = pKF2->UnprojectStereo(idx2);
                                * continue; //No stereo and very low parallax
                            * 检测生成的3D点是否在相机前方
                                * float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);//z>0,
                            * 计算3D点在 当前关键帧( mpCurrentKeyFrame)下 的重投影误差
                            * 计算3D点在 候选关键帧(pKF2)下 的重投影误差    (两头都要检查,互相投影)
                            * 检查尺度连续性
                            * 三角化生成3D点成功，构造成MapPoint
                                * MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);
                            * 为该MapPoint添加属性：
                                * pMP->AddObservation(mpCurrentKeyFrame,idx1);
                                * pMP->AddObservation(pKF2,idx2);
                                * mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
                                * pKF2->AddMapPoint(pMP,idx2);
                                * pMP->ComputeDistinctiveDescriptors();
                                * pMP->UpdateNormalAndDepth();
                                * mpMap->AddMapPoint(pMP);
                            * 将新产生的点放入检测队列//mlpRecentAddedMapPoints.push_back(pMP);
                                * 这些MapPoints都会经过MapPointCulling函数的检验
                            * nnew++;

            * 判断是否已经处理完队列中的最后的一个关键帧// if(!CheckNewKeyFrames())//!!!运行到这里,最后一帧也没了.最后一次循环了!!!
                * SearchInNeighbors();//!!!只在最后一帧处理完的时候执行!!!
                    * 获得当前关键帧在covisibility图中权重排名前nn的邻接关键帧
                        * 找共视帧(10),加入vpTargetKFs;再对这些加入的一级帧,依次挑选他们各自的共视帧(5),二级帧,加入vpTargetKFs
                        * 将当前帧的地图点`mpCurrentKeyFrame->GetMapPointMatches()`,依次与这些`vpTargetKFs`投影融合比较
                            * Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
                                * 如果MapPoint能匹配关键帧的特征点，并且该点有对应的MapPoint，那么将两个MapPoint合并（选择观测数多的）
                                * 如果MapPoint能匹配关键帧的特征点，并且该点没有对应的MapPoint，那么为该点添加MapPoint
                        * 遍历`vpTargetKF`,再依次遍历他们的MapPoints 全部无重复加入`vpFuseCandidates`
                        * Fuse(mpCurrentKeyFrame,vpFuseCandidates);
                        * 更新当前帧MapPoints的描述子，深度，观测主方向等属性
                        * 更新当前帧的MapPoints后更新与其它帧的连接关系//mpCurrentKeyFrame->UpdateConnections();
            * mbAbortBA = false;//确保每次不打断BA优化
            * 已经处理完队列中的最后的一个关键帧，并且闭环检测没有请求停止LocalMapping//if(!CheckNewKeyFrames() && !stopRequested())
                * 判断地图里的关键帧数是不是大于2，如果是就BA
                * 检测并剔除当前帧相邻的关键帧中冗余的关键帧//KeyFrameCulling();在Covisibility Graph中的关键帧，其90%以上的MapPoints能被其他关键帧（至少3个）观测到，则认为该关键帧为冗余关键帧。
                    * 根据Covisibility Graph提取当前帧的共视关键帧//vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();
                    * 提取每个共视关键帧的MapPoints
                    * 遍历该局部关键帧的MapPoints，判断是否90%以上的MapPoints能被其它关键帧（至少3个）观测到
                        * 对于双目:仅考虑近处的MapPoints，不超过mbf * 35 / fx
                        * MapPoints至少被三个关键帧观测到
                            * 判断该MapPoint是否同时被三个关键帧观测到
                            * 尺度约束，要求MapPoint在该局部关键帧的特征尺度大于（或近似于）其它关键帧的特征尺度
                        * 该局部关键帧90%以上的MapPoints能被其它关键帧（至少3个）观测到，则认为是冗余关键帧
            * 将当前帧加入到闭环检测队列中//mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
                * mlpLoopKeyFrameQueue.push_back(pKF);
            * !!!循环结束!!!
        * else if(Stop())//或者可能收到停止局部建图命令
            * while(isStopped() && !CheckFinish())
            * `std::this_thread::sleep_for(std::chrono::milliseconds(3));`
        * 如果需要reset //ResetIfRequested()
            ```c
                void LocalMapping::ResetIfRequested()
                {
                    unique_lock<mutex> lock(mMutexReset);
                    if(mbResetRequested)
                    {
                        mlNewKeyFrames.clear();
                        mlpRecentAddedMapPoints.clear();
                        mbResetRequested=false;
                    }
                }
            ```
    等待处理的关键帧列表处理完了
    * SetAcceptKeyFrames(true);//告诉Tracking，LocalMapping不繁忙
    * 如果收到确认结束`if(CheckFinish())`
        * break;//退出局部见图
    * std::this_thread::sleep_for(std::chrono::milliseconds(3));
    * SetFinish();//mbFinished = false变色了.不进行while(1)循环了.

### LoopClosing线程
#### Tips：

* [Eigen::aligned_allocator的用法](http://blog.csdn.net/rs_huangzs/article/details/50574141)
    * 如果STL容器中的元素是Eigen库数据结构，例如这里定义一个vector容器，元素是Matrix4d ，如下所示：
    `vector<Eigen::Matrix4d>;`  
    * 这个错误比较难发现，因为它在编译的时候是不会提示有错误的，只会在运行的时候提示出错，错误的提示就是`Assertion failed: (reinterpret_cast<size_t>(array) & 0xf) == 0 && "this assertion is explained here:`
    * 这是因为你在使用这个类的时候用到了new方法，这个方法是开辟一个内存，但是呢在上面的代码中没有自定义构造函数，所以在new的时候会调用默认的构造函数，调用默认的构造函数的错误在于内存的位数不对齐，所以会导致程序运行的时候出错。 解决的方法就是在类中加入宏`EIGEN_MAKE_ALIGNED_OPERATOR_NEW`,改为`vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>>;`
#### 步骤
总的来说,和LocalMapping线程类似.只要LocalMapping发过来一个关键帧,他就运行.属于后台一直等待状态`while(1)`
* mbFinished =false;
* 检查闭环检测队列mlpLoopKeyFrameQueue中的关键帧不为空`CheckNewKeyFrames()`  
    // Loopclosing中的关键帧是LocalMapping发送过来的，LocalMapping是Tracking中发过来的  
    //在LocalMapping中通过InsertKeyFrame将关键帧插入闭环检测队列mlpLoopKeyFrameQueue
    * 检测回环`DetectLoop()`
        * 锁住`mMutexLoopQueue`,不让他进来新的关键帧.
        * 弹出当前帧
        * mpCurrentKF->SetNotErase();//Avoid that a keyframe can be erased while it is being process by this thread.你只是检测他是不是回环,不能把他弄没了.
            * unique_lock<mutex> lock(mMutexConnections);
            * mbNotErase = true;
        * 如果距离上次闭环没多久（小于10帧），或者map中关键帧总共还没有10帧，则不进行闭环检测
            * mpKeyFrameDB->add(mpCurrentKF);//首先加入到关键帧的数据库里
            * mpCurrentKF->SetErase();
            * return false;//结束闭环检测
        * 先遍历所有共视关键帧，算个分,阈值.计算当前关键帧与每个共视关键的bow相似度得分，并得到最低得分`minScore`
            * `const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();`
        * 在所有关键帧中找出闭环备选帧,(大于这个最低分`minScore`的所有帧)
            * `vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);`
        * 如果没有这些候选帧.
            * 只是把当前帧加入关键帧数据库
            * mvConsistentGroups.clear();
            * mpCurrentKF->SetErase();
            * return false;
        * 如果有`vCurrentConsistentGroups`
            * mvpEnoughConsistentCandidates.clear();// 这一步应该是清除上次最终筛选后得到的闭环帧
    * 计算相似变换
