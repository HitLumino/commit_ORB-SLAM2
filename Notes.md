# ORBSLAM 源文件
    主要针对ORB-SLAM2代码的研读，作应有的总结与思考。
        伏久者飞必高，开先者谢独早。
        知此，可以免蹭蹬之忧，可以消躁之念。
                                     ----献给自己

## Converter

主要函数接口/方法如下：
- 一个描述子矩阵到一串单行的描述子向量
    + static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);
- cv::Mat 转化成 g2o::SE3Quat
    + static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
- 转化成 CvMat
    + static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
    + static cv::Mat toCvMat(const g2o::Sim3 &Sim3);
    + static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
    + static cv::Mat toCvMat(const Eigen::Matrix3d &m);
    + static cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
    + static cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);
- 转化成 Eigen
    + static Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);
    + static Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);
    + static Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);
    + static std::vector<float> toQuaternion(const cv::Mat &M);

### Converter.h

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

### Converter.cpp

```c
#include "Converter.h"

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
    Eigen::Matrix<double,4,4> eigMat = SE3.to_homogeneous_matrix();
    return toCvMat(eigMat);
}

cv::Mat Converter::toCvMat(const g2o::Sim3 &Sim3)
{
    Eigen::Matrix3d eigR = Sim3.rotation().toRotationMatrix();
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