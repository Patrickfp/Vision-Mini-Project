#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#endif // FUNCTIONS_H


#include<opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"

using namespace cv;


cv::Mat calc_histogram(const cv::Mat& input)
{
    Mat histo;

    calcHist( std::vector<Mat>{input},{0}, noArray(), histo, {256}, {0, 256});

    return histo;
}

cv::Mat draw_histogram(const cv::Mat& histogram)
{
    int rows = histogram.rows;
    Mat histImage( rows, rows, CV_8UC1, Scalar(0));

    // find max value of our histogram for normalization
    double max;
    minMaxLoc(histogram, nullptr, &max); // only when there is 1 channel (greyscale)


    for (int i = 1; i < rows; i++)
    {
        double h = rows * (histogram.at<float>(i)/max); //normalize
        line(histImage,Point(i, rows), Point(i, rows-h),Scalar::all(255));
    }
    return histImage;
}
