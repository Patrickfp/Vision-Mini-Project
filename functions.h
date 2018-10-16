#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#endif // FUNCTIONS_H


#include<opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"

using namespace cv;

// Mostly Taken from https://docs.opencv.org/master/d8/dbc/tutorial_histogram_calculation.html
Mat make_histogram(const cv::Mat& input)
{
    Mat histogram;

    calcHist( std::vector<Mat>{input},{0}, noArray(), histogram, {256}, {0, 256});

    int histSize = 256;

    int hist_w = 257; int hist_h = 256;
    int bin_w = cvRound((double) hist_h/histSize);

    Mat histImage( hist_h, hist_w, CV_8UC1, Scalar(0));


    normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for(int i = 1; i < histSize; i++)
    {
        line(histImage, Point(bin_w*(i+1), hist_h - cvRound(histogram.at<float>(i-1))),
             Point(bin_w*(i), hist_h - cvRound(histogram.at<float>(i))), Scalar::all(255));
    }

    return histImage;
}

Mat calc_dft(const Mat& input)
{
    Mat img, padded;
    cv::cvtColor(input, img, CV_BGR2GRAY);

    int m = getOptimalDFTSize( img.rows );
    int n = getOptimalDFTSize( img.cols );

    copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));

    std::cout <<padded.size() <<" \n";
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);

    dft(complexI,complexI);

    split(complexI,planes);

    planes[0] += cv::Scalar::all(1);                    // switch to logarithmic scale
    log(planes[0], planes[0]);

    planes[0](cv::Rect(0, 0, planes[0].cols & -2, planes[0].rows & -2));

    int cx = planes[0].cols/2;
    int cy = planes[0].rows/2;

    Mat q0(planes[0], Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(planes[0], Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(planes[0], Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(planes[0], Rect(cx, cy, cx, cy)); // Bottom-Right



    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

   return planes[0];

}

void draw_magnitude(const Mat& magI)
{
    Mat temp;

    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
    cv::namedWindow( "spectrum magnitude", cv::WINDOW_NORMAL);
    imshow("spectrum magnitude", magI);
    cv::resizeWindow("spectrum magnitude", magI.cols/2, magI.rows/2);
}
