#include<opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "linkedlist.h"
#include <iostream>

using namespace cv;


void showimg( std::string name ,cv::Mat img,bool resize)
{
    namedWindow(name, WINDOW_NORMAL);
    if (resize)
    resizeWindow(name, 800, 1000);
    imshow(name, img);
}
// Mostly Taken from https://docs.opencv.org/master/d8/dbc/tutorial_histogram_calculation.html
Mat make_histogram(const Mat& input, bool eq)
{
    Mat histogram;
    calcHist( std::vector<Mat>{input},{0}, noArray(), histogram, {256}, {0, 256});

    if (eq)
    {


        // uses the calc_histogram to calculate the histogram for the image
        Mat transformation_function(histogram.size(), CV_32FC1);
        float c = float(histogram.rows - 1) / (input.rows*input.cols);

        for (int i = 0; i < histogram.rows; i++)
        {
            for (int j = 0; j < i; j++)
            {
                transformation_function.at<float>(i) += histogram.at<float>(j);
            }
             transformation_function.at<float>(i) *= c;
        }

        transformation_function.convertTo(transformation_function, CV_8U);
        LUT(input, transformation_function, histogram); // Look-up table of 256
        // this is also why we convert to 8U

        calcHist( std::vector<Mat>{histogram},{0}, noArray(), histogram, {256}, {0, 256});
    }
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

Mat transform_intensity(const Mat& input, int intensity)
{
    Mat img = input.clone();

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
                int value = img.at<uchar>(i,j) + intensity;
                img.at<uchar>(i,j) = saturate_cast<uchar>(value);

        }
    }
    return img;
}

Mat averageFilter(const Mat& input) {
    Mat padded;
    Mat filtered;
    int n = 1;
    cv::copyMakeBorder(input, filtered, 0, 0, 0, 0, BORDER_CONSTANT, Scalar::all(0));
    cv::copyMakeBorder(input, padded, n, n, n, n, BORDER_CONSTANT, Scalar::all(0));
    for (int x = n; x < (filtered.rows - n); x++)
        for (int y = n; y < (filtered.cols - n); y++)
        {
            llist l;
            //std::cout << x << " - " << y  << "\n";
            for (int i = x - n; i <= x + n; i++) {
                for (int j = y - n; j <= y + n; j++) {
                    //std::cout << i << " - " << j  << "\n";
                    //if ((int)padded.at<uchar>(i, j) > 0 && (int)padded.at<uchar>(i, j) < 255)
                        l.append(padded.at<uchar>(i, j));
                }
            }
            int tempval = l.average();
            //std::cout << tempval << "\n";
            filtered.at<uchar>(x-n, y-n) = tempval;
        }

    return filtered;
}

Mat linear_filter(const Mat& input)
{
    cv::Mat output;
    cv::Mat_<float> kernel(3, 3);
    kernel << -1,  -1,  -1,
            -1,  8,  -1,
            -1,  -1,  -1;
    cv::filter2D(input, output, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::threshold(output,output,50,255,THRESH_TOZERO);
    return output;
}


int main(int argc, char* argv[])
{
    // Parse command line arguments -- the first positional argument expects an
    // image path (the default is ./book_cover.jpg)
    CommandLineParser parser(argc, argv,
            // name  | default value    | help message
                                 "{help   |                  | print this message}"
                                 "{@image | ../ImagesForStudents/Image3.png | image path}"
    );

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    // Load image file
    std::string filepath = parser.get<std::string>("@image");
    Mat img3 = imread(filepath,COLOR_BGR2GRAY);

    // Check that the image file was actually loaded
    if (img3.empty()) {
        std::cout << "Input image not found at '" << filepath << "'\n";
        return 1;
    }



        // Image 3
        showimg("Image 3", img3, true);
        // Region of interest
        Mat roi3(img3, Rect(825, 1400, 695, 420));
        // Histogram of image 3 (ROI)
        showimg("Histogram image 3", make_histogram(roi3,false), false);
        // Image 3 after intensity tranformation
        Mat img3_int = transform_intensity(img3,-40);
        showimg("Image after intensity transformation",img3_int,true);
        // Image 3_int after average filter
        Mat img3_ave = averageFilter(img3_int);
        showimg("Image 3 after average filter", img3_ave, true);
        // Histogram after average and intensity transform (ROI)
        Mat roi3_ave(img3_ave, Rect(825, 1400, 695, 420));
        showimg("Histogram image 3 after intensity and average filter changes",make_histogram(roi3_ave, false),false);
        // Image 3_ave after linear filter
        Mat img3_lin = linear_filter(img3_ave);
        showimg("Image 3 after linear filter", img3_lin, true);
        // Image 3 after adding the linear filter to the averaged imaged from before
        Mat img3_f;
        addWeighted(img3_lin,0.7,img3_ave,1.0,0.0,img3_f);
        showimg("Image 3 after all operations",img3_f,true);
        // Histogram after all the linear filter has also been applyed (ROI)
        Mat roi3_f(img3_f, Rect(825, 1400, 695, 420));
        showimg("Histogram of image 3 after all operations",make_histogram(roi3_f, false), false);

        // Intensity transformation + 3x3 average filter + linear filter to sharpen




    while (waitKey() != 27)
        ; // (do nothing)


    return 0;
}
