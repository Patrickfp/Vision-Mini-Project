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
Mat make_histogram(const cv::Mat& input, bool eq)
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

Mat medianFilter(const Mat& input) {
    //Mat filtered(input.cols,input.rows,input.type());
    Mat padded;
    Mat filtered;
    int n = 2;
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
                    if ((int)padded.at<uchar>(i, j) > 0 && (int)padded.at<uchar>(i, j) < 255)
                        l.insertSort(padded.at<uchar>(i, j));
                }
            }
            int tempval = l.median();
            //std::cout << tempval << "\n";
            filtered.at<uchar>(x-n, y-n) = tempval;
        }

        return filtered;
}


int main(int argc, char* argv[])
{
    // Parse command line arguments -- the first positional argument expects an
    // image path (the default is ./book_cover.jpg)
    CommandLineParser parser(argc, argv,
            // name  | default value    | help message
                                 "{help   |                  | print this message}"
                                 "{@image | ../ImagesForStudents/Image1.png | image path}"
    );

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    // Load image file
    std::string filepath = parser.get<std::string>("@image");
    Mat img1 = imread(filepath,COLOR_BGR2GRAY);

    // Check that the image file was actually loaded
    if (img1.empty()) {
        std::cout << "Input image not found at '" << filepath << "'\n";
        return 1;
    }

       // Image 1
       showimg("Image 1", img1, true);
       // Region of interest
       Mat roi1(img1, Rect(825, 1400, 695, 420));
       // Filtered region of interest
       Mat img1_f = medianFilter(img1);
       Mat roi1_f(img1_f, Rect(825, 1400, 695, 420));
       // Hiistogram of Image 1 (ROI)
       imshow("Histogram of image 1", make_histogram(img1, false));
       // Filtered Image
       showimg("Image 1 with filter",img1_f,true);
       // Histogram of filtered image (ROI)
       showimg("Histogram image 1 with filter",make_histogram(roi1_f,false),false);




    while (waitKey() != 27)
        ; // (do nothing)


    return 0;
}
