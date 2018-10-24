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

Mat adaptiveFilter(const Mat& input)
{
    Mat filtered, padded;
    const int n = 4; // padding
    int count = 0;
    cv::copyMakeBorder(input, filtered, n, n, n, n, BORDER_CONSTANT, Scalar::all(0));
    cv::copyMakeBorder(input, padded, n, n, n, n, BORDER_CONSTANT, Scalar::all(0));
    //std::cout << input.size();
    for (int x = n; x < (filtered.rows - n); x++)
        for (int y = n; y < (filtered.cols - n); y++)
        {
            //std::cout << x << " - " << y  << "\n";

            bool done = false;
            int m = 1;
            while(not(done)) {
                llist l;
                //std::cout << x << " - " << y  << "\n";
                for (int i = x - m; i <= x + m; i++) {
                    for (int j = y - m; j <= y + m; j++) {
                        //std::cout << i << " - " << j  << "\n";
                        l.insertSort(padded.at<uchar>(i, j));
                    }
                }
                //l.display();
                //std::cout << "\n";
                int med = l.median();
                int min = l.min();
                int max = l.max();
                //std::cout <<"m: " << m << std::endl;
                //std::cout << "min: " << min << ", max: " << max << ", med: " << med << "\n";
                //std::cout << (int)filtered.at<uchar>(x,y) << "\n";
                if (med == min || med == max) {
                    m++;
                    if (m > n) {
                        count++;
                        done = true; //filtered.at<uchar>(x,y) = ;
                    }
                }
                else if((int)filtered.at<uchar>(x,y) == min || (int)filtered.at<uchar>(x,y) == max ) {
                    //std::cout << (int)filtered.at<uchar>(x,y) <<", med: " << med << "\n";
                    filtered.at<uchar>(x, y) = med;
                    done = true;
                }
                else {
                    done = true;
                }
                //delete l;
            }
            //std::cout << (int)filtered.at<uchar>(x,y)  << "\n";
        }
        std::cout << count << std::endl;
        return filtered;
}


int main(int argc, char* argv[])
{
    // Parse command line arguments -- the first positional argument expects an
    // image path (the default is ./book_cover.jpg)
    CommandLineParser parser(argc, argv,
            // name  | default value    | help message
                                 "{help   |                  | print this message}"
                                 "{@image | ../ImagesForStudents/Image2.png | image path}"
    );

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    // Load image file
    std::string filepath = parser.get<std::string>("@image");
     Mat img2 = imread(filepath,COLOR_BGR2GRAY);

    // Check that the image file was actually loaded
    if (img2.empty()) {
        std::cout << "Input image not found at '" << filepath << "'\n";
        return 1;
    }


        // Image 2
        showimg("Image 2", img2, true);
        // Region of interest
        Mat roi2(img2, Rect(825, 1400, 695, 420));
        // Histogram image 2 (ROI)
        imshow("histogram image 2's roi", make_histogram(img2, false));
        Mat img2_f = adaptiveFilter(img2);
        Mat roi2_f(img2_f, Rect(825, 1400, 695, 420));
        // Histogram of image 2 with filter (ROI)
        showimg("histogram image 2 after filtering", make_histogram(roi2_f,false), false);
        // Image 2 with filter
        showimg("Image 2 after applying filter", img2_f, true);
        // adaptive 3x3 to 7x7 filter


    while (waitKey() != 27)
        ; // (do nothing)


    return 0;
}
