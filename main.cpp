/*
  RoVi1
  Example application to load and display an image.


  Version: $$version$$
*/

#include<opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "functions.h"


#include <iostream>

using namespace cv;


int main(int argc, char* argv[])
{
    // Parse command line arguments -- the first positional argument expects an
    // image path (the default is ./book_cover.jpg)
    cv::CommandLineParser parser(argc, argv,
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
    cv::Mat img1 = cv::imread("../ImagesForStudents/Image1.png");

    // Check that the image file was actually loaded
    if (img1.empty()) {
        std::cout << "Input image not found at '" << filepath << "'\n";
        return 1;
    }

    // Image 1
    cv::namedWindow("Image 1", CV_WINDOW_NORMAL);
    cv::resizeWindow("Image 1", 800, 1000);
    cv::imshow("Image 1", img1);

    // Image 2
    cv::Mat img2 = cv::imread("../ImagesForStudents/Image2.png");
    cv::namedWindow("Image 2", CV_WINDOW_NORMAL);
    cv::resizeWindow("Image 2", 800, 1000);
    cv::imshow("Image 2", img2);

    Mat histogram_image2 = calc_histogram(img2);
    imshow("histogram of Image 2", draw_histogram(histogram_image2));

    // Image 3
    cv::Mat img3 = cv::imread("../ImagesForStudents/Image3.png");
    cv::namedWindow("Image 3", CV_WINDOW_NORMAL);
    cv::resizeWindow("Image 3", 800, 1000);
    cv::imshow("Image 3", img3);

    // Image 4
    cv::Mat img4 = cv::imread("../ImagesForStudents/Image4_1.png");
    cv::namedWindow("Image 4", CV_WINDOW_NORMAL);
    cv::resizeWindow("Image 4", 800, 1000);
    cv::imshow("Image 4", img4);
    // Image 5
    cv::Mat img5 = cv::imread("../ImagesForStudents/Image5_optional.png");
    cv::namedWindow("Image 5", CV_WINDOW_NORMAL);
    cv::resizeWindow("Image 5", 800, 1000);
    cv::imshow("Image 5", img5);


    // Wait for escape key press before returning
    while (cv::waitKey() != 27)
        ; // (do nothing)

    return 0;
}
