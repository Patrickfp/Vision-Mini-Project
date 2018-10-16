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
    Mat img1 = imread("../ImagesForStudents/Image1.png",COLOR_BGR2GRAY);

    // Check that the image file was actually loaded
    if (img1.empty()) {
        std::cout << "Input image not found at '" << filepath << "'\n";
        return 1;
    }

    // Image 1
    namedWindow("Image 1", CV_WINDOW_NORMAL);
    resizeWindow("Image 1", 800, 1000);
    imshow("Image 1", img1);
    Mat roi1(img1, Rect(825, 1400, 695, 420));
    imshow("histogram of image 1 at roi", make_histogram(img1));
/*
    // Image 2
    Mat img2 = imread("../ImagesForStudents/Image2.png",COLOR_BGR2GRAY);
    namedWindow("Image 2", CV_WINDOW_NORMAL);
    resizeWindow("Image 2", 800, 1000);
    imshow("Image 2", img2);
    Mat roi2(img2, Rect(825, 1400, 695, 420));
    imshow("histogram of image 2 at roi", make_histogram(roi2));*/
/*
    // Image 3
    Mat img3 = imread("../ImagesForStudents/Image3.png");
    namedWindow("Image 3", CV_WINDOW_NORMAL);
    resizeWindow("Image 3", 800, 1000);
    Mat roi3(img3, Rect(825, 1400, 695, 420));
    imshow("Image 3", img3);
 // Alpha trim

    // Image 4
    Mat img4 = imread("../ImagesForStudents/Image4_1.png");
    namedWindow("Image 4", CV_WINDOW_NORMAL);
    resizeWindow("Image 4", 800, 1000);
    imshow("Image 4", img4);
    Mat roi4(img4, Rect(825, 1400, 695, 420));*/
  // Notch reject



    // Image 5
//    Mat img5 = imread("../ImagesForStudents/Image5_optional.png");
//    namedWindow("Image 5", CV_WINDOW_NORMAL);
//    resizeWindow("Image 5", 800, 1000);
//    imshow("Image 5", img5);
//    Mat roi5(img5, Rect(825, 1400, 695, 420));

    namedWindow("Filtered", CV_WINDOW_NORMAL);
    resizeWindow("Filtered", 800, 1000);
    Mat temp = AdaptiveFilter(img1);
    imshow("Filtered", temp);
    Mat roi(temp, Rect(825, 1400, 695, 420));
    imshow("histogram of filter at roi", make_histogram(temp));


    // Wait for escape key press before returning
    while (waitKey() != 27)
        ; // (do nothing)


    return 0;
}
