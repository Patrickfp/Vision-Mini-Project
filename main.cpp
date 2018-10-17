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

 /*   // Image 1
    namedWindow("Image 1", CV_WINDOW_NORMAL);
    resizeWindow("Image 1", 800, 1000);
    imshow("Image 1", img1);
    Mat roi1(img1, Rect(825, 1400, 695, 420));
    imshow("histogram of image 1 at roi", make_histogram(img1));
*/
//    // Image 2
//    Mat img2 = imread("../ImagesForStudents/Image2.png",COLOR_BGR2GRAY);
//    namedWindow("Image 2", CV_WINDOW_NORMAL);
//    resizeWindow("Image 2", 800, 1000);
//    imshow("Image 2", img2);
//    Mat roi2(img2, Rect(825, 1400, 695, 420));
//    Mat temp = adaptiveFilter(img2);
//    imshow("histogram of image 2 med eq", make_histogram(temp, true));
//    imshow("histogram image 2 uden eq", make_histogram(temp, false));
//    namedWindow("Image with", CV_WINDOW_NORMAL);
//    resizeWindow("Image with", 800, 1000);
//    imshow("Image with", equalize_image( temp));
//    namedWindow("Image without", CV_WINDOW_NORMAL);
//    resizeWindow("Image without", 800, 1000);
//    imshow("Image without", temp);



//    // Image 3
//    Mat img3 = imread("../ImagesForStudents/Image3.png");
//    namedWindow("Image 3", CV_WINDOW_NORMAL);
//    resizeWindow("Image 3", 800, 1000);
//    Mat roi3(img3, Rect(825, 1400, 695, 420));
//    imshow("Image 3", img3);
// // Alpha trim

    // Image 4
    Mat img4 = imread("../ImagesForStudents/Image4_1.png",COLOR_BGR2GRAY);
    showimg("Image 4",img4,true);
    Mat roi4(img4, Rect(825, 1400, 695, 420));
    imshow("Image 4 Histogram ROI",make_histogram( roi4, false));
    Mat freq4 = calc_dft(img4);
     //draw_magnitude(freq4);
     Mat freq4_filt;
     Mat filter = Notch_reject(30,1500,700,freq4.size());
     multiply(freq4,filter,freq4_filt);
     draw_magnitude(freq4_filt);
     Mat inv_freq4;
     idft(freq4_filt,inv_freq4);//, DFT_SCALE);// | DFT_REAL_OUTPUT));
     cv::normalize(inv_freq4, inv_freq4, 0, 1, NORM_MINMAX);
     showimg("inv_dft",inv_freq4,true);

  // Notch reject



    // Image 5
//    Mat img5 = imread("../ImagesForStudents/Image5_optional.png");
//    namedWindow("Image 5", CV_WINDOW_NORMAL);
//    resizeWindow("Image 5", 800, 1000);
//    imshow("Image 5", img5);
//    Mat roi5(img5, Rect(825, 1400, 695, 420));

//    namedWindow("Filtered", CV_WINDOW_NORMAL);
//    resizeWindow("Filtered", 800, 1000);
//    Mat temp = adaptiveFilter(img2);
//   // Mat temp2;
//   // cv::add(cv_linear_filter(temp),temp,temp2);
//    imshow("Filtered",temp);
//    Mat roi(temp, Rect(825, 1400, 695, 420));
//    imshow("histogram of filter at roi", make_histogram(roi, true));


     // Wait for escape key press before returning
    while (waitKey() != 27)
        ; // (do nothing)


    return 0;
}
