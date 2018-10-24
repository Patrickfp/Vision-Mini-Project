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
     Mat img2 = imread(filepath,COLOR_BGR2GRAY);

    // Check that the image file was actually loaded
    if (img1.empty()) {
        std::cout << "Input image not found at '" << filepath << "'\n";
        return 1;
    }




        // Image 2
        showimg("Image 2", img2, true);
        Mat roi2(img2, Rect(825, 1400, 695, 420));
        imshow("histogram image 2's roi", make_histogram(img2, false));
        Mat temp = adaptiveFilter(img2);
        // imshow("histogram of image 2 with eq", make_histogram(temp, true));
        imshow("histogram image 2 after filtering", make_histogram(temp, false));
        showimg("Image 2 after applying filter", temp, true);
        // adaptive 3x3 to 7x7 filter


    while (waitKey() != 27)
        ; // (do nothing)


    return 0;
}
