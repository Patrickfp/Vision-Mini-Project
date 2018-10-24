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
                                 "{@image | ../ImagesForStudents/Image1.png | image path}"
    );

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    // Load image file
    std::string filepath = parser.get<std::string>("@image");
    Mat img = imread(filepath,COLOR_BGR2GRAY);

    // Check that the image file was actually loaded
    if (img1.empty()) {
        std::cout << "Input image not found at '" << filepath << "'\n";
        return 1;
    }

       // Image 1
       showimg("Image 1", img, true);
       Mat roi1(img1, Rect(825, 1400, 695, 420));
       imshow("histogram of image 1 at roi", make_histogram(img1, false));
       // Median 5x5 filter no black pixels



    while (waitKey() != 27)
        ; // (do nothing)


    return 0;
}
