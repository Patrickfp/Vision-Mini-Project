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
    if (img1.empty()) {
        std::cout << "Input image not found at '" << filepath << "'\n";
        return 1;
    }



        // Image 3
        showimg("Image 3", img3, true);
        Mat roi3(img3, Rect(825, 1400, 695, 420));
        Mat freq3 = calc_dft(img3);
        draw_magnitude(freq3);
        imshow("histogram image 3", make_histogram(temp, false));
        // Intensity transformation + 3x3 average filter + linear filter to sharpen




    while (waitKey() != 27)
        ; // (do nothing)


    return 0;
}
