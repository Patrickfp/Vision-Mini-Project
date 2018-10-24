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
                                 "{@image | ../ImagesForStudents/Image4_1.png | image path}"
    );

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    // Load image file
    std::string filepath = parser.get<std::string>("@image");
    Mat img4 = imread(filepath,COLOR_BGR2GRAY);

    // Check that the image file was actually loaded
    if (img4.empty()) {
        std::cout << "Input image not found at '" << filepath << "'\n";
        return 1;
    }

    // Image 4
    showimg("Image 4",img4,true);
    Mat roi4(img4, Rect(825, 1400, 695, 420));
    imshow("Image 4 Histogram ROI",make_histogram( roi4, false));
    Mat freq4 = calc_dft(img4);
    draw_magnitude(freq4);



    while (waitKey() != 27)
        ; // (do nothing)


    return 0;
}
