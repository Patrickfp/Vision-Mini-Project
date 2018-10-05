/*
  RoVi1
  Example application to load and display an image.


  Version: $$version$$
*/

#include <opencv2/highgui.hpp>

#include <iostream>

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
    cv::Mat img = cv::imread(filepath);

    // Check that the image file was actually loaded
    if (img.empty()) {
        std::cout << "Input image not found at '" << filepath << "'\n";
        return 1;
    }

    // Show the image
    cv::imshow("Image", img);

    // Wait for escape key press before returning
    while (cv::waitKey() != 27)
        ; // (do nothing)

    return 0;
}
