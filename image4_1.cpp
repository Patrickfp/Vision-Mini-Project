#include<opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
//#include "functions.h"


#include <iostream>

using namespace cv;
void showimg( std::string name ,cv::Mat img,bool resize)
{
    namedWindow(name, WINDOW_NORMAL);
    if (resize)
        resizeWindow(name, 800, 1000);
    imshow(name, img);
}
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
void dftshift(cv::Mat& mag);
Mat calc_dft(const Mat& input, bool comp)
{
    Mat img, padded;
    img = input;
    //cv::cvtColor(input, img, CV_BGR2GRAY);

    int m = getOptimalDFTSize( img.rows );
    int n = getOptimalDFTSize( img.cols );

    copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));

    //std::cout <<padded.size() <<" \n";
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);

    dft(complexI,complexI);

    split(complexI,planes);

    planes[0] += cv::Scalar::all(1);                    // switch to logarithmic scale
    log(planes[0], planes[0]);

    planes[0](cv::Rect(0, 0, planes[0].cols & -2, planes[0].rows & -2));

    dftshift(planes[0]);

    //std::cout << planes[0].type() << "\n";
    if (!comp)
        return planes[0];
    else
        return planes[1];

}
void draw_magnitude(const Mat& magI, char*  name)
{
    Mat temp;

    normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
    cv::namedWindow( name, cv::WINDOW_NORMAL);
    Mat gray;
    magI.convertTo(gray, CV_8U, 255);
    dftshift(gray);
    imwrite( "Magnitude.png", gray );

    imshow(name, magI);

    cv::resizeWindow(name, magI.cols/2, magI.rows/2);
}
//----------------------- Image Processing ------------------------//
void dftshift(cv::Mat& mag)             // Templated by Kim
{
    int cx = mag.cols / 2;
    int cy = mag.rows / 2;

    cv::Mat tmp;
    cv::Mat q0(mag, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(mag, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(mag, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(mag, cv::Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
void updateMag(Mat complex) {       // taken from https://vgg.fiit.stuba.sk/2012-05/frequency-domain-filtration/?fbclid=IwAR1QUGYdykE8qR_o3HTQKS_SXLDQv8YkLhV8uF0QjKNyga0dElNmACDzDpk
    Mat magI;
    Mat planes[] = {
            Mat::zeros(complex.size(), CV_32F),
            Mat::zeros(complex.size(), CV_32F)
    };
    split(complex, planes); // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], magI); // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
    // switch to logarithmic scale: log(1 + magnitude)
    magI += Scalar::all(1);
    log(magI, magI);
    dftshift(magI); // rearrage quadrants
    // Transform the magnitude matrix into a viewable image (float values 0-1)
    normalize(magI, magI, 1, 0, NORM_INF);
    showimg("spectrum", magI,true);
}

double pythagoreanDistance(int x, int y , int p0, int p1)
{
    return sqrt((x-p0)*(x-p0) + (y-p1)*(y-p1));
}

Mat Notch_reject(int notch_size, int x, int y, Size img_size)
{
    Mat_<Vec2f> notchf(img_size);

    for (int i = 0; i < img_size.height; i++)
    {
        for (int j = 0; j < img_size.width; j++)
        {
            notchf(i,j)[1] = 0; // Imaginary

            float distance =  pythagoreanDistance(x, y, i , j);

            if ( distance > notch_size)
            {
                notchf(i, j)[0]= 1;
            }
            else if(distance < (float)notch_size/4.0f)
            {
                notchf(i,j)[0] = 0.0f;
            }
            else
            {
                notchf(i,j)[0] = pow(distance/(float)notch_size,1);
            }

        }
    }
    return notchf;
}
void Frequency_filtering(Mat& img)      // templated by Kim
{
    cv::Mat padded;
    int opt_rows = cv::getOptimalDFTSize(img.rows * 2 - 1);
    int opt_cols = cv::getOptimalDFTSize(img.cols * 2 - 1);
    cv::copyMakeBorder(img, padded, 0, opt_rows - img.rows, 0, opt_cols - img.cols,
                       cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // Make place for both the real and complex values by merging planes into a
    // cv::Mat with 2 channels.
    // Use float element type because frequency domain ranges are large.
    cv::Mat planes[] = {
            cv::Mat_<float>(padded),
            cv::Mat_<float>::zeros(padded.size())
    };
    cv::Mat complex;
    cv::merge(planes, 2, complex);

    // Compute DFT of image
    cv::dft(complex, complex);2140,

    // Shift quadrants to center
    dftshift(complex);

    // Create a complex filter
    cv::Mat filter,filter1, filter2,filter3,filter4;
    filter = Notch_reject(100,1778,2141,complex.size());
    filter2 = Notch_reject(50,2609,1736,complex.size());
    filter3 = Notch_reject(100,3024,934,complex.size());
    filter4 = Notch_reject(50,2195,1336,complex.size());
    // Multiply filters to same image
    mulSpectrums(filter,filter2,filter,0);
    mulSpectrums(filter,filter3,filter,0);
    mulSpectrums(filter,filter4,filter,0);

    // Multiply Fourier image with filter
    cv::mulSpectrums(complex, filter, complex, 0);

    // Shift back
    dftshift(complex);

    // Visualize filter in frequency spectrum
    updateMag(complex);

    // Compute inverse DFT
    cv::Mat filtered;
    cv::idft(complex, filtered, (cv::DFT_SCALE | cv::DFT_REAL_OUTPUT));

    // Crop image (remove padded borders)
    filtered = cv::Mat(filtered, cv::Rect(cv::Point(0, 0), img.size()));

    // Visualize
    cv::Mat filter_planes[2];
    cv::split(filter, filter_planes); // We can only display the real part
    cv::normalize(filter_planes[0], filter_planes[0], 0, 1, cv::NORM_MINMAX);
    showimg("Filtered image frequency real part", filter_planes[0],true);

    cv::normalize(filtered, filtered, 0, 1, cv::NORM_MINMAX);
    showimg("Filtered image", filtered,true);
}



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
    // Show Real part of frequency spectrum
    Mat freq4 = calc_dft(img4,false);
    draw_magnitude(freq4, "Image 4 Real part of frequency");
    // Filter image in frequency spectrum
    Frequency_filtering(img4);


    while (waitKey() != 27)
        ; // (do nothing)


    return 0;
}
