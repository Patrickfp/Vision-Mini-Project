#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#endif // FUNCTIONS_H


#include<opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "linkedlist.h"


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
Mat equalize_image(cv::Mat& input)
{
    Mat histo;
    calcHist( std::vector<Mat>{input},{0}, noArray(), histo, {256}, {0, 256});
    // uses the calc_histogram to calculate the histogram for the image
    Mat transformation_function(histo.size(), CV_32FC1);
    float c = float(histo.rows - 1) / (input.rows*input.cols);

    for (int i = 0; i < histo.rows; i++)
    {
        for (int j = 0; j < i; j++)
        {
            transformation_function.at<float>(i) += histo.at<float>(j);
        }
        transformation_function.at<float>(i) *= c;
    }

    transformation_function.convertTo(transformation_function, CV_8U);
    Mat equalized_output;
    LUT(input, transformation_function, equalized_output); // Look-up table of 256
    // this is why we convert to 8U

    return equalized_output;
}
void dftshift(cv::Mat& mag)
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
void updateMag(Mat complex) {
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

    //dftshift(planes[0]);

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

//------------------- Filters -----------------------//

Mat medianFilter(const Mat& input) {
    //Mat filtered(input.cols,input.rows,input.type());
    Mat padded;
    Mat filtered;
    int n = 2;
    cv::copyMakeBorder(input, filtered, n, n, n, n, BORDER_CONSTANT, Scalar::all(0));
    cv::copyMakeBorder(input, padded, n, n, n, n, BORDER_CONSTANT, Scalar::all(0));
    for (int x = n; x < (filtered.rows - n); x++)
        for (int y = n; y < (filtered.cols - n); y++)
        {
            llist l;
            //std::cout << x << " - " << y  << "\n";
            for (int i = x - n; i <= x + n; i++) {
                for (int j = y - n; j <= y + n; j++) {
                    //std::cout << i << " - " << j  << "\n";
                    if ((int)padded.at<uchar>(i, j) > 0 && (int)padded.at<uchar>(i, j) < 255)
                        l.insertSort(padded.at<uchar>(i, j));
                }
            }
            int tempval = l.median();
            //std::cout << tempval << "\n";
            filtered.at<uchar>(x, y) = tempval;
        }

        return filtered;
}
Mat averageFilter(const Mat& input) {
    Mat padded;
    Mat filtered;
    int n = 1;
    cv::copyMakeBorder(input, filtered, 0, 0, 0, 0, BORDER_CONSTANT, Scalar::all(0));
    cv::copyMakeBorder(input, padded, n, n, n, n, BORDER_CONSTANT, Scalar::all(0));
    for (int x = n; x < (filtered.rows - n); x++)
        for (int y = n; y < (filtered.cols - n); y++)
        {
            llist l;
            //std::cout << x << " - " << y  << "\n";
            for (int i = x - n; i <= x + n; i++) {
                for (int j = y - n; j <= y + n; j++) {
                    //std::cout << i << " - " << j  << "\n";
                    //if ((int)padded.at<uchar>(i, j) > 0 && (int)padded.at<uchar>(i, j) < 255)
                        l.append(padded.at<uchar>(i, j));
                }
            }
            int tempval = l.average();
            //std::cout << tempval << "\n";
            filtered.at<uchar>(x-n, y-n) = tempval;
        }

    return filtered;
}

cv::Mat linear_filter(const cv::Mat& input)
{
    cv::Mat output;
    copyMakeBorder(input, output, 2, 2, 2, 2, BORDER_REFLECT);

    for (int i = 2; i < output.rows - 2; i++)
    {
        uchar* fiveRows[] = {
                output.ptr(i -2),
                output.ptr(i -1),
                output.ptr(i),
                output.ptr(i + 1),
                output.ptr(i +2)
        };

        for (int j = 2; j < output.cols - 2; j++)
        {
            Vec<uchar, 25> neightbours;
            neightbours << fiveRows[0][j-2], fiveRows[1][j-2], fiveRows[2][j-2], fiveRows[3][j-2],fiveRows[4][j-2],
                    fiveRows[0][j-1], fiveRows[1][j-1], fiveRows[2][j-1], fiveRows[3][j-1],fiveRows[4][j-1],
                    fiveRows[0][j], fiveRows[1][j], fiveRows[2][j], fiveRows[3][j],fiveRows[4][j],
                    fiveRows[0][j+1], fiveRows[1][j+1], fiveRows[2][j+1], fiveRows[3][j+1],fiveRows[4][j+1],
                    fiveRows[0][j+2], fiveRows[1][j+2], fiveRows[2][j+2], fiveRows[3][j+2],fiveRows[4][j+2];


            sort(neightbours, neightbours, SORT_EVERY_COLUMN | SORT_ASCENDING);
            /*int sum = 0;
            int count = 0;
            for (int n = 0; n < 25;n++)
                if(neightbours(n) != 0)
                {
                    sum += neightbours(n);
                    count++;
                }
                if (count > 0)
                    output.at<uchar>(i-2,j-2) = sum/count;*/

            output.at<uchar>(i - 1, j - 1) = (neightbours(10) + neightbours(11) + neightbours(12)+neightbours(13)+neightbours(14)) / 5;
        }

    }
    return output;
}
cv::Mat cv_linear_filter(const cv::Mat& input)
{
    cv::Mat output;
    cv::Mat_<float> kernel(3, 3);
    kernel << -1,  -1,  -1,
            -1,  8,  -1,
            -1,  -1,  -1;
    cv::filter2D(input, output, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::threshold(output,output,50,255,THRESH_TOZERO);
    return output;
}

cv::Mat transform_intensity(const cv::Mat& input, int intensity)
{
    Mat img = input.clone();

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
                int value = img.at<uchar>(i,j) + intensity;
                img.at<uchar>(i,j) = saturate_cast<uchar>(value);

        }
    }
    return img;
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

Mat Notch_reject(int notch_size, int x, int y, Size img_size)
{
    Mat_<Vec2f> notchf(img_size);

    for (int i = 0; i < img_size.height; i++)
    {
        for (int j = 0; j < img_size.width; j++)
        {
            notchf(i,j)[1] = 0; // Imaginary


            if (i > x + notch_size/2 || j > y + notch_size/2 || j < y-notch_size/2 || i < x-notch_size/2)
            {
                notchf(i, j)[0]= 1; // Real
            } else
            {
                notchf(i,j)[0] = 0; // Real
            }

        }
    }
/*
    for (int i = x; i < notch_size;i++)
        for (int j = y; j < notch_size; j++) {
            notchf(i,j)[0] = 0; // Real
        }
*/
    return notchf;
}
void drawfil(Mat_<Vec2f>& img, int x, int y, float val)
{

    if( (img(x-1,y)[0] > val &&  img(x,y-1)[0]  > val && img(x+1,y)[0]  > val && img(x,y+1)[0] > val && img(x,y)[0] > val) /* || val == 1*/)
    {
        std::cout << "nu "<< x << " - " << y << std::endl;
        return;
    }
    if(img(x-1,y)[0] > val)
        drawfil(img,x-1,y,val);
    if(img(x,y-1)[0] > val)
     drawfil(img,x,y-1,val);
    if(img(x+1,y)[0] > val)
      drawfil(img,x+1,y,val);
    if(img(x,y+1)[0] > val)
      drawfil(img,x,y+1,val);
    img(x,y)[0] = val;
    std::cout << x << " - " << y << std::endl;
}
Mat Notch_reject2(int notch_size, int x, int y, Size img_size)
{
    Mat_<Vec2f> notchf(img_size);

    for (int i = 0; i < img_size.height; i++)
    {
        for (int j = 0; j < img_size.width; j++)
        {
            notchf(i,j)[1] = 0; // Imaginary
            notchf(i,j)[0] = 1;

        }
    }
    std::cout << " fuuuuuu" << std::endl;
    for (int i = 0; i < notch_size; i++)
    {
        notchf(x,y)[0] = 0;
        drawfil(notchf,x,y,0);
    }

    return notchf;

}
double calcdist(int x, int y , int p0, int p1)
{
    return sqrt((x-p0)*(x-p0) + (y-p1)*(y-p1));
}

Mat Notch_reject1(int notch_size, int x, int y, Size img_size)
{
    Mat_<Vec2f> notchf(img_size);

    for (int i = 0; i < img_size.height; i++)
    {
        for (int j = 0; j < img_size.width; j++)
        {
            notchf(i,j)[1] = 0; // Imaginary

            if (calcdist(x, y, i , j) > notch_size)
            {
                notchf(i, j)[0]= 1; // Real
            } else
            {
                notchf(i,j)[0] = pow((float)calcdist(x, y, i , j)/(float)notch_size,7); // Real
            }

        }
    }
/*
    for (int i = x; i < notch_size;i++)
        for (int j = y; j < notch_size; j++) {
            notchf(i,j)[0] = 0; // Real
        }
*/
    return notchf;
}
Mat fig4filter()

{

}
void cheat(Mat& img)
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
    cv::Mat filter_plane[2];
    cv::split(complex, filter_plane); // We can only display the real part
    Mat gray;
    filter_plane[0].convertTo(gray, CV_8U, 255);
    //dftshift(gray);
    imwrite( "Magnitude.png", gray );
    showimg("mag1",gray,true);

    // Create a complex filter
    cv::Mat filter,filter1, filter2,filter3,filter4;
    //std::cout << padded.size() << std::endl;
    filter = Notch_reject1(500,1778,2141,complex.size());
    filter2 = Notch_reject1(30,2609,1736,complex.size());
    filter3 = Notch_reject1(500,3024,934,complex.size());
    filter4 = Notch_reject1(30,2195,1336,complex.size());
    mulSpectrums(filter,filter2,filter,0);
    mulSpectrums(filter,filter3,filter,0);
    mulSpectrums(filter,filter4,filter,0);

    // Multiply Fourier image with filter
    cv::mulSpectrums(complex, filter, complex, 0);

    /*cv::mulSpectrums(complex, filter2, complex, 0);
    cv::mulSpectrums(complex, filter3, complex, 0);
    cv::mulSpectrums(complex, filter4, complex, 0);*/


    // Shift back
    dftshift(complex);

    updateMag(complex);
    // Compute inverse DFT
    cv::Mat filtered;
    cv::idft(complex, filtered, (cv::DFT_SCALE | cv::DFT_REAL_OUTPUT));

    // Crop image (remove padded borders)
    filtered = cv::Mat(filtered, cv::Rect(cv::Point(0, 0), img.size()));

    // Visualize
    showimg("Input", img,true);

    cv::Mat filter_planes[2];
    cv::split(filter, filter_planes); // We can only display the real part
    cv::normalize(filter_planes[0], filter_planes[0], 0, 1, cv::NORM_MINMAX);
    showimg("Filter", filter_planes[0],true);

    cv::normalize(filtered, filtered, 0, 1, cv::NORM_MINMAX);
    showimg("Filtered image", filtered,true);
}