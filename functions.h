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

Mat calc_dft(const Mat& input)
{
    Mat img, padded;
//    cv::cvtColor(input, img, CV_BGR2GRAY);
    img = input;
    int m = getOptimalDFTSize( img.rows );
    int n = getOptimalDFTSize( img.cols );

    copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));

    std::cout <<padded.size() <<" \n";
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);

    dft(complexI,complexI);

    split(complexI,planes);

    planes[0] += cv::Scalar::all(1);                    // switch to logarithmic scale
    log(planes[0], planes[0]);

    planes[0](cv::Rect(0, 0, planes[0].cols & -2, planes[0].rows & -2));

    int cx = planes[0].cols/2;
    int cy = planes[0].rows/2;

    Mat q0(planes[0], Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(planes[0], Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(planes[0], Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(planes[0], Rect(cx, cy, cx, cy)); // Bottom-Right



    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

   return planes[0];

}

void draw_magnitude(const Mat& magI)
{
    Mat temp;

    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
    cv::namedWindow( "spectrum magnitude", cv::WINDOW_NORMAL);
    imshow("spectrum magnitude", magI);
    cv::resizeWindow("spectrum magnitude", magI.cols/2, magI.rows/2);
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
                        l.append(padded.at<uchar>(i, j));
                }
            }
            int tempval = l.average();
            //std::cout << tempval << "\n";
            filtered.at<uchar>(x, y) = tempval;
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
            int sum = 0;
            int count = 0;
            for (int n = 0; n < 25;n++)
                if(neightbours(n) != 0)
                {
                    sum += neightbours(n);
                    count++;
                }
                if (count > 0)
                    output.at<uchar>(i-2,j-2) = sum/count;

            //output.at<uchar>(i - 1, j - 1) = (neightbours(3) + neightbours(4) + neightbours(5)) / 3;
        }

    }
    return output;
}
cv::Mat cv_linear_filter(const cv::Mat& input)
{
    cv::Mat output;
    cv::Mat_<float> kernel(3, 3);
    kernel << 0,  1,  0,
            1,  -4,  1,
            0,  1,  0;
    cv::filter2D(input, output, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
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

Mat Notch_reject(int notch_size, int x, int y, Size img_size)
{
    Mat notchf(img_size,CV_32F);

    for (int i = 0; i < img_size.height; i++)
    {
        for (int j = 0; j < img_size.width; j++)
        {
            //notchf(i,j)[1] = 0; // Imaginary


            if (i < x+notch_size && i > x)
            {
                if (j < y+notch_size && j > y)
                {
                    notchf.at<float>(i,j) = 0; // Real
                }
            }
            else
            {
                notchf.at<float>(i, j)= 1; // Real
            }


        }
    }
    return notchf;

}
