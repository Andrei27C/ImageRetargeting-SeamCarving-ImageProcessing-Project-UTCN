#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


using namespace cv;
using namespace std;

void rot90(Mat& matImage, int rotflag);

Mat createEnergyImage(Mat& image) {
    Mat image_blur, image_gray;
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat grad, energy_image;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    // apply a gaussian blur to reduce noise
    GaussianBlur(image, image_blur, Size(3, 3), 0, 0, BORDER_DEFAULT);

    // convert to grayscale
    cvtColor(image_blur, image_gray, cv::COLOR_BGR2GRAY);

    // use Sobel to calculate the gradient of the image in the x and y direction
    Sobel(image_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    Sobel(image_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

    // convert gradients to abosulte versions of themselves
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    grad.convertTo(energy_image, CV_64F, 1.0 / 255.0);

    return energy_image;
}

Mat createCumulativeEnergyMap(Mat& energy_image) {
    double a, b, c;
    int rowsize = energy_image.rows;
    int colsize = energy_image.cols;

    Mat cumulative_energy_map = Mat(rowsize, colsize, CV_64F, double(0));

    // copy the first row
    energy_image.row(0).copyTo(cumulative_energy_map.row(0));

    // take the minimum of the three neighbors and add to total, this creates a running sum which is used to determine the lowest energy path
    for (int row = 1; row < rowsize; row++) {
        for (int col = 0; col < colsize; col++) {
            a = cumulative_energy_map.at<double>(row - 1, max(col - 1, 0));
            b = cumulative_energy_map.at<double>(row - 1, col);
            c = cumulative_energy_map.at<double>(row - 1, min(col + 1, colsize - 1));

            cumulative_energy_map.at<double>(row, col) = energy_image.at<double>(row, col) + std::min(a, min(b, c));
        }
    }
    
    return cumulative_energy_map;
}

vector<int> findOptimalSeam(Mat& cumulative_energy_map) {
    double a, b, c;
    int offset = 0;
    vector<int> path;
    double min_val, max_val;
    Point min_pt, max_pt;

    // get the number of rows and columns in the cumulative energy map
    int rowsize = cumulative_energy_map.rows;
    int colsize = cumulative_energy_map.cols;


    // copy the data from the last row of the cumulative energy map
    Mat row = cumulative_energy_map.row(rowsize - 1);

    // get min and max values and locations
    minMaxLoc(row, &min_val, &max_val, &min_pt, &max_pt);

    // initialize the path vector
    path.resize(rowsize);
    int min_index = min_pt.x;
    path[rowsize - 1] = min_index;

    // starting from the bottom, look at the three adjacent pixels above current pixel, choose the minimum of those and add to the path
    for (int i = rowsize - 2; i >= 0; i--) {
        a = cumulative_energy_map.at<double>(i, max(min_index - 1, 0));
        b = cumulative_energy_map.at<double>(i, min_index);
        c = cumulative_energy_map.at<double>(i, min(min_index + 1, colsize - 1));

        if (min(a, b) > c) {
            offset = 1;
        }
        else if (min(a, c) > b) {
            offset = 0;
        }
        else if (min(b, c) > a) {
            offset = -1;
        }

        min_index += offset;
        min_index = min(max(min_index, 0), colsize - 1); // take care of edge cases
        path[i] = min_index;
    }

    return path;
}

void reduce(Mat& image, vector<int> path) {
    int rowsize = image.rows;
    int colsize = image.cols;

    // create a 1x1x3 dummy matrix to add onto the tail of a new row to maintain image dimensions and mark for deletion
    Mat dummy(1, 1, CV_8UC3, Vec3b(0, 0, 0));

    for (int i = 0; i < rowsize; i++) {
        // take all pixels to the left and right of marked pixel and store them in appropriate subrow variables
        Mat new_row;
        Mat lower = image.rowRange(i, i + 1).colRange(0, path[i]);
        Mat upper = image.rowRange(i, i + 1).colRange(path[i] + 1, colsize);

        // merge the two subrows and dummy matrix/pixel into a full row
        if (!lower.empty() && !upper.empty()) {
            hconcat(lower, upper, new_row);
            hconcat(new_row, dummy, new_row);
        }
        else {
            if (lower.empty()) {
                hconcat(upper, dummy, new_row);
            }
            else if (upper.empty()) {
                hconcat(lower, dummy, new_row);
            }
        }
        // take the newly formed row and place it into the original image
        new_row.copyTo(image.row(i));
    }
    // clip 
    image = image.colRange(0, colsize - 1);
}

void showPath(Mat& energy_image, vector<int> path) {
    // loop through the image and change all pixels in the path to white
        for (int i = 0; i < energy_image.rows; i++) {
            energy_image.at<double>(i, path[i]) = 1;
        }
    // display the seam on top of the energy image
    imshow("Seam on Energy Image", energy_image);
}

void shrink(Mat& image, int iterations, string filename, int reduce_direction) {
    Mat imageClone = image.clone();

    if (reduce_direction == 1)
        rot90(imageClone, 2);
    imshow("Original Image", imageClone);

    for (int i = 0; i < iterations; i++) {
        Mat energy_image = createEnergyImage(image);
        Mat cumulative_energy_map = createCumulativeEnergyMap(energy_image);
        vector<int> path = findOptimalSeam(cumulative_energy_map);
        reduce(image, path);
    }

    string resultFileName = filename.substr(0,filename.size()-4);
    resultFileName.append("_result_");
    resultFileName.append(to_string(reduce_direction));
    resultFileName.append("_");
    resultFileName.append(to_string(iterations));
    resultFileName.append(".jpg");

    if (reduce_direction == 1)
        rot90(image, 2);
    imshow("Reduced Image", image); waitKey(0);
    imwrite(resultFileName, image);
}

void rot90(Mat& matImage, int rotflag) 
{
    if (rotflag == 1) {
        transpose(matImage, matImage);
        flip(matImage, matImage, 1);
    }
    else if (rotflag == 2) {
        transpose(matImage, matImage);
        flip(matImage, matImage, 0);
    }
    else if (rotflag == 3) {
        flip(matImage, matImage, -1);
    }
    else if (rotflag != 0) {
        cout << "Unknown rotation flag(" << rotflag << ")" << endl;
    }
}

int main() {
    string filename, width_height, s_iterations;
    int reduce_direction;
    int iterations;

    filename = "Images/";
    filename.append("1.jpg");

    Mat image = imread(filename);
    if (image.empty()) {
        cout << "Unable to load image, please try again." << endl;
        exit(EXIT_FAILURE);
    }
    cout << "Image Size (cols x rows): " << image.cols << " " << image.rows << "\n";

    cout << "Reduce width or reduce height? (0 to reduce width | 1 to reduce height): ";
    cin >> reduce_direction;

    if (reduce_direction == 0) {
        width_height = "width";
    }
    else if (reduce_direction == 1) {
        width_height = "height";
        rot90(image, 1);
    }
    else {
        cout << "Invalid choice, please re-run and try again" << endl;
        return 0;
    }

    cout << "Reduce " << width_height << " how many times? ";
    cin >> s_iterations;

    iterations = stoi(s_iterations);
    int rowsize = image.rows;
    int colsize = image.cols;

    shrink(image, iterations, filename, reduce_direction);

    return 0;
}