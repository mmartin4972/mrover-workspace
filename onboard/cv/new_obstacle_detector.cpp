#include "perception.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>

using namespace cv;
using namespace std;

#include <ctime>

#include <opencv2/opencv.hpp>

static void draw_object(cv::Mat image, unsigned int x, unsigned int y, unsigned int width=50, unsigned int height=100)
{
    image(cv::Range(y-height, y), cv::Range(x-width/2, x+width/2)) = image.at<unsigned char>(y, x);
}

bool less_by_y(const cv::Point& lhs, const cv::Point& rhs)
{
  return lhs.y < rhs.y;
}

Mat vDisparity(Mat &depthIn, Mat &color)
{
    patchNaNs(depthIn, 0.0);
    depthIn = max(depthIn, 0.7);
    depthIn = min(depthIn, 6);

    unsigned int IMAGE_HEIGHT = depthIn.rows;
    unsigned int IMAGE_WIDTH = depthIn.cols;
    unsigned int MAX_VAL = 400;
    unsigned int BINS = 200;
    float MIN_VAL = 0;
    unsigned int CYCLE = 0;

    //setenv("QT_GRAPHICSSYSTEM", "native", 1);


    // === PREPERATIONS ==
    cv::Mat image = cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8U);
    cv::Mat uhist = cv::Mat::zeros(IMAGE_HEIGHT, BINS, CV_32F);
    cv::Mat vhist = cv::Mat::zeros(BINS, IMAGE_WIDTH, CV_32F);

    cv::Mat tmpImageMat, tmpHistMat;

    float value_ranges[] = {MIN_VAL, (float)MAX_VAL};
    const float* hist_ranges[] = {value_ranges};
    int channels[] = {0};
    int histSize[] = {BINS};


    struct timespec start, finish;
    double elapsed;

    
    
        CYCLE++;

        // === CLEANUP ==
        image = cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8U);
        normalize(depthIn, image, MIN_VAL, MAX_VAL, NORM_MINMAX, CV_8U);
        cerr<<image.at<unsigned int>(300,300)<<endl;
        vhist = cv::Mat::zeros(IMAGE_HEIGHT, BINS, CV_32F);
        uhist = cv::Mat::zeros(BINS, IMAGE_WIDTH, CV_32F);

        /*
        // === CREATE FAKE DISPARITY WITH OBJECTS ===
        for(int i = 0; i < IMAGE_HEIGHT; i++)
            image.row(i) = ((float)i / IMAGE_HEIGHT * MAX_DISP);

        draw_object(image, 200, 500);
        draw_object(image, 525 + CYCLE%100, 275);
        draw_object(image, 500, 300 + CYCLE%100);
        */
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // === CALCULATE V-HIST ===
        for(int i = 0; i < IMAGE_HEIGHT; i++)
        {
            tmpImageMat = image.row(i);
            vhist.row(i).copyTo(tmpHistMat);
            
            cv::calcHist(&tmpImageMat, 1, channels, cv::Mat(), tmpHistMat, 1, histSize, hist_ranges, true, false);
            vhist.row(i) = tmpHistMat.t() / (float) IMAGE_HEIGHT;
        }
        clock_gettime(CLOCK_MONOTONIC, &finish);
        elapsed = (finish.tv_sec - start.tv_sec);
        elapsed += (finish.tv_nsec - start.tv_nsec) * 1e-9;
        cout << "V-HIST-TIME: " << elapsed << endl;

        clock_gettime(CLOCK_MONOTONIC, &start);

        // === CALCULATE U-HIST ===
        for(int i = 0; i < IMAGE_WIDTH; i++)
        {
            tmpImageMat = image.col(i);
            uhist.col(i).copyTo(tmpHistMat);

            cv::calcHist(&tmpImageMat, 1, channels, cv::Mat(), tmpHistMat, 1, histSize, hist_ranges, true, false);

            uhist.col(i) = tmpHistMat / (float) IMAGE_WIDTH;
        }

        clock_gettime(CLOCK_MONOTONIC, &finish);
        elapsed = (finish.tv_sec - start.tv_sec);
        elapsed += (finish.tv_nsec - start.tv_nsec) * 1e-9;
        cout << "U-HIST-TIME: " << elapsed << endl;

        uhist.convertTo(uhist, CV_8U, 255);
        cv::applyColorMap(uhist, uhist, cv::COLORMAP_JET);

        vhist.convertTo(vhist, CV_8U, 255);
        cv::applyColorMap(vhist, vhist, cv::COLORMAP_JET);

        cv::imshow("image", image);
        cv::imshow("uhist", uhist);
        cv::imshow("vhist", vhist);
/*
        // === Canny Contour Detection V ===
        int edgeThresh = 1;
        int lowThreshold = 1;
        int const max_lowThreshold = 100;
        int ratio = 3;
        int kernel_size = 3;
        Mat cannyContoursv;
        Mat vhist_gray;
        Mat uhist_gray;
        Mat cannyContoursu;

        cvtColor(vhist, vhist_gray, CV_BGR2GRAY);
        cannyContoursv = cv::Mat::zeros(IMAGE_HEIGHT,IMAGE_WIDTH, CV_8UC1);
        //Gaussian blur before canny
        GaussianBlur(vhist_gray, vhist_gray, Size(5,5), 0);
        Canny(vhist_gray, cannyContoursv, lowThreshold, lowThreshold*ratio, kernel_size );
        imshow("Cannyv", cannyContoursv);
        // === Hough Line Transform V ====
    
        vector<Vec4i> linesv;
        HoughLinesP( cannyContoursv, linesv, 1, CV_PI/180, 50, 30, 20 );
        for( size_t i = 0; i < linesv.size(); i++ )
        {
            line( vhist, Point(linesv[i][0], linesv[i][1]),
            Point( linesv[i][2], linesv[i][3]), Scalar(0,0,255), 3, 8 );
        }
        imshow("Houghv", vhist);

        // === Canny Contour Detection U ===
        vector<Vec4i> linesu;
        cvtColor(uhist, uhist_gray, CV_BGR2GRAY);
        cannyContoursu = cv::Mat::zeros(IMAGE_HEIGHT,IMAGE_WIDTH, CV_8UC1);
        //Filter out really small and really large numbers
        //Gaussian blur before canny
        GaussianBlur(vhist_gray, vhist_gray, Size(5,5), 0);
        Canny(uhist_gray, cannyContoursu, lowThreshold, lowThreshold*ratio, kernel_size );
        imshow("Cannyu", cannyContoursu);
        // === Hough Line Transform U ====
    
        
        HoughLinesP( cannyContoursu, linesu, 1, CV_PI/180, 80, 90, 10 );
        for( size_t i = 0; i < linesu.size(); i++ )
        {
            line( uhist, Point(linesu[i][0], linesu[i][1]),
            
            Point( linesu[i][2], linesu[i][3]), Scalar(0,0,255), 3, 8 );
            
        }
        cerr<<"Total:"<<linesu.size()<<endl;
        imshow("Houghu", uhist);

        */

       // === Different Method using Contours Instead of Lines ===
       // === Canny Contour Detection V ===

       //Basic Filter for Contours
        Mat vContours, vContours_gray;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        vContours = vhist;
        vContours_gray.convertTo(vContours_gray, CV_8U);
        cvtColor(vContours, vContours_gray, COLOR_BGR2GRAY);
        imshow("Gray", vContours_gray);
        double thresh = 17;
        double maxValue = 255;
        threshold(vContours_gray,vContours_gray, thresh, maxValue, THRESH_BINARY);
        findContours(vContours_gray,contours,hierarchy,RETR_LIST,CV_CHAIN_APPROX_NONE,Point(0,0));
        drawContours(vContours, contours, -1, Scalar(0,255,0), 3);
        int count = 0;
        for(size_t i = 0; i < contours.size(); ++i)
        {
            if(contours[i].size() > 20)
            {
                Point max = *max_element(contours[i].begin(), contours[i].end(), less_by_y);
                Point min = *min_element(contours[i].begin(), contours[i].end(), less_by_y);
                cerr<< max << " " << min << endl;
                circle(vContours,max, 2, Scalar(0,0,255), -1, 1, 1);
                circle(vContours,min, 2, Scalar(0,0,255), -1, 1, 1);  
                count++;
            }
            
        }
        cerr<<endl<<"Contours Drawn: "<<count<<endl;
        imshow("Contours", vContours);
        

/*
        int angle = 5;
        for( size_t i = 0; i < linesv.size(); i++ )
        {
            for(size_t j = 0; j < linesu.size(); j++)
            {
                if(linesv[i][2] < 60 && linesv[i][2] > 8)
                    if(fabs(linesv[i][0]-linesv[i][2]) < angle)
                        if(linesu[j][3] < 60 && linesu[j][3] > 8)
                            if(fabs(linesu[j][1]-linesu[j][3]) < angle)    
                                if(fabs(linesv[i][0]-linesu[j][1]) < angle && fabs(linesv[j][2]-linesu[j][3]) < angle)
                                {
                                    rectangle(color, Point(linesu[i][0],linesv[i][1]), Point(linesu[i][2],linesv[i][3]), Scalar(0,0,255), 1, 8, 0);
                                    cerr<<"OVERLAP"<<endl;
                                }
            }
            
        }
    imshow("Final", color);
    */
    return image;
}
/*
cerr<<"Point 1: "<<linesu[i][0]<<", "<<linesu[i][1];
            cerr<<" Point 2: "<<linesu[i][2]<<", "<<linesu[i][3]<<endl;

int value1 = 50;
int value2 = 50;
int value3 = 50;
//Function that writes depths to a file given a Matrix with int type values
void writeDepths(Mat &depthIn, ofstream &depth)
{
    //Get size
    int cols = depthIn.cols;
    int rows = depthIn.rows;

    //Output cols and rows
    cerr<<cols<<endl;
    cerr<<rows<<endl;
    
    //Write depth values to depth file stream
    for(int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            depth<<depthIn.at<unsigned int>(y,x,0)<<" ";
            //cerr<<depthIn.at<unsigned int>(y,x,0)<<" ";
        }
        depth<<endl<<endl;
        //cerr<<endl<<endl;
    }

    depth.close();
}

//Does calcualations to find normalization multiplier for assigning disparity matrix values
float* calculateMultiplier(float max_depth, float step)
{
    float out[2];
    float columns = max_depth/step;
    float multiplier = columns/max_depth;
    cerr<<columns<<endl;
    cerr<<multiplier<<endl;
    out[0] = columns;
    out[1] = multiplier;
    float* end = &(out[0]);
    return end;
}

//Filter vMap
void updateValues(int val1, int val2, int val3)
{
    value1 = val1;
    value2 = val2;
    value3 = val3;
}

//Calculates vDisparity Map
Mat vDisparity(Mat &depthIn)
{
    patchNaNs(depthIn, 0.0);
    depthIn = max(depthIn, 0.7);
    depthIn = min(depthIn, 10.0);

    //Get size
    int cols = depthIn.cols;
    int rows = depthIn.rows;
    
    //Calculate multiplier and column size for maps
    float* temp = calculateMultiplier(9.3, 0.04);
    int imgCols = *temp;
    float multiplier = *(temp+1);
    cerr<<"Image Columns: "<<imgCols<<endl;
    cerr<<"Multiplier: "<<multiplier<<endl;

    //Create empty vMap
    Mat vMap(rows, imgCols, CV_8UC1, Scalar(0));
    //Traverse through image and increment values in vMap that correspond with depth
    for(int y = 0; y < rows; ++y)
    {
    
        for(int x = 0; x < cols; ++x)
        {
            int column = int((depthIn.at<float>(y, x, 0)-0.7)*multiplier);
            cout<<"Col:"<<column<< " ";
            cout<<"D: "<<depthIn.at<float>(y,x)<<" ";
            vMap.at<int>(y, column) = vMap.at<int>(y, column)+1;
            cout<<vMap.at<int>(y, column)<<" "<<endl;
        }
        cout<<endl<<endl<<endl;

    }
    
   int bins = 20;
   float range[] = {0.7, 10.0};
   const float* histRange = {range};
   int x[720][20];
   Mat vMap(720, 20, CV_8U);
    //OpenCV Histogram approach
    for(int i = 0; i < rows; ++i){
        calcHist(&depthIn.row(i), 1, 0, Mat(), x[i], bins, true, false);
    }
    memcpy(vMap.data, x, 720*20*sizeof(int));
    imshow("Test", vMap);
    waitKey();
    ofstream out;
    out.open("vMap.txt");
    writeDepths(vMap, out);
    out.close();
    waitKey();
    
    //Write out rows and cols for debugging
    cout<<rows<<endl;
    cout<<cols<<endl;

    //Basic Filter for Contours
    Mat basicContours(rows,imgCols, CV_8UC1, Scalar(0));
    vector<vector<Point> > contours;
    Scalar color = {255,0,0};
    vector<Vec4i> hierarchy;
    vMap.convertTo(vMap,CV_8UC1);
    basicContours = vMap;
    findContours(basicContours,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0,0));
    drawContours(basicContours, contours, 0, color, -1);
    cerr<<"Fish"<<endl;
    cvtColor(basicContours, basicContours, COLOR_GRAY2BGR);
    basicContours.convertTo(basicContours, CV_8UC1);
    vector<Vec4i> lines;
    cerr<<"RIP"<<endl;
    
    HoughLinesP( basicContours, lines, 10, CV_PI/180, 80, 30, 10 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
        line( basicContours, Point(lines[i][0], lines[i][1]),
        Point( lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
    }
    imshow("Basic",basicContours);
    

    //Canny Filter for contours
    //vMap.convertTo(vMap,CV_8UC1);
    int edgeThresh = 1;
    int lowThreshold = 10;
    int const max_lowThreshold = 100;
    int ratio = 3;
    int kernel_size = 3;
    cerr<<"Weird"<<endl;
    Mat cannyContours(rows,imgCols, CV_8UC1, Scalar(0));
    cerr<<"Gotcha"<<endl;
    Mat dst(rows,imgCols, CV_8UC1, Scalar(0));
    cerr<<"Interseting"<<endl;
   //Gaussian blur before canny
   GaussianBlur(vMap, vMap, Size(5,5), 0);
    cerr<<"Ok"<<endl;
    Canny(vMap, cannyContours, lowThreshold, lowThreshold*ratio, kernel_size );
    vMap.copyTo(dst,cannyContours);
    imshow("Canny", dst);

    //Hough Line Transform
    Mat cdstP;
    cvtColor(dst, cdstP, COLOR_GRAY2BGR);
    // Probabilistic Line Transform
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP(dst, linesP, 10, CV_PI/180, 80, value2, value3 ); // runs the actual detection
    // Draw the lines
    for( size_t i = 0; i < linesP.size(); i++ )
    {
        Vec4i l = linesP[i];
        line( cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, LINE_AA);
    }
    
    imshow("Hough", cdstP);
   cvWaitKey(0);
    
   waitKey(0);
    return vMap;

}
*/


















































/*
What needs to be done: 
If I want this to work what I need to do is to find away to filter out the ground that
way I can identify objects that are on the ground and just focus on finding contours
without the background noise of the ground. If I am able to filter out the ground my contour
and hough line transformation should become adequate for identifying areas containing pixels
in close proximity. Once I find the contours of these pixels I can itterate through the contour
vector and check those values against those on the u map and we should be chilling. Right now
I need to focus on filtering out the ground. 2/21/20
*/


/*
 //Image Erosion Techniques

    //blur( vMap, vMap, Size(3,3) );    
    
    //Morphological Erosion
    
    int erosion_elem = 0.3;
    int erosion_size = 0.3;
    int dilation_elem = 15;
    int dilation_size = 15;
    int const max_elem = 2;
    int const max_kernel_size = 21;
    
    //Mat element = getStructuringElement( MORPH_RECT,
    //                                   Size( 2*erosion_size + 1, 2*erosion_size+1 ),
    //                                   Point( erosion_size, erosion_size ) );
    
    //erode(vMap,vMap,element);
    
*/
/*
    //Basic Filter for Conotours
    Mat basicContours(rows,imgCols, CV_32FC1, Scalar(0));
    vector<vector<Point> > contours;
    Scalar color = {255,0,0};
    vector<Vec4i> hierarchy;
    vMap.convertTo(vMap,CV_8UC1);
    basicContours = vMap;
    findContours(basicContours,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0,0));
    drawContours(basicContours, contours, 0, color, -1);
    vector<Vec4i> lines;
    HoughLinesP( basicContours, lines, 1, CV_PI/180, 80, 30, 10 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
        line( basicContours, Point(lines[i][0], lines[i][1]),
        Point( lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
    }
    imshow("Basic",basicContours);
*/
/*
//Create file stream
    ofstream depth;
    ofstream v;
    v.open("vMap.txt");
    depth.open("outputDepth1.txt");

    patchNaNs(depthIn, 0.0);
    
    for(int y = 0; i < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            v<<depthIn.at<float>(y,x)<<" ";
        }
        v<<endl<<endl;
    }
   
    for(int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            depth<<depthIn.at<double>(i,j)<<" ";
        }
        depth<<endl<<endl;
    }
    
    
    depth.close();
    v.close();

        
    //Turn super small values to zeros and divide all values by 1000
    for(int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if(depthIn.at<double>(j,i) > 0.09)
            {
                depthIn.at<double>(j,i) = 9;
            }
            //else if(depthIn.at<double>(i,j)<.00700)
              //  depthIn.at<double>(i,j) = 0.7;
            
            else if(depthIn.at<double>(j,i) < 0.09)
            {
                depthIn.at<double>(j,i) = depthIn.at<double>(j,i)*100;
            }
        }
    }
    

    patchNaNs(depthIn, 0.0);
    depthIn = max(depthIn, 0.7);
    depthIn = min(depthIn, 10.0);
    

    //Write and show normalized vDisparity Mat
    Mat normalized(720,imgCols,CV_32FC1);
    normalize(vMap, normalized, 255, 0.0, NORM_MINMAX);
    ofstream normal;
    normal.open("Noramlized.txt");
    writeDepths(normalized, normal);

    //Theoretically multiply all values by 20
    normalized.convertTo(normalized,-1,20,0);

        //Write and show raw vDisparity Mat
    //ofstream vMapS;
    //vMapS.open("vMap.txt");
    //imshow("vMap", vMap);
    //writeDepths(vMap,vMapS);

    //Write and show raw vDisparity Mat
    ofstream vMapS;
    vMapS.open("vMap.txt");
    imshow("vMap", vMap);
    writeDepths(vMap,vMapS);
*/