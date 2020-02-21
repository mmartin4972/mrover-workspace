#include "perception.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>

using namespace cv;
using namespace std;

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
            depth<<depthIn.at<int>(y,x)<<" ";
        }
        depth<<endl<<endl;
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
    float* temp = calculateMultiplier(9.3, 0.02);
    float imgCols = *temp;
    float multiplier = *(temp+1);
    cerr<<"Image Columns: "<<imgCols<<endl;
    cerr<<"Multiplier: "<<multiplier<<endl;

    //Create empty vMap
    Mat vMap(rows, imgCols, CV_32SC1, Scalar(0));
    
    //Traverse through image and increment values in vMap that correspond with depth
    for(int y = 0; y < rows; ++y)
    {
    
        for(int x = 0; x < cols; ++x)
        {
            int column = int((depthIn.at<float>(y, x)-0.7)*multiplier);
            vMap.at<int>(y, column) = vMap.at<int>(y, column)+1;
        }

    }

    //Write out rows and cols for debugging
    cout<<rows<<endl;
    cout<<cols<<endl;

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


    //Canny Filter for contours
    int edgeThresh = 1;
    int lowThreshold = 10;
    int const max_lowThreshold = 100;
    int ratio = 3;
    int kernel_size = 3;
    char* window_name = "Canny";
    Mat cannyContours(rows,imgCols, CV_8UC1, Scalar(0));
    Mat dst(rows,imgCols, CV_8UC1, Scalar(0));
    
    //Image Erosion Techniques

    //blur( vMap, vMap, Size(3,3) );    
    
    //Morphological Erosion
    /*
    int erosion_elem = 0.3;
    int erosion_size = 0.3;
    int dilation_elem = 15;
    int dilation_size = 15;
    int const max_elem = 2;
    int const max_kernel_size = 21;
    
    Mat element = getStructuringElement( MORPH_RECT,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
    
    erode(vMap,vMap,element);
    

    Canny(vMap, cannyContours, lowThreshold, lowThreshold*ratio, kernel_size );
   vector<Vec4i> lines;
    vMap.copyTo(dst,cannyContours);
    HoughLinesP( dst, lines, 1, CV_PI/180, 80, 30, 10 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
        line( dst, Point(lines[i][0], lines[i][1]),
        Point( lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
    }
    imshow(window_name, dst);
    */
    return vMap;

}





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