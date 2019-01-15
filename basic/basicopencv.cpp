#include "basicopencv.h"

BasicOpencv::BasicOpencv()
{

}

void BasicOpencv::MatImageshow(char *path)
{
    Mat image;
    image=imread(path,0);
    namedWindow("show",CV_WINDOW_AUTOSIZE);
    imshow("show",image);
   // waitKey(0);
}

void BasicOpencv::IplImageShow(char *path)
{
    IplImage *pImg;

    pImg = cvLoadImage(path,1);
    CvSize size = cvGetSize(pImg);
    if(pImg != NULL )
    {
        IplImage* pImg2 = cvCreateImage(size,pImg->depth,pImg->nChannels);
        cout <<size.height << size.width<< pImg->imageSize<< "pImg->depth=" <<pImg->depth << "pImg->nChannels="<<pImg->nChannels;
        cvCopy(pImg, pImg2, NULL);
        cvSaveImage("../copy.jpg", pImg2);
        cvNamedWindow( "Image", CV_WINDOW_AUTOSIZE);
        cvShowImage( "Image", pImg );

        cvWaitKey(0); //等待按键

        cvDestroyWindow("Image");
        cvReleaseImage(&pImg);
        cvReleaseImage(&pImg2);
    }
}

void BasicOpencv::CannyCheck(char *path)
{
    IplImage* pImg = NULL;
    IplImage* pCannyImg = NULL;

    pImg = cvLoadImage(path, 0);
    if(pImg!=NULL){
        pCannyImg = cvCreateImage(cvGetSize(pImg),pImg->depth,pImg->nChannels);
        cvCanny(pImg,pCannyImg,50,150,3); //查看这个函数

        cvNamedWindow("src", 1);
        cvNamedWindow("canny",1);

        cvShowImage("src",pImg);
        cvShowImage("canny",pCannyImg);
        cvWaitKey(0); //等待按键

        //销毁窗口
        cvDestroyWindow("src");
        cvDestroyWindow("canny");

        //释放图像
        cvReleaseImage(&pImg);
        cvReleaseImage(&pCannyImg);
    }
}

static int levels = 3;
static  CvSeq* contours = 0;

void on_trackbar(int pos)
{
    IplImage* cnt_img = cvCreateImage(cvSize(w,w),8,3);
    CvSeq  * _contours = contours;
    int _levels = levels - 3;

    if( _levels <= 0 ) // get to the nearest face to make it look more funny

    _contours = _contours->h_next->h_next->h_next;
    cvZero(cnt_img);
    cvDrawContours(cnt_img,_contours,CV_RGB(255,0,0),CV_RGB(0,255,0),_levels, 3, CV_AA, cvPoint(0,0) );
    cvShowImage("contours",cnt_img );
    cvReleaseImage(&cnt_img);
}

void BasicOpencv::OutLineShow()
{
    int i, j;
    CvMemStorage* storage = cvCreateMemStorage(0);
    IplImage* img = cvCreateImage(cvSize(w,w),8,1);
    cvZero(img);

    for( i=0; i < 6; i++ )
    {
        int dx = (i%2)*250 - 30;
        int dy = (i/2)*150;
        CvScalar white = cvRealScalar(255);
        CvScalar black = cvRealScalar(0);

        if( i == 0 )
        {
            for( j = 0; j <=10; j++ )
            {
                double angle = (j+5)*CV_PI/21;
                cvLine(img, cvPoint(cvRound(dx+100+j*10-80*cos(angle)),
                cvRound(dy+100-90*sin(angle))),
                cvPoint(cvRound(dx+100+j*10-30*cos(angle)),
                cvRound(dy+100-30*sin(angle))), white, 1, 8, 0);
            }
        }

        cvEllipse(img, cvPoint(dx+150, dy+100), cvSize(100,70), 0, 0, 360, white, -1, 8, 0 );
        cvEllipse(img, cvPoint(dx+115, dy+70), cvSize(30,20), 0, 0, 360, black, -1, 8, 0 );
        cvEllipse(img, cvPoint(dx+185, dy+70), cvSize(30,20), 0, 0, 360, black, -1, 8, 0 );
        cvEllipse(img, cvPoint(dx+115, dy+70), cvSize(15,15), 0, 0, 360, white, -1, 8, 0 );
        cvEllipse(img, cvPoint(dx+185, dy+70), cvSize(15,15), 0, 0, 360, white, -1, 8, 0 );
        cvEllipse(img, cvPoint(dx+115, dy+70), cvSize(5,5), 0, 0, 360, black, -1, 8, 0 );
        cvEllipse(img, cvPoint(dx+185, dy+70), cvSize(5,5), 0, 0, 360, black, -1, 8, 0 );
        cvEllipse(img, cvPoint(dx+150, dy+100), cvSize(10,5), 0, 0, 360, black, -1, 8, 0 );
        cvEllipse(img, cvPoint(dx+150, dy+150), cvSize(40,10), 0, 0, 360, black, -1, 8, 0 );
        cvEllipse(img, cvPoint(dx+27, dy+100), cvSize(20,35), 0, 0, 360, white, -1, 8, 0 );
        cvEllipse(img, cvPoint(dx+273, dy+100), cvSize(20,35), 0, 0, 360, white, -1, 8, 0 );
    }

    cvNamedWindow( "image", 1 );
    cvShowImage( "image", img );

    cvFindContours(img, storage, &contours, sizeof(CvContour),
    CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );

    // comment this out if you do not want approximation
    contours = cvApproxPoly( contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, 3, 1 );
    cvNamedWindow( "contours", 1 );
    cvCreateTrackbar("levels+3", "contours", &levels, 7, on_trackbar);

    on_trackbar(0);
    cvWaitKey(0);
    cvReleaseMemStorage( &storage );
    cvReleaseImage( &img );
}

void BasicOpencv::ZoomInOutShow(char *path)
{
    IplImage *src = 0;
    IplImage *dst = 0;

    src = cvLoadImage(path, CV_LOAD_IMAGE_GRAYSCALE);

    if (src != 0)
    {
        int delta = 1;
        int angle = 0;
        int opt = 0; // 1： 旋转加缩放
        // 0: 仅仅旋转

        double factor;
        dst = cvCloneImage (src);
        cvNamedWindow ("src", 1);
        cvShowImage ("src", src);

        for (;;)
        {
            float m[6];

            // Matrix m looks like:
            //
            // [ m0 m1 m2 ] ===> [ A11 A12 b1 ]
            // [ m3 m4 m5 ] [ A21 A22 b2 ]
            //
            CvMat M = cvMat(2, 3, CV_32F, m);
            int w1 = src->width;
            int h = src->height;

            if (opt) // 旋转加缩放
                factor = (cos (angle * CV_PI / 180.) + 1.0) * 2;
            else // 仅仅旋转
                factor = 1;
            m[0] = (float) (factor * cos (-angle * 2 * CV_PI / 180.));
            m[1] = (float) (factor * sin (-angle * 2 * CV_PI / 180.));
            m[3] = -m[1];
            m[4] = m[0];

            // 将旋转中心移至图像中间
            m[2] = w1 * 0.5f;
            m[5] = h * 0.5f;

            // dst(x,y) = A * src(x,y) + b
            cvZero (dst);
            cvGetQuadrangleSubPix (src, dst, &M);
            cvNamedWindow ("dst", 1);
            cvShowImage ("dst", dst);

            if (cvWaitKey(1) == 27) //ESC
                break;
            angle = (int) (angle + delta) % 360;
        } // for-loop
    }
}

void BasicOpencv::VideoCaptuerShow(char *path)
{
    IplImage* pFrame = NULL;
    IplImage* pFrImg = NULL;

    IplImage* pBkImg = NULL;

    CvMat* pFrameMat = NULL;
    CvMat* pFrMat = NULL;
    CvMat* pBkMat = NULL;
    CvCapture* pCapture = NULL;
    int nFrmNum = 0;

    cvNamedWindow("video", 1);
    cvNamedWindow("background",1);
    cvNamedWindow("foreground",1);

    //使窗口有序排列
    cvMoveWindow("video", 30, 0);
    cvMoveWindow("background", 360, 0);
    cvMoveWindow("foreground", 690, 0);

    string avi_path = "F:/OpenCV/test.mp4";

    cout << "avi_path.c_str()=" << avi_path.c_str()<<endl;

    if(!(pCapture = cvCaptureFromFile(avi_path.c_str())))
    {
        fprintf(stderr, "Can not open video file %s\n", path);
        return ;
    }

    while(pFrame = cvQueryFrame( pCapture ))
    {
    nFrmNum++;
    //如果是第一帧，需要申请内存，并初始化
    if(nFrmNum == 1)
    {
    pBkImg = cvCreateImage(cvSize(pFrame->width, pFrame->height), IPL_DEPTH_8U,1);
    pFrImg = cvCreateImage(cvSize(pFrame->width, pFrame->height), IPL_DEPTH_8U,1);
    pBkMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);
    pFrMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);
    pFrameMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);

    //转化成单通道图像再处理
    cvCvtColor(pFrame, pBkImg, CV_BGR2GRAY);
    cvCvtColor(pFrame, pFrImg, CV_BGR2GRAY);
    cvConvert(pFrImg, pFrameMat);
    cvConvert(pFrImg, pFrMat);
    cvConvert(pFrImg, pBkMat);
    }
    else
    {
    cvCvtColor(pFrame, pFrImg, CV_BGR2GRAY);
    cvConvert(pFrImg, pFrameMat);
    //高斯滤波先，以平滑图像
    //cvSmooth(pFrameMat, pFrameMat, CV_GAUSSIAN, 3, 0, 0);
    //当前帧跟背景图相减
    cvAbsDiff(pFrameMat, pBkMat, pFrMat);



    //二值化前景图
    cvThreshold(pFrMat, pFrImg, 60, 255.0, CV_THRESH_BINARY);
    //进行形态学滤波，去掉噪音
    //cvErode(pFrImg, pFrImg, 0, 1);
    //cvDilate(pFrImg, pFrImg, 0, 1);

    //更新背景
    cvRunningAvg(pFrameMat, pBkMat, 0.003, 0);
    //将背景转化为图像格式，用以显示
    cvConvert(pBkMat, pBkImg);

    //显示图像
    cvShowImage("video", pFrame);
    cvShowImage("background", pBkImg);
    cvShowImage("foreground", pFrImg);

    //如果有按键事件，则跳出循环
     //此等待也为cvShowImage函数提供时间完成显示
     //等待时间可以根据CPU速度调整
     if( cvWaitKey(2) >= 0 )
     break;
     }
     }

     //销毁窗口
     cvDestroyWindow("video");
     cvDestroyWindow("background");
     cvDestroyWindow("foreground");

     //释放图像和矩阵
     cvReleaseImage(&pFrImg);
     cvReleaseImage(&pBkImg);
     cvReleaseMat(&pFrameMat);
     cvReleaseMat(&pFrMat);
     cvReleaseMat(&pBkMat);
     cvReleaseCapture(&pCapture);
}


