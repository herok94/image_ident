#ifndef BASICOPENCV_H
#define BASICOPENCV_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "cv.h"
#include "highgui.h"

#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include <math.h>
#endif


#define w (500)

using namespace std;
using namespace cv;



class BasicOpencv
{
public:
    BasicOpencv();

    void MatImageshow(char *path);
    void IplImageShow(char *path);
    void CannyCheck(char *path);
    void OutLineShow(void);
    void ZoomInOutShow(char *path);
    void VideoCaptuerShow(char *path);

private:

};

#endif // BASICOPENCV_H
