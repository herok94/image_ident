/*
 * author :  Shucheng Yin
 * stu_id :  SA14011016
 * e-mail :  ysc6688@mail.ustc.edu.cn
 *  date  :  2014-11-22
 *  file  :  Image.h
 *  dscp  :  The declaration of class Image
 */

#include <fstream>
#include <dirent.h>
#include <sys/types.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <cstdio>
#include <iostream>

#include "Image.h"


using namespace cv;
using namespace cv::ml;
using namespace std;

#define Herok(pcFormat, ...)   printf("%s %s %d " pcFormat,__FILE__,__FUNCTION__,__LINE__, ##__VA_ARGS__)


#define CHECK_PATH "../check.jpg"

#define SinglePath   "../div/244160.jpg"

#define CHAR_SIZE 6


const char res_dir[] = "../res/";
const char xun_dir[] = "../xun/";
const char err_dir[] = "../error/";
const char out_file[] = "../file/recognition.data";
const char xml_file[] = "../file/train_out.xml";

const int OFFSET = 7;
const int VECTOR_SIZE = 16;

bool read_num_class_data( const string& filename, int var_count,
                     Mat* _data, Mat* _responses )
{
    const int M = 1024;
    char buf[M+2];

    Mat el_ptr(1, var_count, CV_32F);
    int i;
    vector<int> responses;

    _data->release();
    _responses->release();

    FILE* f = fopen(filename.c_str(), "rt" );
    if(!f)
    {
        cout << "Could not read the database " << filename << endl;
        return false;
    }

    for(;;)
    {
        char* ptr;
        if( !fgets( buf, M, f ) || !strchr( buf, ',' ) )
            break;
        responses.push_back((int)buf[0]);
        ptr = buf+2;
        for( i = 0; i < var_count; i++ )
        {
            int n = 0;
            sscanf( ptr, "%f%n", &el_ptr.at<float>(i), &n );
            ptr += n + 1;
        }
        if( i < var_count )
            break;
        _data->push_back(el_ptr);
    }
    fclose(f);
    Mat(responses).copyTo(*_responses);

    cout << "The database " << filename << " is loaded.\n";

    return true;
}

bool build_svm_classifier( const string& data_filename,
                      const string& filename_to_save)
{
    int i;
    Mat data;
    Mat responses;
    bool ok = read_num_class_data( data_filename, VECTOR_SIZE, &data, &responses );
    if( !ok )
        return ok;
    int nsamples_all = data.rows;
    int ntrain_samples = (int)(nsamples_all*0.8);
    cout << "nsamples_all"<< nsamples_all<< "   "<< nsamples_all-ntrain_samples <<endl;
    Mat train_data = data.rowRange(0,ntrain_samples);
    Mat test_data  = data.rowRange(ntrain_samples,nsamples_all);
    Mat train_response = responses.rowRange(0,ntrain_samples);
    Mat test_response = responses.rowRange(ntrain_samples,nsamples_all);
    cout << "Training the classifier ...\n";

    // Set up SVM's parameters
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    //! [train]
    //! ml::SampleTypes

    svm->train(train_data, ROW_SAMPLE, train_response);
    svm->save(filename_to_save.c_str());
     // Train the SVM
    cout << "Begin to test the classifier ..." << endl;
    int right = 0;
	int cnt=0;
    for(i=0;i<nsamples_all - ntrain_samples;i++)
    {
        Mat sample = test_data.row(i);
		cnt++;
        if(svm->predict(sample)  == test_response.at<int>(i) )
            right++;
	else
		cout << "predict is fail!!" << (char)test_response.at<int>(i) << "   cnt=" << (cnt+ntrain_samples) << " num "<<(int)(cnt+ntrain_samples)/6 << " % "\
		<< (cnt+ntrain_samples)%6<<endl;
    }
    cout << "The correct rate of the " << nsamples_all - ntrain_samples << " test cases is: " << right*100.0 / (nsamples_all-ntrain_samples)  << "%"<< endl;

    return true;
}

int predict(const string& sample)
{
    int i; 
    char buf[80],*ptr;

    Ptr<SVM> svm = SVM::load(xml_file);

    Mat sample_mat = Mat(1,VECTOR_SIZE,CV_32F);
    strcpy(buf,sample.c_str());
    ptr = buf;

    for (i = 0; i < VECTOR_SIZE; ++i)
    {
        int n = 0;
        sscanf( ptr, "%f%n", &sample_mat.at<float>(i), &n );
        ptr += n + 1;
    }

	int val = svm->predict(sample_mat);

	if(val<0)
		cout << "svm->predict is failed!!!" <<endl;

    return val;
}

int main()
{
    int i,divide_fail = 0;
	char full_name[256];
	fstream out;

	DIR *pDir;
	struct dirent *pFile;
	std::vector<string> name;
	std::vector<int> count,result;
	string test_vector;
	int cnt=0;
	int AllMap=0;
	cout << "input work mode:";
    int val =1 ;

    scanf("%d",&val);
	if(val==4)  /* test and check recognition  */
	{
		char fun_name[20];
		strcpy(fun_name,SinglePath);
		Image img;
		img.LoadImg(fun_name);
		img.toGray();
		Herok("\n");
       	img.show();
		Herok("\n");
		img.Binarization();
		img.show();
		img.TabPiexl();
		img.show();
		img.NaiveRemoveNoise(1);
		img.show();
		img.ContoursRemoveNoise(5.0f);
		img.show();

		std::vector<string> data;
		img.FloodFillDivide(data,10,fun_name+7,1);

		if (data.size() != CHAR_SIZE)
		{
			name.push_back(string(full_name+7));
			count.push_back(data.size());
			divide_fail++;
		}
		else
		{
			AllMap++;
            for (i = 0; i < (int)data.size(); ++i)
                out << data.at(i) <<endl;
		}

		out.close();
		std::cout << "all count=" << AllMap<<"   "<< divide_fail << " cases failed when divided!" << std::endl;
		}	
	else if (val==1)
	{
		remove(out_file);
		out.open(out_file,ios::out | ios::app);
		if ((pDir=opendir(res_dir))==NULL)
		{
			cerr << "Can't open dir " << res_dir << " !" << endl;
			exit(0);
		}    
		while( (pFile=readdir(pDir)) != NULL )
		{
			Image img;
			char *pName = pFile->d_name;

			if (!strcmp(pName,".") || !strcmp(pName,".."))  continue;

			strcpy(full_name,"../res/");
			strcat(full_name,pName);
			img.LoadImg(full_name);
			img.toGray();
			img.Binarization();
           		img.TabPiexl();
			img.NaiveRemoveNoise(1);
			img.ContoursRemoveNoise(5.0f);

			std::vector<string> data;
            img.FloodFillDivide(data,10,full_name+OFFSET,0);
			cout << "*************cnt=" << ++cnt << "**********" <<endl;

			if (data.size() != CHAR_SIZE)
			{
				name.push_back(string(full_name+7));
				count.push_back(data.size());
				divide_fail++;
			}
			else
			{
				AllMap++;
                for (i = 0; i < (int)data.size(); ++i)
				out << data.at(i) <<endl;
			}

			delete pName;
		}

		out.close();
		char src[50],dst[50];
		std::cout << "all count=" << AllMap<<"   "<< divide_fail << " cases failed when divided!" << std::endl;
		for (i = 0; i < (int)name.size(); ++i)
		{
			std::cout << name.at(i) << "   "  << count.at(i) << std::endl;
			sprintf(src,"%s%s",res_dir,name.at(i).data());
			sprintf(dst,"%s%s",err_dir,name.at(i).data());
			//cout << "src=" << src<< "  dst=" << dst<<endl;
			rename(src,dst);  /* mv into ../error dir */
		}
	}
	else if(val==2)  /* train mode */
	{
		build_svm_classifier(out_file,xml_file);
	}
	else if(val==3)  /* modify file name */
	{
		if ((pDir=opendir(xun_dir))==NULL)
		{
			cerr << "Can't open dir " << xun_dir << " !" << endl;
			exit(0);
		}
		while( (pFile=readdir(pDir)) != NULL )
		{
			char *pName = pFile->d_name;

			if (!strcmp(pName,".") || !strcmp(pName,".."))  continue;

			strcpy(full_name,"../xun/");
			strcat(full_name,pName);

			Image img;
			img.LoadImg(full_name);
			img.toGray();
			img.Binarization();
			img.TabPiexl();
			img.NaiveRemoveNoise(1);
			img.ContoursRemoveNoise(5.0f);
			std::vector<string> data;
			data.clear();
			img.FloodFillDivide(data,10,full_name,0);

			// std::cout << "***********************************" << std::endl;
			// std::cout << "The charactor vectors are as follows:" << std::endl;
			result.clear();
			test_vector="";
			for (i = 0; i < (int)data.size(); ++i)
			{
				test_vector = data.at(i).substr(2,data.at(i).size());
				// std::cout << test_vector << std::endl;
				result.push_back(predict(test_vector));
			}
			//std::cout << "***********************************" << std::endl;
			//cout << "The final charactors recognized is :" << endl;
			char tmp[5];
			char tmp1[20];
			sprintf(tmp1,"%s","../xun/");
			for (i = 0; i < (int)data.size(); ++i)
			{
				int ch = result.at(i);
				sprintf(tmp,"%d",(ch-48-48+'0'));
				strcat(tmp1,tmp);
				if (ch <= 57)
					cout << (char)(ch-48+'0') << " ";
				else
					cout << (char)(ch-65+'A') << " ";
			}
			cout << endl;
			sprintf(tmp,"%s",".jpg");
			strcat(tmp1,tmp);
			rename(full_name,tmp1);
			cout << "old_name = " << full_name << "  new_name =" <<tmp1<<endl;
		}
	}
	else { /* recognition image */
		Image img;
		img.LoadImg(CHECK_PATH);
		img.toGray();
		img.Binarization();
		img.TabPiexl();
		img.NaiveRemoveNoise(1);
		img.ContoursRemoveNoise(5.0f);
		std::vector<string> data;
		data.clear();
       	 	img.FloodFillDivide(data,10,(char *)CHECK_PATH,0);

		std::cout << "***********************************" << std::endl;
		std::cout << "The charactor vectors are as follows:" << std::endl;
		result.clear();
		test_vector="";
		for (i = 0; i < (int)data.size(); ++i)
		{
			test_vector = data.at(i).substr(2,data.at(i).size());
			// std::cout << test_vector << std::endl;
			result.push_back(predict(test_vector));
		}
		std::cout << "***********************************" << std::endl;
		cout << "The final charactors recognized is :" << endl;

		for (i = 0; i < (int)data.size(); ++i)
		{
			int ch = result.at(i);
			if (ch <= 57)
				cout << (char)(ch-48+'0') << " ";
			else
				cout << (char)(ch-65+'A') << " ";
		}
		cout << endl;

	}
     
    return 0;
}
