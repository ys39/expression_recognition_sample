#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <Python.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/threading.h>

using namespace std;
using namespace cv;
using namespace dlib;
using namespace boost::property_tree;

/* 変数宣言
*****************************************/
bool protonect_shutdown = false;
const float RESOLUTION = 0.3;   // 解像度576×316
const int IMSIZE = 256;         // 画像サイズ256×256
const int CASCADE = 1;          // 0 -> haarlike,1 -> lbp
const int METHODMODE = 0;       // 0 -> 従来手法,1 -> 新手法

// 分類器
string HAARLIKE_XML = "//home//ubuntu//libfreenect2//examples//protonect//haarcascade_frontalface_default.xml";
string LBP_XML = "//home//ubuntu//libfreenect2//examples//protonect//lbpcascade_frontalface.xml";

// 顔特徴点
const string LANDMARK_DAT       = "//home//ubuntu//libfreenect2//examples//protonect//shape_predictor_68_face_landmarks.dat";

// 顔パーツ保存ディレクトリ
const string DOCUMENT_ROOT          = "//home//ubuntu//Documents//image//CURRENT//FeatureVersion";
const string DOCUMENT_PROPOSED      = "//PROPOSED";
const string DOCUMENT_CONVENTIONAL  = "//CONVENTIONAL";
const string DOCUMENT_FACE          = "//face//kinect_face";
const string DOCUMENT_RIGHTEYE      = "//righteye//kinect_righteye";
const string DOCUMENT_LEFTEYE       = "//lefteye//kinect_lefteye";
const string DOCUMENT_MOUTH         = "//mouth//kinect_mouth";

int init_x=0;
int init_y=0;
int side_x=0;
int side_y=0;
int min_y = 0;

// 顔特徴点の座標
class p{
public:
    int x;
    int y;
    p(int a=0){
        x = y = a;
    }
};
p INNER_RIGHT_EYE;
p OUTER_RIGHT_EYE;
p INNER_LEFT_EYE;
p OUTER_LEFT_EYE;
p CENTEREYE;
p RIGHTMOUTH;
p LEFTMOUTH;
p TOPMOUTH;

/* 初回実行
*****************************************/
libfreenect2::Freenect2 freenect2;libfreenect2::Freenect2Device *dev = freenect2.openDefaultDevice();
frontal_face_detector detector = get_frontal_face_detector();
shape_predictor pose_model;

/* Chainerを用いた顔認識 (Python API)
*****************************************/
int pyWrapper(int count)
{

	FILE * exp_file;
	PyObject *module;
	PyObject *global_dict, *expression, *result, *pArgs;
	string ret = "";

    const char *exp = "//home//ubuntu//Documents//source//python//chainer//examples//imagenet//inspection_conventional_dev.py";
    const char* func_name = "inspection";

	exp_file = fopen(exp, "r");
	PyRun_SimpleFile(exp_file, exp);
	module = PyImport_ImportModule("__main__");
	global_dict = PyModule_GetDict(module);
	expression = PyDict_GetItemString(global_dict, func_name);

    ostringstream oss;
    oss << DOCUMENT_ROOT << DOCUMENT_CONVENTIONAL << DOCUMENT_FACE << count << ".png";
    string str = oss.str();

	pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs,0,PyString_FromString(str.c_str()));
    //PyTuple_SetItem(pArgs,1,PyString_FromString(a.c_str()));
	//result = PyObject_CallObject(expression, NULL);
	result = PyObject_CallObject(expression, pArgs);

    if(PyString_Check(result))
    {
        ret = PyString_AsString(result);
    }

    if(ret == "success")
    {
        return 1;
    }
    else
    {
        return 0;
    }

}

int pyWrapper_proposed(int count)
{

	FILE *exp_file1, *exp_file2, *exp_file3, *exp_file4;
	PyObject *module1, *module2, *module3, *module4;
	PyObject *global_dict1, *global_dict2, *global_dict3, *global_dict4;
	PyObject *expression1, *expression2, *expression3, *expression4;
	PyObject *result1, *result2, *result3, *result4;
	PyObject *pArgs1, *pArgs2, *pArgs3, *pArgs4;
	string ret = "";

    const char *exp1 = "//home//ubuntu//Documents//source//python//chainer//examples//imagenet//inspection_proposed_face_dev.py";
    const char *exp2 = "//home//ubuntu//Documents//source//python//chainer//examples//imagenet//inspection_proposed_righteye_dev.py";
    const char *exp3 = "//home//ubuntu//Documents//source//python//chainer//examples//imagenet//inspection_proposed_lefteye_dev.py";
    const char *exp4 = "//home//ubuntu//Documents//source//python//chainer//examples//imagenet//inspection_proposed_mouth_dev.py";

    const char *func_name = "inspection";

	exp_file1 = fopen(exp1, "r");
	exp_file2 = fopen(exp2, "r");
	exp_file3 = fopen(exp3, "r");
	exp_file4 = fopen(exp4, "r");

	PyRun_SimpleFile(exp_file1, exp1);
	PyRun_SimpleFile(exp_file2, exp2);
	PyRun_SimpleFile(exp_file3, exp3);
	PyRun_SimpleFile(exp_file4, exp4);

	module1 = PyImport_ImportModule("__main__");
	module2 = PyImport_ImportModule("__main__");
	module3 = PyImport_ImportModule("__main__");
	module4 = PyImport_ImportModule("__main__");

	global_dict1 = PyModule_GetDict(module1);
	global_dict2 = PyModule_GetDict(module2);
	global_dict3 = PyModule_GetDict(module3);
	global_dict4 = PyModule_GetDict(module4);

	expression1 = PyDict_GetItemString(global_dict1, func_name);
	expression2 = PyDict_GetItemString(global_dict2, func_name);
	expression3 = PyDict_GetItemString(global_dict3, func_name);
	expression4 = PyDict_GetItemString(global_dict4, func_name);

    ostringstream oss;
    oss << DOCUMENT_ROOT << DOCUMENT_PROPOSED << DOCUMENT_FACE << count << ".png";
    string str = oss.str();

    ostringstream oss2;
    oss2 << DOCUMENT_ROOT << DOCUMENT_PROPOSED << DOCUMENT_RIGHTEYE << count << ".png";
    string str2 = oss2.str();

    ostringstream oss3;
    oss3 << DOCUMENT_ROOT << DOCUMENT_PROPOSED << DOCUMENT_LEFTEYE << count << ".png";
    string str3 = oss3.str();

    ostringstream oss4;
    oss4 << DOCUMENT_ROOT << DOCUMENT_PROPOSED << DOCUMENT_MOUTH << count << ".png";
    string str4 = oss4.str();

	pArgs1 = PyTuple_New(1);
	pArgs2 = PyTuple_New(1);
	pArgs3 = PyTuple_New(1);
	pArgs4 = PyTuple_New(1);

    PyTuple_SetItem(pArgs1,0,PyString_FromString(str.c_str()));
    PyTuple_SetItem(pArgs2,0,PyString_FromString(str2.c_str()));
    PyTuple_SetItem(pArgs3,0,PyString_FromString(str3.c_str()));
    PyTuple_SetItem(pArgs4,0,PyString_FromString(str4.c_str()));

	result1 = PyObject_CallObject(expression1, pArgs1);
    result2 = PyObject_CallObject(expression2, pArgs2);
    result3 = PyObject_CallObject(expression3, pArgs3);
    result4 = PyObject_CallObject(expression4, pArgs4);

    // すべて関数が実行されたのち結果を返す

    if(PyString_Check(result4))
    {
        ret = PyString_AsString(result4);
    }

    if(ret == "success")
    {
        return 1;
    }
    else
    {
        return 0;
    }

}

int pyWrapper_proposed_all(int count)
{

	FILE *exp_file1;
	PyObject *module1;
	PyObject *global_dict1;
	PyObject *expression1;
	PyObject *result1;
	PyObject *pArgs1;
	string ret = "";

    const char *exp1 = "//home//ubuntu//Documents//source//python//chainer//examples//imagenet//inspection_proposed_all_dev.py";
    const char *func_name = "inspection";

	exp_file1 = fopen(exp1, "r");
	PyRun_SimpleFile(exp_file1, exp1);
	module1 = PyImport_ImportModule("__main__");
	global_dict1 = PyModule_GetDict(module1);
	expression1 = PyDict_GetItemString(global_dict1, func_name);

    ostringstream oss;
    oss << DOCUMENT_ROOT << DOCUMENT_PROPOSED << DOCUMENT_FACE << count << ".png";
    string str1 = oss.str();

    ostringstream oss2;
    oss2 << DOCUMENT_ROOT << DOCUMENT_PROPOSED << DOCUMENT_RIGHTEYE << count << ".png";
    string str2 = oss2.str();

    ostringstream oss3;
    oss3 << DOCUMENT_ROOT << DOCUMENT_PROPOSED << DOCUMENT_LEFTEYE << count << ".png";
    string str3 = oss3.str();

    ostringstream oss4;
    oss4 << DOCUMENT_ROOT << DOCUMENT_PROPOSED << DOCUMENT_MOUTH << count << ".png";
    string str4 = oss4.str();

	pArgs1 = PyTuple_New(4);
    PyTuple_SetItem(pArgs1,0,PyString_FromString(str1.c_str()));
    PyTuple_SetItem(pArgs1,1,PyString_FromString(str2.c_str()));
    PyTuple_SetItem(pArgs1,2,PyString_FromString(str3.c_str()));
    PyTuple_SetItem(pArgs1,3,PyString_FromString(str4.c_str()));

	result1 = PyObject_CallObject(expression1, pArgs1);

    // 表情の結果が返る

    if(PyString_Check(result1))
    {
        ret = PyString_AsString(result1);
    }

    cout << " 表情は :" << ret << endl;

/*
    if(ret == "success")
    {
        return 1;
    }
    else
    {
        return 0;
    }
*/

    return 0;

}


/* OpenCVを用いた顔検出
*****************************************/
Mat detectFaceInImage(Mat &image,string &cascade_file)
{
	CascadeClassifier cascade;
	cascade.load(cascade_file);
	std::vector<Rect> faces;
	cascade.detectMultiScale(image, faces, 1.1,3,0,Size(20,20));

	if(faces.size() != 0)
    {
	    static int count = 1;
	    int res = 0;

	    //顔を切り出す→256×256にリサイズ→3チャネルのグレースケールに変換
        Mat trimming_image(image, Rect(faces[0].x, faces[0].y,faces[0].width,faces[0].height));
        resize(trimming_image, trimming_image, cv::Size(), (float)IMSIZE/trimming_image.cols,(float)IMSIZE/trimming_image.rows);
        cvtColor(trimming_image,trimming_image,CV_RGB2GRAY);
        cvtColor(trimming_image,trimming_image,CV_GRAY2BGR);

        // 従来手法では画像全体を保存する
        if(METHODMODE == 0)
        {
            //保存
            ostringstream oss;
            oss << DOCUMENT_ROOT << DOCUMENT_CONVENTIONAL << DOCUMENT_FACE << count << ".png";
            string str = oss.str();
            imwrite(str,trimming_image);

            //表示
            imshow("trim",trimming_image);
            res = pyWrapper(count);//chainer実行
        }

        // 新手法では分割画像を保存する
        else if(METHODMODE == 1)
        {
            ostringstream oss;
            oss << DOCUMENT_ROOT << DOCUMENT_PROPOSED << DOCUMENT_FACE << count << ".png";
            string str = oss.str();
            imwrite(str,trimming_image);

            imshow("face",trimming_image);

            array2d<rgb_pixel> img;
            load_image(img, str);
            std::vector<dlib::rectangle> dets = detector(img);
            std::vector<full_object_detection> shapes;

            // detsが空ならばtrue, 空でなければfalse
            if(!dets.empty())
            {

                full_object_detection shape = pose_model(img, dets[0]);

                shapes.push_back(shape);
                dlib::vector<double,2> darr;
                dlib::vector<double,2> darr2;

                // 右目尻
                darr = shape.part(36);
                std::vector<double> sarr1(darr.begin(), darr.end());
                INNER_RIGHT_EYE.x = sarr1[0];
                INNER_RIGHT_EYE.y = sarr1[1];

                // 右目頭
                darr = shape.part(39);
                std::vector<double> sarr2(darr.begin(), darr.end());
                OUTER_RIGHT_EYE.x = sarr2[0];
                OUTER_RIGHT_EYE.y = sarr2[1];

                // 左目尻
                darr = shape.part(42);
                std::vector<double> sarr3(darr.begin(), darr.end());
                INNER_LEFT_EYE.x = sarr3[0];
                INNER_LEFT_EYE.y = sarr3[1];

                // 左目頭
                darr = shape.part(45);
                std::vector<double> sarr4(darr.begin(), darr.end());
                OUTER_LEFT_EYE.x = sarr4[0];
                OUTER_LEFT_EYE.y = sarr4[1];

                // 口右端
                darr = shape.part(60);
                std::vector<double> sarr5(darr.begin(), darr.end());
                RIGHTMOUTH.x = sarr5[0];
                RIGHTMOUTH.y = sarr5[1];

                // 口左端
                darr = shape.part(64);
                std::vector<double> sarr6(darr.begin(), darr.end());
                LEFTMOUTH.x = sarr6[0];
                LEFTMOUTH.y = sarr6[1];

                // 眉間
                darr = shape.part(21);
                darr2 = shape.part(22);
                std::vector<double> sarr7(darr.begin(), darr.end());
                std::vector<double> sarr8(darr2.begin(), darr2.end());
                CENTEREYE.x = ((sarr8[0] - sarr7[0]) / 2) + sarr7[0];
                CENTEREYE.y = ((sarr8[1] - sarr7[1]) / 2) + sarr7[1];

                // 口てっぺん
                darr = shape.part(51);
                std::vector<double> sarr9(darr.begin(), darr.end());
                TOPMOUTH.x = sarr9[0];
                TOPMOUTH.y = sarr9[1];

                // dlib→opencv変換
                cv::Mat convertedMat = dlib::toMat(img);

                // トリミング座標がずれるとabort
                //右目
                    // trimming
                    init_x = OUTER_RIGHT_EYE.x - CENTEREYE.x + INNER_RIGHT_EYE.x;
                    init_y = CENTEREYE.y - 40;
                    side_x = side_y = (2 * CENTEREYE.x) - INNER_RIGHT_EYE.x - OUTER_RIGHT_EYE.x;
                    cv::Mat trimming_righteye(convertedMat, cv::Rect(init_x, init_y, side_x, side_y));
                    // resize,保存
                    cv::resize(trimming_righteye, trimming_righteye, cv::Size(), (float)IMSIZE/trimming_righteye.cols,(float)IMSIZE/trimming_righteye.rows);
                    ostringstream oss2;
                    oss2 << DOCUMENT_ROOT << DOCUMENT_PROPOSED << DOCUMENT_RIGHTEYE << count << ".png";
                    string str2 = oss2.str();
                    cv::imwrite(str2, trimming_righteye);

                //左目
                    // trimming
                    init_x = CENTEREYE.x;
                    init_y = CENTEREYE.y - 40;
                    side_x = side_y = -(2 * CENTEREYE.x) + INNER_LEFT_EYE.x + OUTER_LEFT_EYE.x;
                    if((init_x + side_x) > IMSIZE){
                        int tmp = (init_x + side_x) - IMSIZE;
                        side_x = side_x - tmp;
                    }
                    cv::Mat trimming_lefteye(convertedMat, cv::Rect(init_x, init_y, side_x, side_y));
                    // resize,保存
                    cv::resize(trimming_lefteye, trimming_lefteye, cv::Size(), (float)IMSIZE/trimming_lefteye.cols,(float)IMSIZE/trimming_lefteye.rows);
                    ostringstream oss3;
                    oss3 << DOCUMENT_ROOT << DOCUMENT_PROPOSED << DOCUMENT_LEFTEYE << count << ".png";
                    string str3 = oss3.str();
                    cv::imwrite(str3, trimming_lefteye);

                //口全体
                    // trimming
                    init_x = RIGHTMOUTH.x - 30;
                    init_y = RIGHTMOUTH.y - 40;
                    min_y = RIGHTMOUTH.y;
                    if(min_y > RIGHTMOUTH.y){
                        min_y = RIGHTMOUTH.y;
                    }
                    side_x = 60 + LEFTMOUTH.x - RIGHTMOUTH.x;
                    side_y = 296 - min_y;
                    cv::Mat trimming_mouth(convertedMat, cv::Rect(init_x, init_y, side_x, side_y));
                    // resize,保存
                    cv::resize(trimming_mouth, trimming_mouth, cv::Size(), (float)IMSIZE/trimming_mouth.cols,(float)IMSIZE/trimming_mouth.rows);
                    ostringstream oss4;
                    oss4 << DOCUMENT_ROOT << DOCUMENT_PROPOSED << DOCUMENT_MOUTH << count << ".png";
                    string str4 = oss4.str();
                    cv::imwrite(str4, trimming_mouth);


                //res = pyWrapper_proposed(count);//chainer実行

                res = pyWrapper_proposed_all(count); //chainer実行

            }//detsここまで
        }//新手法ここまで

        if(res == 1)
        {
            // 緑色で囲む
            for (int i = 0; i < faces.size(); i++)
            {
                cv::rectangle(image, Point(faces[i].x,faces[i].y),Point(faces[i].x + faces[i].width,faces[i].y + faces[i].height),Scalar(0,200,0),1,CV_AA);
            }
        }
        else
        {
            cout << "not face"<< endl;
            // 青色で囲む
            for (int i = 0; i < faces.size(); i++)
            {
                cv::rectangle(image, Point(faces[i].x,faces[i].y),Point(faces[i].x + faces[i].width,faces[i].y + faces[i].height),Scalar(200,0,0),1,CV_AA);
            }
        }
        count++;
	}
	return image;
}

int main(int argc, char *argv[])
{
    Py_SetProgramName(argv[0]);
    Py_Initialize();

    // dlib 初回実行
    deserialize(LANDMARK_DAT) >> pose_model;
    image_window win, win_faces;

    // kinect
    if(dev == 0)
    {
        cout << "デバイスが接続されていないため検出できないよ！" << endl;
        return -1;
    }

    protonect_shutdown = false;
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
    //libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color);
    libfreenect2::FrameMap frames;

    // window設定
    namedWindow("DETECT_FACE",CV_WINDOW_NORMAL) ;
    resizeWindow("DETECT_FACE",1920/6,1080/6) ;
    moveWindow("DETECT_FACE",560,430) ;

    dev->setColorFrameListener(&listener);
    dev->setIrAndDepthFrameListener(&listener);
    dev->start();

    Mat colorMapImage;
    Mat depthConverted ;
    int colorMapType = COLORMAP_JET ;

    while(!protonect_shutdown)
    {

        listener.waitForNewFrame(frames);
        libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
        libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
        libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
        unsigned char **pprgba = reinterpret_cast<unsigned char **>(rgb->data);

        Mat rgba(1080, 1920, CV_8UC4, pprgba[0]);
        Mat bgra;

        resize(rgba,bgra,Size(),RESOLUTION,RESOLUTION);
        cvtColor(bgra, bgra, COLOR_RGB2BGR);
        flip(bgra,bgra,1);

        cv::Mat detectFaceImage;
        if(CASCADE == 0)
        {
            detectFaceImage = detectFaceInImage(bgra,HAARLIKE_XML);
        }
        else if(CASCADE == 1)
        {
            detectFaceImage = detectFaceInImage(bgra,LBP_XML);
        }

        imshow("DETECT_FACE",detectFaceImage);
        int key = waitKey(1);
        protonect_shutdown = protonect_shutdown || (key > 0 && ((key & 0xFF) == 27));
        listener.release(frames);

    }

    cout << "おわり" << endl;
    dev->stop();
    dev->close();
    Py_Finalize();

    return 0;
}

