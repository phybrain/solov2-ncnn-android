// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "nanodet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <set>
#include <map>

#include "cpu.h"
using namespace std;
float iou_thresh = 0.75;

NanoDet::NanoDet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
    colorFlag=0;
}

LcNet::LcNet() {
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
};
int LcNet::load(AAssetManager* mgrthis) {
    lcnet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
    lcnet.opt = ncnn::Option();

#if NCNN_VULKAN
    lcnet.opt.use_vulkan_compute = true;
#endif

    lcnet.opt.num_threads = ncnn::get_big_cpu_count();
    lcnet.opt.blob_allocator = &blob_pool_allocator;
    lcnet.opt.workspace_allocator = &workspace_pool_allocator;
    const int num_threads = 1;
    lcnet.opt.num_threads = num_threads;
    lcnet.opt.use_fp16_arithmetic = false;
    lcnet.opt.use_fp16_storage = false;
    lcnet.opt.use_packing_layout = false;
    int a = lcnet.load_param(mgrthis,"pplcnet-sim-sim-opt-fp16.param");
    int b = lcnet.load_model(mgrthis,"pplcnet-sim-sim-opt-fp16.bin");
    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "lcnet load ------------------: %d  %d",a,b);
    return 0;
};

ncnn::Mat softmax(ncnn::Mat& bottom_top_blob)  {
    // value = exp( value - global max value )
    // sum all value
    // value = value / sum
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    size_t elemsize = bottom_top_blob.elemsize;
    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "lcnet dims------------------: %d %d",dims,w);
    if (dims == 1) // positive_axis == 0
    {


        float *ptr = bottom_top_blob;

        float max = -FLT_MAX;
        for (int i = 0; i < w; i++) {
            max = std::max(max, ptr[i]);
        }

        float sum = 0.f;
        for (int i = 0; i < w; i++) {
            ptr[i] = static_cast<float>(exp(ptr[i] - max));
            sum += ptr[i];
        }
//        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "lcnet exp------------------: %.2f %.2f %.2f %.2f",max,sum,ptr[0],ptr[1]);
        for (int i = 0; i < w; i++) {
            ptr[i] /= sum;
        }
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "lcnet exp------------------: %.2f %.2f %.2f %.2f",max,sum,ptr[0],ptr[1]);
    }
    ncnn::Mat out = bottom_top_blob.clone();
    return out;
}
float LcNet::cls(cv::Mat &rgb) {
    cv::Mat inputimage;
    cv::cvtColor(rgb,inputimage,cv::COLOR_BGR2GRAY);
//    cv::Mat faceRoiImage = rgb.clone();
    ncnn::Extractor ex_cls = lcnet.create_extractor();


    ncnn::Option opt;

    int img_w = inputimage.cols;
    int img_h = inputimage.rows;
    int align_size = 64;
    cv::Mat resizeimage;
//    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "input image  %.2f , %.2f",ptr[0],ptr[1]);
    if(img_w>img_h){
        float ratio = target_size/float(img_w);
        int newh = int(img_h*ratio);
        cv::resize(inputimage,resizeimage,cv::Size(target_size,newh));
    } else{
        float ratio = target_size/float(img_h);
        int neww = int(img_w*ratio);
        cv::resize(inputimage,resizeimage,cv::Size(neww,target_size));
    };
    int w=resizeimage.cols;
    int h=resizeimage.rows;
    int wpad = target_size-w;
    int hpad = target_size-h;


    ncnn::Mat in_pad;
//    ncnn::Mat ncnn_in = ncnn::Mat::from_pixels_resize(inputimage.data,
//                                               ncnn::Mat::PIXEL_GRAY, resizeimage.cols, resizeimage.rows,800,800);
    ncnn::Mat ncnn_in = ncnn::Mat::from_pixels(resizeimage.data,
                                               ncnn::Mat::PIXEL_GRAY, resizeimage.cols, resizeimage.rows);

    ncnn::copy_make_border(ncnn_in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 255.f);
    int target_w = in_pad.w;
    int target_h = in_pad.h;
//    const float meanValues[1] = {127.5f};
//    const float normValues[1] = {1.0f / 127.5f};
//    ncnn_in.substract_mean_normalize(meanValues, normValues);
    ex_cls.input("input",ncnn_in);
    float *ptrin = ncnn_in;
    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "lcnet.input(\"input\",ncnn_in); %.2f ,%.2f ,%.2f ,%.2f ,%.2f ,",ptrin[0],ptrin[10],ptrin[20],ptrin[30],ptrin[40]);
    ncnn::Mat out;
    ex_cls.extract("output",out);
    float *ptr = out;
    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "lcnet.output(\"output\",output); %.2f , %.2f",ptr[0],ptr[1]);
    ncnn::Mat final = softmax(out);
    float *ptrf = final;
    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "lcnet.output(softmax); %.2f , %.2f",ptrf[0],ptrf[1]);
    return ptrf[0];

};

int NanoDet::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    hairseg.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    hairseg.opt = ncnn::Option();

#if NCNN_VULKAN
    hairseg.opt.use_vulkan_compute = use_gpu;
#endif

    hairseg.opt.num_threads = ncnn::get_big_cpu_count();
    hairseg.opt.blob_allocator = &blob_pool_allocator;
    hairseg.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    hairseg.load_param(parampath);
    hairseg.load_model(modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];
    return 0;
}

int NanoDet::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    hairseg.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    hairseg.opt = ncnn::Option();

#if NCNN_VULKAN
    hairseg.opt.use_vulkan_compute = use_gpu;
#endif
    hairseg.opt.lightmode = true;
    hairseg.opt.num_threads = ncnn::get_big_cpu_count();
    hairseg.opt.blob_allocator = &blob_pool_allocator;
    hairseg.opt.workspace_allocator = &workspace_pool_allocator;
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);
    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "param path-----: %s",parampath);
    hairseg.load_param(mgr,parampath);
    hairseg.load_model(mgr,modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];
    clsnet.load(mgr);
    return 0;
}

struct Object_my {
    int cx;
    int cy;
    int label;
    float prob;
    cv::Mat mask;
};

static inline float intersection_area(const Object_my &a, const Object_my &b, int img_w, int img_h) {
    float area = 0.f;
    for (int y = 0; y < img_h; y = y + 4) {
        for (int x = 0; x < img_w; x = x + 4) {
            const uchar *mp1 = a.mask.ptr(y);
            const uchar *mp2 = b.mask.ptr(y);
            if (mp1[x] == 255 && mp2[x] == 255) area += 1.f;
        }
    }
    return area;
}

static inline float area(const Object_my &a, int img_w, int img_h) {
    float area = 0.f;
    for (int y = 0; y < img_h; y = y + 4) {
        for (int x = 0; x < img_w; x = x + 4) {
            const uchar *mp = a.mask.ptr(y);
            if (mp[x] == 255) area += 1.f;
        }
    }
    return area;
}

static void qsort_descent_inplace(std::vector<Object_my> &objects, int left, int right) {
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j) {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object_my> &objects) {
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_segs(const std::vector<Object_my> &objects, std::vector<int> &picked, float nms_threshold, int img_w,
                            int img_h) {
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = area(objects[i], img_w, img_h);
    }

    for (int i = 0; i < n; i++) {
        const Object_my &a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const Object_my &b = objects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b, img_w, img_h);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void ins_decode(const ncnn::Mat &kernel_pred, const ncnn::Mat &feature_pred, std::vector<int> &kernel_picked,
                       std::map<int, int> &kernel_map, ncnn::Mat *ins_pred, int c_in,
                       ncnn::Option &opt) {
    std::set<int> kernel_pick_set;
    kernel_pick_set.insert(kernel_picked.begin(), kernel_picked.end());
    int c_out = kernel_pick_set.size();
    if (c_out > 0) {
        ncnn::Layer *op = ncnn::create_layer("Convolution");
        ncnn::ParamDict pd;
        pd.set(0, c_out);
        pd.set(1, 1);
        pd.set(6, c_in * c_out);
        op->load_param(pd);
        ncnn::Mat weights[1];
        weights[0].create(c_in * c_out);
        float *kernel_pred_data = (float *) kernel_pred.data;
        std::set<int>::iterator pick_c;
        int count_c = 0;
        for (pick_c = kernel_pick_set.begin(); pick_c != kernel_pick_set.end(); pick_c++)
        {
            kernel_map[*pick_c] = count_c;
            for (int j = 0; j < c_in; j++) {
                weights[0][count_c * c_in + j] = kernel_pred_data[c_in * (*pick_c) + j];
            }

            count_c++;
        }

        op->load_model(ncnn::ModelBinFromMatArray(weights));
        op->create_pipeline(opt);
        ncnn::Mat temp_ins;
        op->forward(feature_pred, temp_ins, opt);
        *ins_pred = temp_ins;
        op->destroy_pipeline(opt);
        delete op;
    }
}

static void kernel_pick(const ncnn::Mat &cate_pred, std::vector<int> &picked, int num_class, float cate_thresh)
{
    int w = cate_pred.w;
    int h = cate_pred.h;
    for (int q = 0; q < num_class; q++) {
        const float *cate_ptr = cate_pred.channel(q);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int index = i * w + j;
                float cate_score = cate_ptr[index];
                if (cate_score < cate_thresh) {
                    continue;
                }
                else  picked.push_back(index);
            }
        }
    }
}

static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

void generate_res(ncnn::Mat &cate_pred, ncnn::Mat &ins_pred, std::map<int, int> &kernel_map,std::vector<std::vector<Object_my> >&objects, float cate_thresh,
                  float conf_thresh, int img_w, int img_h, int num_class, float stride, int wpad, int hpad) {
    int w = cate_pred.w;
    int h = cate_pred.h;
    int w_ins = ins_pred.w;
    int h_ins = ins_pred.h;
    for (int q = 0; q < num_class; q++) {
        const float *cate_ptr = cate_pred.channel(q);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int index = i * w + j;
                float cate_socre = cate_ptr[index];
                if (cate_socre < cate_thresh) {
                    continue;
                }
                const float *ins_ptr = ins_pred.channel(kernel_map[index]);
                cv::Mat mask(h_ins, w_ins, CV_32FC1);
                float sum_mask = 0.f;
                int count_mask = 0;
                {
                    mask = cv::Scalar(0.f);
                    float *mp = (float *) mask.data;
                    for (int m = 0; m < w_ins * h_ins; m++) {
                        float mask_score = sigmoid(ins_ptr[m]);

                        if (mask_score > 0.5) {
                            mp[m] = mask_score;
                            sum_mask += mask_score;
                            count_mask++;
                        }
                    }
                }
                if (count_mask < stride) {
                    continue;
                }
                float mask_score = sum_mask / (float(count_mask) + 1e-6);

//                float socre = mask_score * cate_socre;
                float socre = mask_score * cate_socre;

                if (socre < conf_thresh) {
                    continue;
                }
                cv::Mat mask_cut ;
                cv::Rect rect(wpad/8,hpad/8,w_ins-wpad/4,h_ins-hpad/4);
                mask_cut = mask(rect);
                cv::Mat mask2;
                cv::resize(mask_cut, mask2, cv::Size(img_w, img_h));
                Object_my obj;
                obj.mask = cv::Mat(img_h, img_w, CV_8UC1);
                float sum_mask_y = 0.f;
                float sum_mask_x = 0.f;
                int area = 0;
                {
                    obj.mask = cv::Scalar(0);
                    for (int y = 0; y < img_h; y++) {
                        const float *mp2 = mask2.ptr<const float>(y);
                        uchar *bmp = obj.mask.ptr<uchar>(y);
                        for (int x = 0; x < img_w; x++) {

                            if (mp2[x] > 0.5f) {
                                bmp[x] = 255;
                                sum_mask_y += (float) y;
                                sum_mask_x += (float) x;
                                area++;

                            } else bmp[x] = 0;
                        }
                    }
                }

                if (area < 100) continue;

                obj.cx = int(sum_mask_x / area);
                obj.cy = int(sum_mask_y / area);
                obj.label = q + 1;


                obj.prob = socre;
                objects[q].push_back(obj);

            }
        }
    }

}

cv::Mat BGRToRGB(cv::Mat img)
{
    cv::Mat image(img.rows, img.cols, CV_8UC3); //CV_32FC1
    for(int i=0; i<img.rows; ++i)
    { //获取第i行首像素指针
         cv::Vec3b *p1 = img.ptr<cv::Vec3b>(i);
         cv::Vec3b *p2 = image.ptr<cv::Vec3b>(i);
         for(int j=0; j<img.cols; ++j)
         { //将img的bgr转为image的rgb
              p2[j][2] = p1[j][0];
              p2[j][1] = p1[j][1];
              p2[j][0] = p1[j][2];
         }
    }
    return image;
}

void getLinePoints(cv::Mat &dstImage,cv::Mat &rgb,vector<vector<cv::Point>> &contours,vector<cv::Vec4i> &hierarcy){//td::vector<cv::Point2f>
    int width = dstImage.cols;
    int height = dstImage.rows;
    cv::Mat drawImage = cv::Mat::zeros(dstImage.size(), CV_8UC1);
    for (size_t t = 0; t < contours.size(); t++) {
        cv::Rect rect = boundingRect(contours[t]);
        if (rect.width > width / 2 && rect.width < width - 5) {
            cv::drawContours(drawImage, contours, static_cast<int>(t), cv::Scalar(255), 4, 8,
                         hierarcy, 0, cv::Point());
        }
    }
    cv::imwrite("/storage/emulated/0/DCIM/mask.jpg",drawImage);
    vector<cv::Vec4i> lines;
    int accu = min(width*0.3, height*0.3);
    HoughLinesP(drawImage, lines, 1, CV_PI / 180.0, accu, accu, 10);
    int A = 1;
    double B = CV_PI / 180;


    for (size_t t = 0; t < lines.size(); t++) {
        cv::Vec4i ln = lines[t];
        cv::line(rgb, cv::Point(ln[0], ln[1]), cv::Point(ln[2], ln[3]), cv::Scalar(0, 255, 0), 2, 8, 0);
    }
//    printf("number of lines : %d\n", lines.size());
    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "number of lines : %d\n",lines.size());
    // 寻找与定位上下左右四条直线
    int deltah = 0;
    cv::Vec4i topLine, bottomLine;
    cv::Vec4i leftLine, rightLine;
    for (int i = 0; i < lines.size(); i++) {
        cv::Vec4i ln = lines[i];
        deltah = abs(ln[3] - ln[1]);
        if (ln[3] < height / 2.0 && ln[1] < height / 2.0 && deltah < accu - 1) {
            if (topLine[3] > ln[3] && topLine[3]>0) {
                topLine = lines[i];
            } else {
                topLine = lines[i];
            }
        }
        if (ln[3] > height / 2.0 && ln[1] > height / 2.0 && deltah < accu - 1) {
            bottomLine = lines[i];
        }
        if (ln[0] < width / 2.0 && ln[2] < width/2.0) {
            leftLine = lines[i];
        }
        if (ln[0] > width / 2.0 && ln[2] > width / 2.0) {
            rightLine = lines[i];
        }
    }

    // 拟合四条直线方程
    float k1, c1;
    k1 = float(topLine[3] - topLine[1]) / float(topLine[2] - topLine[0]);
    c1 = topLine[1] - k1*topLine[0];
    float k2, c2;
    k2 = float(bottomLine[3] - bottomLine[1]) / float(bottomLine[2] - bottomLine[0]);
    c2 = bottomLine[1] - k2*bottomLine[0];
    float k3, c3;
    k3 = float(leftLine[3] - leftLine[1]) / float(leftLine[2] - leftLine[0]);
    c3 = leftLine[1] - k3*leftLine[0];
    float k4, c4;
    k4 = float(rightLine[3] - rightLine[1]) / float(rightLine[2] - rightLine[0]);
    c4 = rightLine[1] - k4*rightLine[0];

    // 四条直线交点
    cv::Point p1; // 左上角
    p1.x = static_cast<int>((c1 - c3) / (k3 - k1));
    p1.y = static_cast<int>(k1*p1.x + c1);
    cv::Point p2; // 右上角
    p2.x = static_cast<int>((c1 - c4) / (k4 - k1));
    p2.y = static_cast<int>(k1*p2.x + c1);
    cv::Point p3; // 左下角
    p3.x = static_cast<int>((c2 - c3) / (k3 - k2));
    p3.y = static_cast<int>(k2*p3.x + c2);
    cv::Point p4; // 右下角
    p4.x = static_cast<int>((c2 - c4) / (k4 - k2));
    p4.y = static_cast<int>(k2*p4.x + c2);

    // 显示四个点坐标
    cv::circle(rgb, p1, 2, cv::Scalar(255, 0, 0), 2, 8, 0);
    cv::circle(rgb, p2, 2, cv::Scalar(255, 0, 0), 2, 8, 0);
    cv::circle(rgb, p3, 2, cv::Scalar(255, 0, 0), 2, 8, 0);
    cv::circle(rgb, p4, 2, cv::Scalar(255, 0, 0), 2, 8, 0);
    //cv::line(rgb, cv::Point(topLine[0], topLine[1]), cv::Point(topLine[2], topLine[3]), cv::Scalar(0, 255, 0), 2, 8, 0);

};
double EuDis(cv::Point pt1, cv::Point pt2)
{
    return sqrt((pt2.x - pt1.x)*(pt2.x - pt1.x) + (pt2.y - pt1.y)*(pt2.y - pt1.y));
};

double getTheta(cv::Point pt1, cv::Point pt2){
    double k = (double)(pt1.y-pt2.y)/(double)(pt1.x-pt2.x);
    double theta = atan(k);
    return theta;
};

int getPoints(cv::Mat &dstImage,cv::Mat &rgb,vector<vector<cv::Point>> &contours,vector<cv::Vec4i> &hierarcy,float iou){
    int affine = 0;
    int width = dstImage.cols;
    int height = dstImage.rows;
    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "------------------image w,h: %d, %d",width,height);
    vector<vector<cv::Point>> conPoly(contours.size());
    vector<cv::Point> srcPts;
    int maxindex=0;
    double maxarea = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        double area = cv::contourArea(contours[i]);
        if(area>maxarea){
          maxindex = i;
          maxarea = area;
        };

    }
    double peri = cv::arcLength(contours[maxindex], true);

    cv::approxPolyDP(contours[maxindex], conPoly[maxindex], 0.02 * peri, true);
    //获取矩形四个角点
    srcPts = {conPoly[maxindex][0], conPoly[maxindex][1], conPoly[maxindex][2],
              conPoly[maxindex][3]};
    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "------polysize: %d",
                        conPoly[maxindex].size());
    if(maxarea>width*height*0.25 && conPoly[maxindex].size()==4) {

        int T_L, T_R, B_R, B_L;
        float w = width / 2.0;
        float h = height / 2.0;
        for (int i = 0; i < srcPts.size(); i++) {
            if (srcPts[i].x < w && srcPts[i].y < w) {
                T_L = i;
            }
            if (srcPts[i].x > w && srcPts[i].y < h) {
                T_R = i;
            }
            if (srcPts[i].x > w && srcPts[i].y > h) {
                B_R = i;
            }
            if (srcPts[i].x < w && srcPts[i].y > h) {
                B_L = i;
            }

        }

        cv::circle(rgb, srcPts[T_L], 10, cv::Scalar(0, 0, 255), -1);
        cv::circle(rgb, srcPts[T_R], 10, cv::Scalar(0, 255, 255), -1);
        cv::circle(rgb, srcPts[B_R], 10, cv::Scalar(255, 0, 0), -1);
        cv::circle(rgb, srcPts[B_L], 10, cv::Scalar(0, 255, 0), -1);
        double theta1 = getTheta(srcPts[T_L], srcPts[B_L]);
        double theta2 = getTheta(srcPts[T_R], srcPts[B_R]);
        double theta3 = getTheta(srcPts[T_L], srcPts[T_R]);
        double theta4 = getTheta(srcPts[B_L], srcPts[B_R]);
        double B = CV_PI*5/180;
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "------lines theta: %.2f, %.2f, %.2f, %.2f B:%.2f",theta1,theta2,theta3,theta4,B);
       if(abs(abs(theta1)-abs(theta2))<B and iou>=iou_thresh) {

           double LeftHeight = EuDis(srcPts[T_L], srcPts[B_L]);
           double RightHeight = EuDis(srcPts[T_R], srcPts[B_R]);
           double MaxHeight = max(LeftHeight, RightHeight);

           double UpWidth = EuDis(srcPts[T_L], srcPts[T_R]);
           double DownWidth = EuDis(srcPts[B_L], srcPts[B_R]);
           double MaxWidth = max(UpWidth, DownWidth);

//           cv::Point2f SrcAffinePts[4] = {cv::Point2f(srcPts[T_L]), cv::Point2f(srcPts[T_R]),
//                                          cv::Point2f(srcPts[B_R]), cv::Point2f(srcPts[B_L])};

           cv::Point2f AffineSrcPts[4] = { cv::Point2f(srcPts[T_L]) ,cv::Point2f(srcPts[T_R]) ,cv::Point2f(srcPts[B_L]) ,cv::Point2f(srcPts[B_R]) };

           cv::Point2f AffineDstPts[4] = { cv::Point2f(0, 0),cv::Point2f(MaxWidth , 0),cv::Point2f(0, MaxHeight),cv::Point2f(MaxWidth , MaxHeight) };

           cv::Mat M = getPerspectiveTransform(AffineSrcPts, AffineDstPts);

           cv::Mat DstImg;
           warpPerspective(rgb, DstImg, M, cv::Point(MaxWidth, MaxHeight));
           cv::imwrite("/storage/emulated/0/DCIM/warp.jpg", DstImg);
           affine = 1;
       }

    };
    return affine;
};

float calcIOU(cv::RotatedRect rect1, cv::RotatedRect rect2) {
    float areaRect1 = rect1.size.width * rect1.size.height;
    float areaRect2 = rect2.size.width * rect2.size.height;
    vector<cv::Point2f> vertices;

    int intersectionType = cv::rotatedRectangleIntersection(rect1, rect2, vertices);
    if (vertices.size() == 0)
        return 0.0;
    else {
        vector<cv::Point2f> order_pts;
        // 找到交集（交集的区域），对轮廓的各个点进行排序

        cv::convexHull(cv::Mat(vertices), order_pts, true);
        double area = cv::contourArea(order_pts);
        float inner = (float) (area / (areaRect1 + areaRect2 - area + 0.0001));

        return inner;
    }
}

float findRect(cv::Mat &dstImage,cv::Mat &rgb,cv::RotatedRect gtbox,vector<vector<cv::Point>> &contours,vector<cv::Vec4i> &hierarcy){

//    vector<vector<cv::Point>> contours;
//    vector<cv::Vec4i> hierarcy;
    findContours(dstImage, contours, hierarcy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    vector<cv::RotatedRect> box(contours.size()); //最小外接矩形
    cv::Point2f rect[4];
    float width = 0;//外接矩形的宽和高
    float height = 0;
    float ratio = 0;  //存储长宽比=width/heigth
    int max_index = 0;
    float area = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        box[i] = cv::minAreaRect(cv::Mat(contours[i]));
        box[i].points(rect);          //最小外接矩形的4个端点
        width = box[i].size.width;
        height = box[i].size.height;
        if (height >= width)
        {
            float x = 0;
            x = height;
            height = width;
            width = x;
        }
        if(height*width>area)
        {
            area = height*width;
            max_index = i;
        }
        ratio = width / height;

    }
    if (contours.size()>0) {
        int left = 10000;
        int right = 0;
        int top = 10000;
        int bottom = 0;
        int x,y;
        box[max_index].points(rect);
        for (int j = 0; j < 4; j++) {
            cv::line(rgb, rect[j], rect[(j + 1) % 4], cv::Scalar(0, 0, 255), 1, 8);//绘制最小外接矩形的每条边
            x = rect[j].x;
            y = rect[j].y;
            if(x<left){left=x; if(left<0){left=0;}}
            if(x>right){right=x;if(right>rgb.size().width){right=rgb.size().width;}}
            if(y<top){top=y; if(top<0){top=0;}}
            if(y>bottom){bottom=y;if(bottom>rgb.size().height){bottom=rgb.size().height;}}

        }


        float iou = calcIOU(box[max_index],gtbox);
        if(iou>iou_thresh) {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "left,right,top,bottom: %d %d %d %d",left,right,top,bottom);
            cv::Rect irect(left, top, right - left,
                           bottom - top);   //创建一个Rect框，属于cv中的类，四个参数代表x,y,width,height
            cv::Mat image_cut = cv::Mat(rgb,
                                        irect);      //从img中按照rect进行切割，此时修改image_cut时image中对应部分也会修改，因此需要copy
            cv::Mat image_crop = image_cut.clone();
            cv::Point2f newpoints[4];
            for (int j = 0; j < 4; j++) {
                x = rect[j].x;
                y = rect[j].y;
                cv::Point2f newp = cv::Point2f(x - left, y - top);
                newpoints[j] = newp;
            }

            int img_crop_width = int(cv::norm(newpoints[0] - newpoints[1]));
            int img_crop_height = int(cv::norm(newpoints[0] - newpoints[3]));

            cv::Point2f pts_std[4];
            pts_std[0] = cv::Point2f(0, 0);
            pts_std[1] = cv::Point2f(img_crop_width, 0);
            pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
            pts_std[3] = cv::Point2f(0, img_crop_height);

            cv::Mat rotation, img_warp;
            rotation = cv::getPerspectiveTransform(newpoints, pts_std);
            cv::warpPerspective(image_crop, img_warp, rotation,
                                cv::Size(img_crop_width, img_crop_height));
            cv::imwrite("/storage/emulated/0/DCIM/opencv.jpg",img_warp);
        }
        return iou;
    }

    return 0;
}




int NanoDet::seg(cv::Mat &rgb,cv::Mat &mask,cv::Rect &box)
{
    cv::Mat faceRoiImage0 = rgb.clone();
    cv::Mat faceRoiImage = BGRToRGB(faceRoiImage0);
//    cv::Mat faceRoiImage = rgb.clone();
    ncnn::Extractor ex_hair = hairseg.create_extractor();
    ncnn::Mat ncnn_in = ncnn::Mat::from_pixels_resize(faceRoiImage.data,
        ncnn::Mat::PIXEL_RGB, faceRoiImage.cols, faceRoiImage.rows,target_size,target_size);

    std::vector <Object_my> objects;
    ncnn::Option opt;
    const int num_threads = 1;
    const float cate_thresh = 0.3f;
    const float confidence_thresh = 0.6f;
    const float nms_threshold = 0.1f;
    const int keep_top_k = 100;
    opt.num_threads = num_threads;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;
    int img_w = faceRoiImage.cols;
    int img_h = faceRoiImage.rows;
    int align_size = 64;
    int w=target_size;
    int h=target_size;

    int wpad = (w + align_size - 1) / align_size * align_size - w;
    int hpad = (h + align_size -1) / align_size * align_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(ncnn_in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    int target_w = in_pad.w;
    int target_h = in_pad.h;

    ncnn_in.substract_mean_normalize(mean_vals, norm_vals);

    size_t elemsize = sizeof(float);

    ncnn::Mat x_p3;
    ncnn::Mat x_p4;
    ncnn::Mat x_p5;
    // coord conv
    int pw = int(target_w / 8);
    int ph = int(target_h / 8);
    x_p3.create(pw, ph, 2, elemsize);
    float step_h = 2.f / (ph - 1);
    float step_w = 2.f / (pw - 1);
    for (int h = 0; h < ph; h++) {
        for (int w = 0; w < pw; w++) {
            x_p3.channel(0)[h * pw + w] = -1.f + step_w * (float) w;
            x_p3.channel(1)[h * pw + w] = -1.f + step_h * (float) h;
        }
    }

    pw = int(target_w / 16);
    ph = int(target_h / 16);
    x_p4.create(pw, ph, 2, elemsize);
    step_h = 2.f / (ph - 1);
    step_w = 2.f / (pw - 1);
    for (int h = 0; h < ph; h++) {
        for (int w = 0; w < pw; w++) {
            x_p4.channel(0)[h * pw + w] = -1.f + step_w * (float) w;
            x_p4.channel(1)[h * pw + w] = -1.f + step_h * (float) h;
        }
    }

    pw = int(target_w / 32);
    ph = int(target_h / 32);
    x_p5.create(pw, ph, 2, elemsize);
    step_h = 2.f / (ph - 1);
    step_w = 2.f / (pw - 1);
    for (int h = 0; h < ph; h++) {
        for (int w = 0; w < pw; w++) {
            x_p5.channel(0)[h * pw + w] = -1.f + step_w * (float) w;
            x_p5.channel(1)[h * pw + w] = -1.f + step_h * (float) h;
        }
    }

//    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "%d %d %d",ncnn_in.h,ncnn_in.w,ncnn_in.c);
    ex_hair.input("input",ncnn_in);
    ex_hair.input("p3_input", x_p3);
    ex_hair.input("p4_input", x_p4);
    ex_hair.input("p5_input", x_p5);
//    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "ex_hair.input(\"input\",ncnn_in);");
    ncnn::Mat out;
    ex_hair.extract("feature_pred",out);//ex_hair.extract("1006",out);
//    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "ex_hair.extract(\"feature_pred\",out);");
    ncnn::Mat cate_pred1, cate_pred2, cate_pred3, cate_pred4, cate_pred5, kernel_pred1, kernel_pred2, kernel_pred3, kernel_pred4, kernel_pred5;
    ex_hair.extract("cate_pred1", cate_pred1);
    ex_hair.extract("cate_pred2", cate_pred2);
    ex_hair.extract("cate_pred3", cate_pred3);
    ex_hair.extract("cate_pred4", cate_pred4);
    ex_hair.extract("cate_pred5", cate_pred5);
    ex_hair.extract("kernel_pred1", kernel_pred1);
    ex_hair.extract("kernel_pred2", kernel_pred2);
    ex_hair.extract("kernel_pred3", kernel_pred3);
    ex_hair.extract("kernel_pred4", kernel_pred4);
    ex_hair.extract("kernel_pred5", kernel_pred5);

    int num_class = cate_pred1.c;

    //ins decode

    int c_in = out.c;

    std::vector<int> kernel_picked1, kernel_picked2, kernel_picked3, kernel_picked4, kernel_picked5;
    kernel_pick(cate_pred1, kernel_picked1, num_class, cate_thresh);
    kernel_pick(cate_pred2, kernel_picked2, num_class, cate_thresh);
    kernel_pick(cate_pred3, kernel_picked3, num_class, cate_thresh);
    kernel_pick(cate_pred4, kernel_picked4, num_class, cate_thresh);
    kernel_pick(cate_pred5, kernel_picked5, num_class, cate_thresh);
//    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "kernel_pick");
    std::map<int, int> kernel_map1, kernel_map2, kernel_map3, kernel_map4, kernel_map5;
    ncnn::Mat ins_pred1, ins_pred2, ins_pred3, ins_pred4, ins_pred5;

    ins_decode(kernel_pred1, out, kernel_picked1,kernel_map1, &ins_pred1, c_in, opt);
    ins_decode(kernel_pred2, out, kernel_picked2,kernel_map2, &ins_pred2, c_in, opt);
    ins_decode(kernel_pred3, out, kernel_picked3,kernel_map3, &ins_pred3, c_in, opt);
    ins_decode(kernel_pred4, out, kernel_picked4,kernel_map4, &ins_pred4, c_in, opt);
    ins_decode(kernel_pred5, out, kernel_picked5,kernel_map5, &ins_pred5, c_in, opt);
//    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "ins_decode");
    std::vector<std::vector<Object_my> > class_candidates;
    class_candidates.resize(num_class);
    generate_res(cate_pred1, ins_pred1, kernel_map1, class_candidates, cate_thresh, confidence_thresh, img_w, img_h,
                 num_class, 8.f,wpad,hpad);
    generate_res(cate_pred2, ins_pred2, kernel_map2, class_candidates, cate_thresh, confidence_thresh, img_w, img_h,
                 num_class, 8.f,wpad,hpad);
    generate_res(cate_pred3, ins_pred3, kernel_map3, class_candidates, cate_thresh, confidence_thresh, img_w, img_h,
                 num_class,16.f,wpad,hpad);
    generate_res(cate_pred4, ins_pred4, kernel_map4, class_candidates, cate_thresh, confidence_thresh, img_w, img_h,
                 num_class,32.f,wpad,hpad);
    generate_res(cate_pred5, ins_pred5, kernel_map5, class_candidates, cate_thresh, confidence_thresh, img_w, img_h,
                 num_class,32.f,wpad,hpad);
//    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "generate_res");
    objects.clear();
    for (int i = 0; i < (int) class_candidates.size(); i++) {
        std::vector<Object_my> &candidates = class_candidates[i];

        qsort_descent_inplace(candidates);

        std::vector<int> picked;
        nms_sorted_segs(candidates, picked, nms_threshold, img_w, img_h);

        for (int j = 0; j < (int) picked.size(); j++) {
            int z = picked[j];
            objects.push_back(candidates[z]);
        }
    }
//    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "class_candidates");
    qsort_descent_inplace(objects);

    // keep_top_k
    if (keep_top_k < (int) objects.size()) {
        objects.resize(keep_top_k);
    }
//    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "keep_top_k");
//    float *scoredata = (float*)out.data;
//    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "float *scoredata = (float*)out.data;");
//    mask = cv::Mat(target_size, target_size, CV_32FC1, scoredata);

//    int w_ins = ins_pred1.w;
//    int h_ins = ins_pred1.h;
//    mask=cv::Mat(h_ins, w_ins, CV_32FC1);
//    __android_log_print(ANDROID_LOG_ERROR, "ncnn ","%d", objects.size());
//    mask = objects[0].mask; //size=0闪退
//    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "mask");

//    static const unsigned  char part_colors[8][3] = {{0, 0, 255}, {255, 85, 0}, {255, 170, 0},
//                                                     {255, 0, 85}, {255, 0, 170},
//                                                     {0, 255, 0}, {170, 255, 255}, {255, 255, 255}};
//    static const char *class_names[] = {"background",
//                                        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
//                                        "train", "truck", "boat", "traffic light", "fire hydrant",
//                                        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
//                                        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
//                                        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//                                        "skis", "snowboard", "sports ball", "kite", "baseball bat",
//                                        "baseball glove", "skateboard", "surfboard", "tennis racket",
//                                        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
//                                        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
//                                        "hot dog", "pizza", "donut", "cake", "chair", "couch",
//                                        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
//                                        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
//                                        "toaster", "sink", "refrigerator", "book", "clock", "vase",
//                                        "scissors", "teddy bear", "hair drier", "toothbrush"
//    };
    static const char *class_names[] = {"background",
                                        "invoice"};
    static const unsigned char colors[81][3] = {
            {56,  0,   255},
            {226, 255, 0},
            {0,   94,  255},
            {0,   37,  255},
            {0,   255, 94},
            {255, 226, 0},
            {0,   18,  255},
            {255, 151, 0},
            {170, 0,   255},
            {0,   255, 56},
            {255, 0,   75},
            {0,   75,  255},
            {0,   255, 169},
            {255, 0,   207},
            {75,  255, 0},
            {207, 0,   255},
            {37,  0,   255},
            {0,   207, 255},
            {94,  0,   255},
            {0,   255, 113},
            {255, 18,  0},
            {255, 0,   56},
            {18,  0,   255},
            {0,   255, 226},
            {170, 255, 0},
            {255, 0,   245},
            {151, 255, 0},
            {132, 255, 0},
            {75,  0,   255},
            {151, 0,   255},
            {0,   151, 255},
            {132, 0,   255},
            {0,   255, 245},
            {255, 132, 0},
            {226, 0,   255},
            {255, 37,  0},
            {207, 255, 0},
            {0,   255, 207},
            {94,  255, 0},
            {0,   226, 255},
            {56,  255, 0},
            {255, 94,  0},
            {255, 113, 0},
            {0,   132, 255},
            {255, 0,   132},
            {255, 170, 0},
            {255, 0,   188},
            {113, 255, 0},
            {245, 0,   255},
            {113, 0,   255},
            {255, 188, 0},
            {0,   113, 255},
            {255, 0,   0},
            {0,   56,  255},
            {255, 0,   113},
            {0,   255, 188},
            {255, 0,   94},
            {255, 0,   18},
            {18,  255, 0},
            {0,   255, 132},
            {0,   188, 255},
            {0,   245, 255},
            {0,   169, 255},
            {37,  255, 0},
            {255, 0,   151},
            {188, 0,   255},
            {0,   255, 37},
            {0,   255, 0},
            {255, 0,   170},
            {255, 0,   37},
            {255, 75,  0},
            {0,   0,   255},
            {255, 207, 0},
            {255, 0,   226},
            {255, 245, 0},
            {188, 255, 0},
            {0,   255, 18},
            {0,   255, 75},
            {0,   255, 151},
            {255, 56,  0},
            {245, 255, 0}
    };
    int color_index = 0;
    float max_prob=0;
    int max_prob_i=0;
    for (size_t i = 0; i < objects.size(); i++) {//objects[0]最大
//        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "%s %d",class_names[objects[i].label],objects[i].label);
        if (class_names[objects[i].label=='book']){
//            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "%f %d",objects[i].prob,i);
            if (objects[i].prob>max_prob){
                max_prob_i=i;
                max_prob=objects[i].prob;
            }
        }
    }
    float blur = clsnet.cls(rgb);
    int boxx = int(img_w*0.05);
    int boxy = int(img_h*0.05);
    int boxw = int(img_w*0.9);
    int boxh = int(img_h*0.9);
    cv::rectangle(rgb, cv::Rect(cv::Point(boxx, boxy), cv::Size(boxw, boxh)),
                  cv::Scalar(255, 0, 0), 3);
    cv::RotatedRect gtbox = cv::RotatedRect(cv::Point2f(boxx, boxy),cv::Point2f(boxx+boxw, boxy),cv::Point2f(boxx+boxw, boxy+boxh));
    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "%d %f %d",max_prob_i,max_prob,objects.size());

    int getInvoice = 0;
    for (size_t i = 0; i < objects.size(); i++) {
        if (i!=max_prob_i){
            continue;
        }
        const Object_my &obj = objects[i];
//        if (obj.label!=74){//book
//            continue;
//        }
        mask=obj.mask;
        cv::Mat maskResize;
        cv::resize(mask, maskResize, cv::Size(rgb.cols, rgb.rows), 0, 0, 1);
        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarcy;
        float iou = findRect(mask,rgb,gtbox,contours,hierarcy);
//        getLinePoints(mask,rgb,contours,hierarcy);
        int affine = getPoints(mask,rgb,contours,hierarcy,iou);
        if(iou>iou_thresh && affine){
            getInvoice=1;

        }
        const unsigned char *color = colors[color_index % 81];
        color_index++;

        char text[256];
//        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
        sprintf(text, "%s iou: %.2f blur: %.2f", class_names[obj.label],iou,blur);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.cx;
        int y = obj.cy;



        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        for (int y = 0; y < rgb.rows; y++) {
            const uchar *mp = obj.mask.ptr(y);
            uchar *p = rgb.ptr(y);
            for (int x = 0; x < rgb.cols; x++) {
                if (mp[x] == 255) {
                    p[0] = cv::saturate_cast<uchar>(p[0] * 0.5 + color[0] * 0.5);
                    p[1] = cv::saturate_cast<uchar>(p[1] * 0.5 + color[1] * 0.5);
                    p[2] = cv::saturate_cast<uchar>(p[2] * 0.5 + color[2] * 0.5);
                }
                p += 3;
            }
        }

/*        for (size_t h = 0; h < maskResize.rows; h++) {
            cv::Vec3b *pRgb = rgb.ptr<cv::Vec3b>(h);
            float *alpha = maskResize.ptr<float>(h);
            for (size_t w = 0; w < maskResize.cols; w++) {
                float weight = alpha[w];
                pRgb[w] = cv::Vec3b(part_colors[colorFlag][2] * weight + pRgb[w][0] * (1 - weight),
                                    part_colors[colorFlag][1] * weight + pRgb[w][1] * (1 - weight),
                                    part_colors[colorFlag][0] * weight + pRgb[w][2] * (1 - weight));
            }
        }
        if (colorFlag < 7)
            colorFlag++;
        else
            colorFlag = 0;*/
    }
//    draw_objects(rgb, objects,"",1)
    return getInvoice;
}

int NanoDet::draw(cv::Mat& rgb)
{
    static const unsigned  char part_colors[8][3] = {{0, 0, 255}, {255, 85, 0}, {255, 170, 0},
                   {255, 0, 85}, {255, 0, 170},
                   {0, 255, 0}, {170, 255, 255}, {255, 255, 255}};
    int color_index = 0;

    cv::Mat mask;
    cv::Rect  box;
    int getInvoice = seg(rgb,mask,box);
//    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "seg(rgb,mask,box);");
//    cv::Mat maskResize;
//    cv::resize(mask,maskResize,cv::Size(rgb.cols,rgb.rows),0,0,1);
//
//    for(size_t h = 0; h < maskResize.rows; h++)
//    {
//        cv::Vec3b* pRgb = rgb.ptr<cv::Vec3b >(h);
//        float *alpha = maskResize.ptr<float>(h);
//        for(size_t w = 0; w < maskResize.cols; w++)
//        {
//            float weight = alpha[w];
//            pRgb[w] = cv::Vec3b(part_colors[colorFlag][2]*weight+pRgb[w][0]*(1-weight),
//                                    part_colors[colorFlag][1]*weight+pRgb[w][1]*(1-weight),
//                                    part_colors[colorFlag][0]*weight+pRgb[w][2]*(1-weight));
//        }
//    }
//    if(colorFlag < 7)
//        colorFlag++;
//    else
//        colorFlag=0;
    return getInvoice;
}
