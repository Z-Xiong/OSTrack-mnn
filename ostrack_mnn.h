//
// Created by zxiong on 23-3-31.
//

#ifndef OSTRACK_H
#define OSTRACK_H

#include <vector> 
#include <map>
#include <memory>

#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Matrix.h>
#include <MNN/Tensor.hpp>

#include "opencv2/opencv.hpp"

struct DrBBox {
    float x0;
    float y0;
    float x1;
    float y1;
};

struct DrOBB {
    DrBBox box;
    float score;
    int class_id;
};

struct Config {
    std::vector<float> window;

    float template_factor = 2.0;
    float search_factor = 4.0; // 5.0
    float template_size = 128; //192
    float search_size = 256; // 384
    float stride = 16;
    int feat_sz = 16; // 24
};

class OSTrack {

public: 
    
    OSTrack(const char *model_path);
    
    ~OSTrack(); 

    void init(const cv::Mat &img, DrOBB bbox);
    
    const DrOBB &track(const cv::Mat &img);
    
    // state  dynamic
    DrBBox state;
    
    // config static
    Config cfg; 

private:

    void map_box_back(DrBBox &pred_box, float resize_factor);

    void clip_box(DrBBox &box, int height, int wight, int margin);

    void cal_bbox(MNN::Tensor &scores_tensor, MNN::Tensor &offsets_tensor, MNN::Tensor &sizes_tensor, DrBBox &pred_box, float &max_score, float resize_factor);

    void sample_target(const cv::Mat &im, cv::Mat &croped, DrBBox target_bb, float search_area_factor, int output_sz, float &resize_factor);

    const float means[3]  = {0.406*255, 0.485*255, 0.456*255}; // BGR
    const float norms[3] = {1/(0.225*255), 1/(0.229*255), 1/(0.224*255)}; // BGR
    
    std::unique_ptr<MNN::Interpreter> mnn_net = nullptr;

    MNN::Tensor *x = nullptr;
    MNN::Tensor *z = nullptr;

    MNN::Session *session = nullptr;

    MNN::ScheduleConfig net_config;

    cv::Mat z_patch;

    MNN::CV::ImageProcess::Config config;

    std::unique_ptr<MNN::CV::ImageProcess> pretreat;

    DrOBB object_box;
};

#endif 
