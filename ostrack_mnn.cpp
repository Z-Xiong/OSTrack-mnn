//
// Created by zxiong on 23-3-31.
//

#include <cstdlib>
#include <string>
#include <cmath>

#include "ostrack_mnn.h"

#define TIME
#ifdef TIME
#include <sys/time.h>
#endif

#ifdef TIME
    struct timeval tv;
    uint64_t time_last;
    double time_ms;
#endif

using namespace MNN;

static std::vector<float> hann(int sz) {
    std::vector<float> hann1d(sz);
    std::vector<float> hann2d(sz * sz);
    for (int i = 1; i < sz + 1; i++) {
        float w = 0.5f - 0.5f * std::cos(2 * 3.1415926535898f * i / (sz+1) );
        hann1d[i-1] = w;
    }
    for (int i = 0; i < sz; i++) {
        for (int j=0; j < sz; j++) {
            hann2d[i*sz + j] = hann1d[i] * hann1d[j];
        }
    }
   return hann2d;
}

OSTrack::OSTrack(const char *model_path) {   
    this->mnn_net = std::unique_ptr<Interpreter>(Interpreter::createFromFile(model_path));

    // Set seesion.
    this->net_config.type = MNN_FORWARD_CPU;
    this->net_config.numThread = 8;
    this->session =  this->mnn_net->createSession(this->net_config);

    // Get input tesnor.
    this->x = this->mnn_net->getSessionInput(this->session, "x");
    this->z = this->mnn_net->getSessionInput(this->session, "z");

    // Update model config.
    if (this->z->width() == 192) {
        this->cfg.search_size = 384;
        this->cfg.search_factor = 5.0;
        this->cfg.template_size = 192;
        this->cfg.feat_sz = 24;
    }

    // Set input convert config.
    ::memcpy(this->config.mean,   this->means, sizeof(this->means));
    ::memcpy(this->config.normal, this->norms, sizeof(this->norms));
    config.sourceFormat = CV::BGR;
    config.destFormat = CV::RGB;
    this->pretreat = std::unique_ptr<CV::ImageProcess>(CV::ImageProcess::create(config));

    // Generate hann2d window.
    this->cfg.window = hann(this->cfg.feat_sz);
}

OSTrack::~OSTrack()
{
    if (this->session != nullptr) {
        this->mnn_net->releaseSession(this->session);
        this->session = nullptr;
    }
    this->mnn_net.reset();
}


void OSTrack::init(const cv::Mat &img, DrOBB bbox)
{
    // Get subwindow.
    cv::Mat z_patch;
    float resize_factor = 1.f;
    this->sample_target(img, z_patch, bbox.box, this->cfg.template_factor, this->cfg.template_size, resize_factor);
    // cv::Mat -> MNN input tensor
    this->z_patch = z_patch;
    this->state = bbox.box;
}

const DrOBB &OSTrack::track(const cv::Mat &img)
{
    // Get subwindow.
    cv::Mat x_patch;
    float resize_factor = 1.f;
    this->sample_target(img, x_patch, this->state, this->cfg.search_factor, this->cfg.search_size, resize_factor);

    // Preprocess.
    this->pretreat->convert((uint8_t*)x_patch.data, x->width(), x->height(), 0, x);
    this->pretreat->convert((uint8_t*)this->z_patch.data, z->width(), z->height(), 0, z);
    
    // Run.
    auto ret = this->mnn_net->runSession(this->session);

    // Get output.
    auto scores  = this->mnn_net->getSessionOutput(this->session, "score_map");
    auto sizes   = this->mnn_net->getSessionOutput(this->session, "size_map");
    auto offsets = this->mnn_net->getSessionOutput(this->session, "offset_map");

    // Postprocess.
    Tensor scores_host(scores, Tensor::CAFFE);
    Tensor sizes_host(sizes, Tensor::CAFFE);
    Tensor offsets_host(offsets, Tensor::CAFFE);

    scores ->copyToHostTensor(&scores_host);
    sizes  ->copyToHostTensor(&sizes_host);
    offsets->copyToHostTensor(&offsets_host);

    DrBBox pred_box;
    float max_score;
    this->cal_bbox(scores_host, offsets_host, sizes_host, pred_box, max_score, resize_factor);

    this->map_box_back(pred_box, resize_factor);
    this->clip_box(pred_box, img.rows, img.cols, 10);
    
    object_box.box = pred_box;
    object_box.class_id = 0;
    object_box.score = max_score;

    this->state = object_box.box;

    return object_box;
}

void OSTrack::cal_bbox(Tensor &scores_tensor, Tensor &offsets_tensor, Tensor &sizes_tensor, DrBBox &pred_box, float &max_score, float resize_factor) {
    // Add hann window, get max value and index.
    auto scores_ptr = scores_tensor.host<float>();
    float max_value = this->cfg.window[0] * scores_ptr[0];
    int max_idx_y = 0; int max_idx_x = 0; int max_idx = 0;
    float tmp_score = 0.f;
    for (int i = 0; i < scores_tensor.elementSize(); i++) {
        tmp_score = this->cfg.window[i] * scores_ptr[i];
        if (tmp_score > max_value) {
            max_idx = i;
            max_value = tmp_score;
        }
    }
    max_idx_y = max_idx / scores_tensor.width();
    max_idx_x = max_idx % scores_tensor.width();

    auto sizes_ptr = sizes_tensor.host<float>();
    auto offsets_ptr = offsets_tensor.host<float>();

    float cx = (max_idx_x + offsets_ptr[max_idx_y * offsets_tensor.width() + max_idx_x]) * 1.f / this->cfg.feat_sz;
    float cy = (max_idx_y + offsets_ptr[offsets_tensor.width() * offsets_tensor.height() + max_idx_y * offsets_tensor.width() + max_idx_x]) *1.f / this->cfg.feat_sz;

    float w = sizes_ptr[max_idx_y * sizes_tensor.width() + max_idx_x];
    float h = sizes_ptr[sizes_tensor.width() * sizes_tensor.height() + max_idx_y * sizes_tensor.width() + max_idx_x];
  
    cx = cx * this->cfg.search_size / resize_factor;
    cy = cy * this->cfg.search_size / resize_factor;
    w = w * this->cfg.search_size / resize_factor;
    h = h * this->cfg.search_size / resize_factor;
    
    pred_box.x0 = cx - 0.5 * w;
    pred_box.y0 = cy - 0.5 * h;
    pred_box.x1 = pred_box.x0 + w;
    pred_box.y1 = pred_box.y0 + h;    

    max_score = max_value;
}

void OSTrack::map_box_back(DrBBox &pred_box, float resize_factor) {
    float cx_prev = this->state.x0 + 0.5 * (this->state.x1 - this->state.x0);
    float cy_prev = this->state.y0 + 0.5 * (this->state.y1 - this->state.y0);

    float half_side = 0.5 * this->cfg.search_size / resize_factor;

    float w = pred_box.x1 - pred_box.x0;
    float h = pred_box.y1 - pred_box.y0;
    float cx = pred_box.x0 + 0.5 * w;
    float cy = pred_box.y0 + 0.5 * h;

    float cx_real = cx + (cx_prev - half_side);
    float cy_real = cy + (cy_prev - half_side);

    pred_box.x0 = cx_real - 0.5 * w;
    pred_box.y0 = cy_real - 0.5 * h;
    pred_box.x1 = cx_real + 0.5 * w;
    pred_box.y1 = cy_real + 0.5 * h;
}

void OSTrack::clip_box(DrBBox &box, int height, int wight, int margin) {
    box.x0 = std::min(std::max(0, int(box.x0)), wight - margin);
    box.y0 = std::min(std::max(0, int(box.y0)), height - margin);
    box.x1 = std::min(std::max(margin, int(box.x1)), wight);
    box.y1 = std::min(std::max(margin, int(box.y1)), height);
}

void OSTrack::sample_target(const cv::Mat &im, cv::Mat &croped, DrBBox target_bb, float search_area_factor, int output_sz, float &resize_factor) {
    /* Extracts a square crop centrered at target_bb box, of are search_area_factor^2 times target_bb area

    args:
        im: Img image
        target_bb - target box [x0, y0, x1, y1]
        search_area_factor - Ratio of crop size to target size
        output_sz - Size
    
    */
   int x = target_bb.x0;
   int y = target_bb.y0;
   int w = target_bb.x1 - target_bb.x0;
   int h = target_bb.y1 - target_bb.y0;
   int crop_sz = std::ceil(std::sqrt(w *h) * search_area_factor);

   float cx = x + 0.5 * w;
   float cy = y + 0.5 * h;
   int x1 = std::round(cx - crop_sz * 0.5);
   int y1 = std::round(cy - crop_sz * 0.5);

   int x2 = x1 + crop_sz;
   int y2 = y1 + crop_sz;

   int x1_pad = std::max(0, -x1);
   int x2_pad = std::max(x2 - im.cols +1, 0);
   
   int y1_pad = std::max(0, -y1);
   int y2_pad = std::max(y2- im.rows + 1, 0);

   // Crop target
   cv::Rect roi_rect(x1+x1_pad, y1+y1_pad, (x2-x2_pad)-(x1+x1_pad), (y2-y2_pad)-(y1+y1_pad));
   cv::Mat roi = im(roi_rect);

   // Pad
   cv::copyMakeBorder(roi, croped, y1_pad, y2_pad, x1_pad, x2_pad, cv::BORDER_CONSTANT);

   // Resize
   cv::resize(croped, croped, cv::Size(output_sz, output_sz));

   resize_factor = output_sz * 1.f / crop_sz;
}
