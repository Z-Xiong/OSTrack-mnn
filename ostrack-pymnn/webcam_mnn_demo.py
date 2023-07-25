import cv2
import numpy as np
import math

import MNN

import torch

from utils import cxy_wh_2_rect, hann1d, hann2d, img2tensor

class Tracker(object):
    """Wraps the tracker for evaluation and running purposes.
    args:
        model_path - path of model.
    """
    def __init__(self, model_path: str) -> None:
        self.interpreter = MNN.Interpreter(model_path) 
        self.session = self.interpreter.createSession()
        

        self.template_factor = 2.0
        self.search_factor = 4.0 # 4.0
        self.template_size = 128 #128
        self.search_size = 256 #256
        self.stride = 16
        self.feat_sz = self.search_size // self.stride
        self.output_window = hann2d(np.array([self.feat_sz, self.feat_sz]), centered=True)
        self.state = None
        

    def initialize(self, image, target_bb):
        # get subwindow
        z_patch_arr, resize_factor, z_amask_arr = self.sample_target(image, target_bb, self.template_factor,
                                                    output_sz=self.template_size)
        # nparry -> MNN input tensor
        self.z = img2tensor(z_patch_arr)

        # save states
        self.state = target_bb
    
    def track(self, image):
        img_H, img_W, _ = image.shape

        input_tensors_z = self.interpreter.getSessionInput(self.session, 'z')
        input_tensors_x = self.interpreter.getSessionInput(self.session, 'x')

        B, C, H, W = self.z.shape
        tmp_input_z = MNN.Tensor([1, C, H, W], MNN.Halide_Type_Float, self.z.astype(np.float32), MNN.Tensor_DimensionType_Caffe)
        input_tensors_z.copyFrom(tmp_input_z)

        # get subwindow
        x_patch_arr, resize_factor, x_amask_arr = self.sample_target(image, self.state, self.search_factor,
                                                    output_sz=self.search_size)
        # nparry -> MNN input tensor
        x = img2tensor(x_patch_arr)
        B, C, H, W = x.shape
        tmp_input_x = MNN.Tensor([1, C, H, W], MNN.Halide_Type_Float, x.astype(np.float32), MNN.Tensor_DimensionType_Caffe)
        input_tensors_x.copyFrom(tmp_input_x)

        ret = self.interpreter.runSession(self.session)
        output_tensors_dict = self.interpreter.getSessionOutputAll(self.session)

        #constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
        feat_sz = int(self.feat_sz)

        out_score_map = MNN.Tensor((1, 1, feat_sz, feat_sz), MNN.Halide_Type_Float, np.ones([1, 1, feat_sz, feat_sz]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
        output_tensors_dict['score_map'].copyToHostTensor(out_score_map)

        out_size_map = MNN.Tensor((1, 2, feat_sz, feat_sz), MNN.Halide_Type_Float, np.ones([1, 2, feat_sz, feat_sz]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
        output_tensors_dict['size_map'].copyToHostTensor(out_size_map)

        out_offset_map = MNN.Tensor((1, 2, feat_sz, feat_sz), MNN.Halide_Type_Float, np.ones([1, 2, feat_sz, feat_sz]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
        output_tensors_dict['ConvInt81489'].copyToHostTensor(out_offset_map)

        score_map_arr = out_score_map.getNumpyData()
        size_map_arr = out_size_map.getNumpyData()
        offset_map_arr = out_offset_map.getNumpyData()

        # add hann windows
        response = self.output_window * score_map_arr
        pred_boxes = self.cal_bbox(response, size_map_arr, offset_map_arr)
        pred_box = (pred_boxes * self.search_size / resize_factor).tolist()
        self.state = self.clip_box(self.map_box_back(pred_box, resize_factor), img_H, img_W, margin=10)

        return self.state


    def sample_target(self, im, target_bb, search_area_factor, output_sz):
        """Extracts a square crop centered at target_bb box, of are search_area_factor^2 times target_bb area

        args: 
            im - cv image
            target_bb - target box [x_left, y_left, w, h]
            search_area_factor - Ratio of crop size to target size
            output_sz - (float) Size
        """
        if not isinstance(target_bb, list):
            x, y, w, h = list(target_bb)
        else:
            x, y , w, h = target_bb
        # crop image
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        if crop_sz < 1:
            raise Exception("Too small bounding box.")
        
        cx, cy = x + 0.5 * w, y + 0.5 * h
        x1 = round(cx - crop_sz * 0.5)
        y1 = round(cy - crop_sz * 0.5)

        x2 = x1 + crop_sz
        y2 = y1 + crop_sz 

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - im.shape[1] + 1, 0)

        y1_pad = max(0, -y1)
        y2_pad = max(y2 - im.shape[0] + 1, 0)

        # Crop target
        im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
        
        # Pad
        im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)

        # deal with attention mask
        H, W, _ = im_crop_padded.shape
        att_mask = np.ones((H,W))
        end_x, end_y = -x2_pad, -y2_pad
        if y2_pad == 0:
            end_y = None
        if x2_pad == 0:
            end_x = None
        att_mask[y1_pad:end_y, x1_pad:end_x] = 0

        resize_factor = output_sz / crop_sz
        im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv2.resize(att_mask, (output_sz, output_sz))

        return im_crop_padded, resize_factor, att_mask
    
    def transform_bbox_to_crop(self, box_in: list, resize_factor, crop_type='template', normalize=True) -> list:
        """Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
        args:
            box_in: list [x1, y1, w, h], not normalized, the box for which the co-ordinates are to be transformed
            resize_factor - the ratio between the original image scale and the scale of the image crop

        returns:
            List - transformed co-ordinates of box_in
        """
        
        
        if crop_type == 'template':
            crop_sz = self.template_size
        elif crop_type == 'search':
            crop_sz = self.search_size
        else:
            raise NotImplementedError
        
        box_out_center_x = (crop_sz[0] - 1) / 2
        box_out_center_y = (crop_sz[1] - 1) / 2
        box_out_w = box_in[2] * resize_factor
        box_out_h = box_in[3] * resize_factor

        # normalized
        box_out_x1 = (box_out_center_x - 0.5 * box_out_w)
        box_out_y1 = (box_out_center_y - 0.5 * box_out_h)
        box_out = [box_out_x1, box_out_y1, box_out_w, box_out_h]

        if normalize:
            return [i / crop_sz for i in box_out]
        else:
            return box_out
        
    def generate_mask_cond(self):
        template_size = self.template_size
        stride = self.stride
        template_feat_size = template_size// stride # 128 // 16 = 8

        # MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'CTR_POINT'

        box_mask_z = np.zeros([1, template_feat_size, template_feat_size])
        box_mask_z[:, slice(3, 4), slice(3, 4)] = 1
        box_mask_z = np.reshape(box_mask_z, (1, -1)).astype(np.int32)

        return box_mask_z
    
    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        score_map_ctr = torch.from_numpy(score_map_ctr)
        size_map = torch.from_numpy(size_map)
        offset_map = torch.from_numpy(offset_map)
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        print("idx: ", idx, "idx_y: ", idx_y, "idx_x: ", idx_x, "max score: ", max_score)

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)
        
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)
        bbox = bbox.numpy()[0]

        if return_score:
            return bbox, max_score
        return bbox
    
    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def clip_box(self, box: list, H, W, margin=0):
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        x1 = min(max(0, x1), W-margin)
        x2 = min(max(margin, x2), W)
        y1 = min(max(0, y1), H-margin)
        y2 = min(max(margin, y2), H)
        w = max(margin, x2-x1)
        h = max(margin, y2-y1)
        return [x1, y1, w, h]

def run(tracker, video_path):
    '''
    tracker: ostrack
    video_path: 0 or video path
    '''

    video_path = eval(video_path) if video_path.isnumeric() else video_path
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == 0:
            cap.release()
            break
        frame_count = frame_count + 1
        frame_track = frame.copy()

        # 0. Init Tracker
        if frame_count == 1:
            # cv2.putText(frame, 'Select target face ROI and press ENTER', (20, 30),
            #             cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #             1, (0, 0, 0), 1)
            
            # bbox = cv2.selectROI("demo", frame, fromCenter=False) # bbox (x, y, w, h)
            bbox = (744, 417, 42, 95)
            tracker.initialize(frame_track, bbox)
        else:
            bbox = tracker.track(frame_track) # bbox (x, y, w, h)
        x, y, w, h = bbox

        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        location = cxy_wh_2_rect(target_pos, target_sz)

        x1, y1, x2, y2 = int(location[0]), int(location[1]), \
            int(location[0] + location[2]), int(location[1] + location[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))

        cv2.imshow("img", frame)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()      

if __name__ == "__main__":
    ostrack = Tracker(model_path="/home/zxiong/hd0/ClionProjects/OSTrack-mnn/model/quant_ostrack-256-ep300.mnn")
    ostrack = Tracker(model_path="/home/zxiong/hd0/ClionProjects/OSTrack-mnn/model/quant_ostrack-256-ep300.mnn")
    ostrack = Tracker(model_path="/home/zxiong/hd0/ClionProjects/OSTrack-mnn/model/quant_ostrack-256-ep300.mnn")
    ostrack = Tracker(model_path="/home/zxiong/hd0/ClionProjects/OSTrack-mnn/model/quant_ostrack-256-ep300.mnn")
    ostrack = Tracker(model_path="/home/zxiong/hd0/ClionProjects/OSTrack-mnn/model/ostrack-384-ep300.mnn")
    ostrack = Tracker(model_path="/home/zxiong/hd0/ClionProjects/OSTrack-mnn/model/quant_ostrack-256-ep300.mnn")
    ostrack = Tracker(model_path="/home/zxiong/hd0/ClionProjects/OSTrack-mnn/model/quant_ostrack-256-ep300.mnn")
    ostrack = Tracker(model_path="/home/zxiong/hd0/ClionProjects/OSTrack-mnn/model/quant_ostrack-256-ep300.mnn")
    ostrack = Tracker(model_path="/home/zxiong/hd0/ClionProjects/OSTrack-mnn/model/quant_ostrack-256-ep300.mnn")
    run(ostrack, video_path="/home/zxiong/hd0/ClionProjects/LightTrack-ncnn/install/lighttrack_demo/1.avi")
    # run(ostrack, video_path="0")

    