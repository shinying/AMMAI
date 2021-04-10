from torchvision.transforms.functional import resize
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .data import cfg_mnet, cfg_re50
from .detect import load_model
from .layers.functions.prior_box import PriorBox
from .models.retinaface import RetinaFace
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms


class Detector:

    def __init__(self, network='resnet50', pretrained='weights/Resnet50_Final.pth',
                 cpu=False, confidence_thresh=0.02, top_k=5000, nms_thresh=0.4,
                 keep_top_k=750, mean=[104, 117, 123]):
        if network == "mobile0.25":
            self.cfg = cfg_mnet
        elif network == "resnet50":
            self.cfg = cfg_re50
        model = RetinaFace(cfg=self.cfg, phase='test')
        self.model = load_model(model, pretrained, cpu)
        self.model.eval()
        print('Finished loading model!')

        cudnn.benchmark = True
        self.device = torch.device('cpu' if cpu else 'cuda')
        self.model = self.model.to(self.device)

        self.confidence_thresh = confidence_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.keep_top_k = keep_top_k
        self.mean = mean
        self.resize = 1

    def __call__(self, img, thresh=0.9):
        img = np.array(img).astype(np.float32)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= self.mean
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.model(img)  # forward pass

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_thresh)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_thresh)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :-1]
        landms = landms[keep]
        scores = scores[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]
        scores = scores[:self.keep_top_k]

        # keep score > 0.9
        dets = dets[scores > thresh]
        landms = landms[scores > thresh]
        scores = scores[scores > thresh]

        return dets, scores, landms.reshape(len(landms), 5, 2)

