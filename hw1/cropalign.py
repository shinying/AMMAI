"""Face aligner and detector
"""

import cv2
import numpy as np


class FaceAligner:
    
    def __init__(self, desired_left_eye=(0.35, 0.45), desired_face_size=256):
        self.desired_left_eye = desired_left_eye
        self.desired_face_size = desired_face_size
        
    def align(self, image, landmarks):
        """Args:
        image (np.array)
        landmarks (np.array): [left eye, right eye, nose,
                               left mouth corner, right mouth corner]
        """
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        dX, dY = right_eye - left_eye
        angle = np.degrees(np.arctan2(dY, dX))
        
        desired_right_eyeX = 1. - self.desired_left_eye[0]
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = (desired_right_eyeX - self.desired_left_eye[0])
        desired_dist *= self.desired_face_size
        scale = desired_dist / dist
        
        eyes_center = (left_eye + right_eye) // 2
        M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)
        
        tX = self.desired_face_size * 0.5
        tY = self.desired_face_size * self.desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])
        
        (w, h) = (self.desired_face_size, self.desired_face_size)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        
        return output
    

class FaceDetector:
    
    def __init__(self, device, network='RetinaFace', pretrained=None, crop_size=256, thresh=0.9, **kwargs):
        self.network = network
        if network == 'RetinaFace':
            from retinaface import Detector
            self.model = Detector(pretrained=pretrained)
        elif network == 'MTCNN':
            from facenet import MTCNN
            self.model = MTCNN(device, **kwargs)
        else:
            raise NotImplementedError("Parameter 'network' only supports 'RetinaFace' and 'MTCNN'")

        self.aligner = FaceAligner(desired_face_size=crop_size)
        self.thresh = thresh
        
    def __call__(self, img, keep_all=False, select='center'):
        if self.network == 'RetinaFace':
            bbox, prob, landmarks = self.model(img)
        elif self.network == 'MTCNN':
            bbox, prob, landmarks = self.model.detect(img, landmarks=True)
            
        if len(bbox) == 0:
            return None, False
        
        is_multi = len(bbox[prob > self.thresh]) > 1
        if keep_all:
            aligned_faces = [self.aligner.align(np.array(img), landmark) \
                for landmark in landmarks[prob>self.thresh]]
            return aligned_faces, is_multi
        else:
            if select == 'max':
                index = prob.argmax()
            elif select == 'center':
                center = np.array(img.size) // 2
                dist = [((np.array([box[0]+box[2], box[1]+box[3]])/2 - center)**2).sum() for box in bbox]
                index = np.argmin(dist)
            else:
                raise NotImplementedError()
            aligned_face = self.aligner.align(np.array(img), landmarks[index])

            return aligned_face, is_multi