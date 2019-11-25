#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
from timeit import default_timer as timer  ### to calculate FPS
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

def main(yolo):

   # Definition of the parameters
    #max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

   # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    #metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    #tracker = Tracker(metric)
    video_path ="C:/Users/Rashid Ali/Desktop/elevator_yolov3/elevator_dataset/V02.mp4"
    writeVideo_flag = True

    video_capture = cv2.VideoCapture(video_path)

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        video_width = int(video_capture.get(3))
        video_height = int(video_capture.get(4))
        video_fps = int(video_capture.get(5))
        video_size   = (int(video_width), int(video_height))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter('output_V04.mp4', fourcc, video_fps, video_size)
        list_file = open('detection_v4 .txt', 'w')
        frame_index = -1

    fps = 0.0

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()
       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image)
       # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        #tracker.predict()
        #tracker.update(detections)
        '''
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
        '''
        

        person_count=0
        count1=0
        for det in detections:
            person_count = person_count+1
            count = '{} {:.1f}'.format('Count', person_count)
            count1 = '{} {:.1f}'.format('Total Count', person_count)
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(247, 7, 7), 2)
            cv2.putText(frame, str(count), (int(bbox[0]), int(bbox[1])), fontFace=cv2.FONT_HERSHEY_COMPLEX, 
                        fontScale=0.5, color=(247, 7, 7), thickness=2)
        cv2.putText(frame, str(count1), org=(20, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(247, 7, 7), thickness=2)
        cv2.putText(frame, '{:.2f}ms'.format((time.time() -t1) * 1000), (20, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(247, 7, 7), thickness=2)
        #cv2.namedWindow("Detections Window", cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Detections Window', frame)

        

        if writeVideo_flag:
            # save a frame
            print("Writing detections in file")
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write(str(person_count)+'\n')

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
