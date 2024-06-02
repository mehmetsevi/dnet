#!/usr/bin/env python
import os
import sys
import cv2
import datetime
import argparse
import numpy as np
from scipy.special import comb

CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".")
sys.path.append(os.path.join(CURRENT_DIR, "../../"))
try:
    from rail_marking.segmentation.deploy import RailtrackSegmentationHandler
    from cfg import BiSeNetV2Config
except Exception as e:
    print(e)
    sys.exit(0)

import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, moveBy, Landing
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
# RTSP Transport Protocol to UDP
DRONE_IP = "10.202.0.1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
# Kamera Açısı
myDrone = olympe.Drone(DRONE_IP)
myDrone.connect()
myDrone(olympe.messages.gimbal.set_target(
gimbal_id=0,
control_mode="position",
yaw_frame_of_reference="relative",
yaw=0.0,
pitch_frame_of_reference="relative",
pitch=-14.5,
roll_frame_of_reference="relative",
roll=0.0,
)).wait()

global for_back_velocity
global left_right_velocity
global up_down_velocity
global yaw_velocity

# args setting
parser2 = argparse.ArgumentParser(description='Process some integers.')
parser2.add_argument('-i', "--input", help="input file video")
parser2.add_argument('--leftPoint', type=int, help="Left rail offset", default=450)
parser2.add_argument('--rightPoint', type=int, help="Right rail offset", default=1540)
parser2.add_argument('--topPoint', type=int, help="Top rail offset", default=1)
parser2.add_argument("-snapshot", type=str, required=True)
args2 = parser2.parse_args()

width, height = 1280, 720

def main():
   
    segmentation_handler = RailtrackSegmentationHandler(args2.snapshot, BiSeNetV2Config())
    
    pid = [0.1, 1, 0]
    pError = 0
    ucus = 0
    takipS = 0
    
    capture = cv2.VideoCapture("rtsp://10.202.0.1/live", cv2.CAP_FFMPEG)

    # Set output frame height and width
    height2 = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # Set playback FPS to 24
    fps = 24

    fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
    writer = cv2.VideoWriter("out.avi", fourcc, fps, (width2, height2))
    # initialization for line detection
    expt_startLeft = args2.leftPoint
    expt_startRight = args2.rightPoint
    expt_startTop = args2.topPoint

    # value initialize
    left_maxpoint = [0] * 50
    right_maxpoint = [500] * 50

    # convolution filter
    kernel = np.array([
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1]
    ])

    # Next frame availability
    r = True
    first = True
    forward_speed = 0.01
    turn_speed = 0.01    
    
    if not capture.isOpened():
        raise Exception("failed to open {}".format(args2.video_path))

    _total_ms = 0
    count_frame = 0
    #myDrone(TakeOff() >> FlyingStateChanged(state="hovering", _timeout=5)).wait().success()
    #status = myDrone.start_piloting()
    while capture.isOpened():
        ret, frame = capture.read()
        count_frame += 1

        if not ret:
            break

        start = datetime.datetime.now()
        _, overlay = segmentation_handler.run(frame, only_mask=False)
        _total_ms += (datetime.datetime.now() - start).total_seconds() * 1000
        overlay = cv2.resize(overlay, (width, height))

        cv2.imshow("result", overlay)
        #out_video.write(overlay)
   
        crop_img = frame[100:550, 520:840]
        crop_overlay = overlay[100:550, 520:840]
        
        intersection = np.logical_and(crop_img, crop_overlay)
        union = np.logical_or(crop_img, crop_overlay)
        iou_score = np.sum(intersection) / np.sum(crop_overlay)    
    
        print(" intersection: ")
        print(np.sum(intersection))
        print(" union: ")
        print(np.sum(union))
        print(" iou_score: ")
        print(iou_score)
    
        cv2.imshow('frame orj',frame)  
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray orj',gray)  
        blur = cv2.GaussianBlur(gray,(5,5),0)
        cv2.imshow('blur orj',blur)  
        ret,thresh = cv2.threshold(blur,60,255,cv2.THRESH_BINARY_INV)
        cv2.imshow('thresh orj',thresh)  
        contours,hierarchy = cv2.findContours(thresh.copy(), 1, cv2.CHAIN_APPROX_NONE)
        
            # Find the biggest contour (if detected)
      
        if len(contours) > 0:

            c = max(contours, key=cv2.contourArea)

            M = cv2.moments(c)

            if M['m00'] == 0:
                M['m00'] = 1

            cx = int(M['m10']/M['m00'])

            cy = int(M['m01']/M['m00'])

            cv2.line(crop_img,(cx,0),(cx,720),(255,0,0),1)

            cv2.line(crop_img,(0,cy),(1280,cy),(255,0,0),1)

            cv2.drawContours(crop_img, contours, -1, (0,255,0), 1)

            if cx >= 200:
                print("Turn Right!")
                #myDrone.piloting_pcmd(3, 0, 0, 0, 0.1)

            if cx < 200 and cx > 130:
                print("On Track!")
                #myDrone.piloting_pcmd(0, 20, 0, 0, 0.1)

            if cx <= 130:

                print("Turn Left")
                #myDrone.piloting_pcmd(-3, 0, 0, 0, 0.1)
     
        else:

            print("I don't see the line")
        
        cv2.imshow('frame',crop_img)  
        
        #if takipS == 1:
           #print("Takip Ediliyor")
           #pError = takipEt(myDrone, overlay, width, pid, pError)

        #if takipS == 0:
           #print("Takip Bırakıldı")
            
        # cut away invalid frame area
        valid_frame = frame[expt_startTop:, expt_startLeft:expt_startRight]
        # original_frame = valid_frame.copy()

        # gray scale transform
        gray_frame = cv2.cvtColor(valid_frame, cv2.COLOR_BGR2GRAY)

        # histogram equalization image
        histeqaul_frame = cv2.equalizeHist(gray_frame)

        # apply gaussian blur
        blur_frame = cv2.GaussianBlur(histeqaul_frame, (5, 5), 5)

        # merge current frame and last frame
        if first is True:
            merge_frame = blur_frame
            first = False
            old_valid_frame = merge_frame.copy()
        else:
            merge_frame = cv2.addWeighted(blur_frame, 0.2, old_valid_frame, 0.8, 0)
            old_valid_frame = merge_frame.copy()

        # convolution filter
        conv_frame = cv2.filter2D(merge_frame, -1, kernel)

        # initialization for sliding window property
        sliding_window = [100, 270, 730, 900]
        slide_interval = 15
        slide_height = 60
        slide_width = 60
        # initialization for bezier curve variables
        left_points = []
        right_points = []

        # define count value
        count = 0
        for i in range(540, 240, -slide_interval):
            # get edges in sliding window
            left_edge = conv_frame[i:i + slide_height, sliding_window[0]:sliding_window[1]].sum(axis=0)
            right_edge = conv_frame[i:i + slide_height, sliding_window[2]:sliding_window[3]].sum(axis=0)

            # left railroad line processing
            if left_edge.argmax() > 0:
                left_maxindex = sliding_window[0] + left_edge.argmax()
                left_maxpoint[count] = left_maxindex
                cv2.line(valid_frame, (left_maxindex, i + int(slide_height / 2)),
                         (left_maxindex, i + int(slide_height / 2)), (255, 255, 255), 5, cv2.LINE_AA)
                left_points.append([left_maxindex, i + int(slide_height / 2)])
                sliding_window[0] = max(0, left_maxindex - int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                sliding_window[1] = min(1000, left_maxindex + int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                cv2.rectangle(valid_frame, (sliding_window[0], i + slide_height), (sliding_window[1], i), (0, 255, 0),
                              1)

            # right railroad line processing
            if right_edge.argmax() > 0:
                right_maxindex = sliding_window[2] + right_edge.argmax()
                right_maxpoint[count] = right_maxindex
                cv2.line(valid_frame, (right_maxindex, i + int(slide_height / 2)),
                         (right_maxindex, i + int(slide_height / 2)), (255, 255, 255), 5, cv2.LINE_AA)
                right_points.append([right_maxindex, i + int(slide_height / 2)])
                sliding_window[2] = max(0, right_maxindex - int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                sliding_window[3] = min(1000, right_maxindex + int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                cv2.rectangle(valid_frame, (sliding_window[2], i + slide_height), (sliding_window[3], i), (0, 0, 255),
                              1)
            count += 1

        # bezier curve process
        
        import matplotlib.pyplot as plt

        t, bezier_left_xval, bezier_left_yval = bezier_curve(left_points, 5)
        t, bezier_right_xval, bezier_right_yval = bezier_curve(right_points, 5)

            
        plt.plot(bezier_left_xval, bezier_left_yval, color='red')
        plt.plot(bezier_right_xval, bezier_right_yval, color='blue')
        
        
        bezier_left_points = []
        bezier_right_points = []
        try:
            old_point = (bezier_left_xval[0], bezier_left_yval[0])
            for point in zip(bezier_left_xval, bezier_left_yval):
                cv2.line(valid_frame, old_point, point, (0, 0, 255), 2, cv2.LINE_AA)
                old_point = point
                bezier_left_points.append(point)

            old_point = (bezier_right_xval[0], bezier_right_yval[0])
            for point in zip(bezier_right_xval, bezier_right_yval):
                cv2.line(valid_frame, old_point, point, (255, 0, 0), 2, cv2.LINE_AA)
                old_point = point
                bezier_right_points.append(point)
        except IndexError:
            pass
        '''
        cv2.imshow('frame', np.vstack([
            np.hstack([valid_frame,
                       original_frame,
                       cv2.cvtColor(histeqaul_frame, cv2.COLOR_GRAY2BGR)]),
            np.hstack([cv2.cvtColor(blur_frame, cv2.COLOR_GRAY2BGR),
                       cv2.cvtColor(merge_frame, cv2.COLOR_GRAY2BGR),
                       cv2.cvtColor(conv_frame, cv2.COLOR_GRAY2BGR)])
        ]))
        '''
        # Calculate the midpoint of the bezier curves
        mid_x = int((bezier_left_xval[-1] + 100 + bezier_right_xval[-1]) / 2)
        mid_y = int((bezier_left_yval[-1] + bezier_right_yval[-1]) / 2)

		
		# Orta nokta ile drone hareketi kontrolü
        if mid_x >= 800:  # Sağ
            myDrone(moveBy(0, 0.1, 0, -0.1)).wait()
        elif mid_x < 800 and mid_x > 480:  # Düz
            myDrone(moveBy(0.1, 0, 0, 0)).wait()
        elif mid_x <= 480:  # Sol
            myDrone(moveBy(0, -0.1, 0, 0.1)).wait()
        plt.scatter(mid_x, mid_y, color='green')

        plt.legend()
        plt.title('Bezier Curves')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()
        plt.title('Red: Left Rail   Blue: Right Rail    Green: Midpoint')
        plt.savefig('bezier_curves_plot.png')

        # Show the plot
        plt.show()
        # Display the midpoint on the frame
        cv2.circle(overlay, (mid_x, mid_y), 15, (0, 255, 0), -1)

        
        cv2.imshow('Video overlay', overlay)
        
        key = cv2.waitKey(1) & 0xff
        if key == 27:  # ESC
            break
        elif key == ord('w'):
            myDrone.piloting_pcmd(0, 20, 0, 0, 0.1)
        elif key == ord('t'):
            print("Takip...")
            takipS = 1    
            

# bezier curve function
def bezier_curve(points, ntimes=1000):

    def bernstein_poly(i, n, t):

        return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, ntimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    return t, xvals.astype('int32'), yvals.astype('int32')

def nothing(value):
    pass

if __name__ == "__main__":
    main()

