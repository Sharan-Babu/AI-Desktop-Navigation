import os
import sys
import cv2
import time
import autopy
import logging as log
from collections import namedtuple
from argparse import ArgumentParser
from inference import Network
import numpy as np

# shoppingInfo contains statistics for the shopping information
MyStruct = namedtuple("shoppingInfo", "shopper, looker")
INFO = MyStruct(0, 0)

POSE_CHECKED = False

TOPIC = "shopper_gaze_monitor"

# Global variables
TARGET_DEVICE = 'CPU'
accepted_devices = ['CPU', 'GPU', 'MYRIAD', 'HETERO:FPGA,CPU', 'HDDL']
is_async_mode = True
CONFIG_FILE = '../resources/config.json'

# Flag to control background thread
KEEP_RUNNING = True

DELAY = 5


def args_parser():
    """
    Parse command line arguments.
    :return: Command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", 
                        help="Path to an .xml file with a pre-trained"
                        "face detection model")
    parser.add_argument("-pm", "--posemodel",
                        help="Path to an .xml file with a pre-trained model"
                        "head pose model")
    parser.add_argument("-l", "--cpu_extension", type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers. Absolute "
                        "path to a shared library with the kernels impl.")
    parser.add_argument("-d", "--device", default="CPU", type=str,
                        help="Specify the target device to infer on; "
                        "CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. To run with multiple devices use"
                        " MULTI:<device1>,<device2>,etc. Application "
                        "will look for a suitable plugin for device specified"
                        "(CPU by default)")
    parser.add_argument("-c", "--confidence", default=0.5, type=float,
                        help="Probability threshold for detections filtering")
    parser.add_argument("-f", "--flag", help="sync or async", default="async", type=str)

    global TARGET_DEVICE, is_async_mode
    args = parser.parse_args()
    if args.device:
        TARGET_DEVICE = args.device
    if args.flag == "sync":
        is_async_mode = False
    else:
        is_async_mode = True
    return parser


def check_args():
    # ArgumentParser checks the device

    global TARGET_DEVICE
    if 'MULTI' not in TARGET_DEVICE and TARGET_DEVICE not in accepted_devices:
        print("Unsupported device: " + TARGET_DEVICE)
        sys.exit(1)
    elif 'MULTI' in TARGET_DEVICE:
        target_devices = TARGET_DEVICE.split(':')[1].split(',')
        for multi_device in target_devices:
            if multi_device not in accepted_devices:
                print("Unsupported device: " + TARGET_DEVICE)
                sys.exit(1)

def cart(x, y, w, h):
    u = int(w*(x+1)/2)

    v = int(h*(y+1)/2)
    
    return(u,v)

def face_detection(res, args, initial_wh):
    """
    Parse Face detection output.
    :param res: Detection results
    :param args: Parsed arguments
    :param initial_wh: Initial width and height of the FRAME
    :return: Co-ordinates of the detected face
    """
    global INFO
    faces = []
    INFO = INFO._replace(shopper=0)

    for obj in res[0][0]:
        # Draw only objects when probability more than specified threshold
        if obj[2] > args.confidence:
            if obj[3] < 0:
                obj[3] = -obj[3]
            if obj[4] < 0:
                obj[4] = -obj[4]
            xmin = int(obj[3] * initial_wh[0])
            ymin = int(obj[4] * initial_wh[1])
            xmax = int(obj[5] * initial_wh[0])
            ymax = int(obj[6] * initial_wh[1])
            faces.append([xmin, ymin, xmax, ymax])
            INFO = INFO._replace(shopper=len(faces))
    return faces

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    global INFO
    global DELAY
    global CLIENT
    global KEEP_RUNNING
    global POSE_CHECKED
    global TARGET_DEVICE
    global is_async_mode
    
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = args_parser().parse_args()
    logger = log.getLogger()
    check_args()

    input_stream = 0
    cap = cv2.VideoCapture(input_stream)

    if not cap.isOpened():
        logger.error("ERROR! Unable to open video source")
        return

    if input_stream:
        cap.open(input_stream)
        # Adjust DELAY to match the number of FPS of the video file
        DELAY = 1000 / cap.get(cv2.CAP_PROP_FPS)

    # Init inference request IDs
    cur_request_id = 0
    next_request_id = 1

    # Initialise the class
    infer_network = Network()
    infer_network_pose = Network()
    infer_network_landmarks = Network()
    infer_network_gaze = Network()
    # Load the network to IE plugin to get shape of input layer
    plugin, (n_fd, c_fd, h_fd, w_fd) = infer_network.load_model('./face_detection/face_detection.xml', TARGET_DEVICE, 1, 1, 2,
                                                                args.cpu_extension)
    n_hp, c_hp, h_hp, w_hp = infer_network_pose.load_model('./pose/pose.xml',
                                                           TARGET_DEVICE, 1,
                                                           3, 2,
                                                           args.cpu_extension, plugin)[1]
    
    n_fe, c_fe, h_fe, w_fe = infer_network_landmarks.load_model('./landmarks/landmarks.xml',
                                                           TARGET_DEVICE, 1,
                                                           1, 2,
                                                           args.cpu_extension, plugin)[1]
    infer_network_gaze.load_model('./gaze_estimation/gaze.xml',
                                                           TARGET_DEVICE, 3,
                                                           1, 2,
                                                           args.cpu_extension, plugin)
    
    if is_async_mode:
        print("Application running in async mode...")
    else:
        print("Application running in sync mode...")
    ret, frame = cap.read()
    current_x = 951
    current_y = 455
    while ret:

        ret, frame = cap.read()
        if not ret:
            KEEP_RUNNING = False
            break

        if frame is None:
            KEEP_RUNNING = False
            log.error("ERROR! blank FRAME grabbed")
            break

        initial_wh = [cap.get(3), cap.get(4)]
        in_frame_fd = cv2.resize(frame, (w_fd, h_fd))
        # Change data layout from HWC to CHW
        in_frame_fd = in_frame_fd.transpose((2, 0, 1))
        in_frame_fd = in_frame_fd.reshape((n_fd, c_fd, h_fd, w_fd))

        key_pressed = cv2.waitKey(int(DELAY))

        if is_async_mode:
            # Async enabled and only one video capture
            infer_network.exec_net(next_request_id, in_frame_fd)
        else:
            # Async disabled
            infer_network.exec_net(cur_request_id, in_frame_fd)
        # Wait for the result
        if infer_network.wait(cur_request_id) == 0:
            # Results of the output layer of the network
            res = infer_network.get_output(cur_request_id)
            # Parse face detection output
            faces = face_detection(res, args, initial_wh)
            if len(faces) == 1:
                # Look for poses
                for res_hp in faces:
                    xmin, ymin, xmax, ymax = res_hp
                    head_pose = frame[ymin:ymax, xmin:xmax]
                    in_frame_hp = cv2.resize(head_pose, (w_hp, h_hp))
                    in_frame_hp = in_frame_hp.transpose((2, 0, 1))
                    in_frame_hp = in_frame_hp.reshape((n_hp, c_hp, h_hp, w_hp))
                    
                    infer_network_pose.exec_net(cur_request_id, in_frame_hp)
                    infer_network_pose.wait(cur_request_id)
                    
                    # Parse outputs
                    angle_y_fc = infer_network_pose.get_output(0, "angle_y_fc")[0]
                    angle_p_fc = infer_network_pose.get_output(0, "angle_p_fc")[0]
                    angle_r_fc = infer_network_pose.get_output(0, "angle_r_fc")[0]
                    head_pose_angles = np.array([angle_y_fc , angle_p_fc, angle_r_fc], dtype = 'float32')
                    head_pose_angles = head_pose_angles.transpose()
                    
                    in_frame_eye = cv2.resize(head_pose, (w_fe, h_fe))
                    in_frame_eye = in_frame_eye.transpose((2, 0, 1))
                    in_frame_eye = in_frame_eye.reshape((n_fe, c_fe, h_fe, w_fe))
                    
                    infer_network_landmarks.exec_net(cur_request_id, in_frame_eye)
                    infer_network_landmarks.wait(cur_request_id)
                    
                    align_fc3 = infer_network_landmarks.get_output(0, "align_fc3")
                    align_fc3 = [align_fc3[0][n:n+2] for n in range(0, len(align_fc3[0]), 2)]
                    width = head_pose.shape[1]
                    height = head_pose.shape[0]
                    rx = xmin + int(align_fc3[12][0] * width)-15
                    ry = ymin + int(align_fc3[12][1] * height)
                    rx2 = xmin + int(align_fc3[14][0] * width)
                    ry2 = (ymin + int(align_fc3[12][1] * height)) + (rx2-rx-15)
                    frame = cv2.rectangle(frame, (rx,ry), (rx2,ry2), (255, 255, 255), 2)
                    right_eye = frame[ry:ry2, rx:rx2]
                    right_midpoint = (int((rx+rx2) / 2), int((ry+ry2) / 2))
                    in_frame_right_eye = cv2.resize(right_eye, (60, 60))
                    in_frame_right_eye = in_frame_right_eye.transpose((2, 0, 1))
                    in_frame_right_eye = in_frame_right_eye.reshape((1, 3, 60, 60))

                    lx = xmin + int(align_fc3[15][0] * width)
                    ly = ymin + int(align_fc3[15][1] * height)
                    lx2 = xmin + int(align_fc3[17][0] * width)+15
                    ly2 = (ymin + int(align_fc3[17][1] * height)) + (lx2-lx-15)
                    frame = cv2.rectangle(frame, (lx,ly), (lx2,ly2), (255, 255, 255), 2)
                    left_eye = frame[ly:ly2, lx:lx2]
                    left_midpoint = (int((lx+lx2) / 2), int((ly+ly2) / 2))
                    in_frame_left_eye = cv2.resize(left_eye, (60, 60))
                    in_frame_left_eye = in_frame_left_eye.transpose((2, 0, 1))
                    in_frame_left_eye = in_frame_left_eye.reshape((1, 3, 60, 60))
                    
                    infer_network_gaze.exec_net_g(cur_request_id, in_frame_left_eye, in_frame_right_eye, head_pose_angles)
                    infer_network_gaze.wait(cur_request_id)
                    
                    gaze_vector = infer_network_gaze.get_output(0, 'gaze_vector')
                    #length = sqrt(gaze_vector[0][0]^2 + gaze_vector[0][1]^2 + gaze_vector[0][2]^2)
                    norm = np.linalg.norm(gaze_vector)
                    gaze_vector = gaze_vector / norm
                    
                    arrow_length = int(0.4 * xmax-xmin)
                    gaze_arrow_left = (int(arrow_length * - gaze_vector[0][0] + left_midpoint[0]), int(arrow_length * gaze_vector[0][1] + left_midpoint[1]))
                    gaze_arrow_right = (int(arrow_length * -  gaze_vector[0][0] + right_midpoint[0]), int(arrow_length * gaze_vector[0][1] + right_midpoint[1]))
                    frame = cv2.arrowedLine(frame, left_midpoint, gaze_arrow_left, (0, 0, 0), 2)
                    frame = cv2.arrowedLine(frame, right_midpoint, gaze_arrow_right, (0, 0, 0), 2)
                    
                    #x_angle = float(180.0 / np.pi * (np.pi/2 + np.arctan2(gaze_vector[0][2], gaze_vector[0][0])))
                    #y_angle = float(180.0 / np.pi * (np.pi/2 - np.arccos(gaze_vector[0][1] / norm)))
                    
                    current_position = autopy.mouse.location()
                    x_new = (int(1 * gaze_vector[0][0] + current_position[0]), int(1 * - gaze_vector[0][1] + current_position[1]))
                    autopy.mouse.move(x_new[0], x_new[1])
                    #if left_midpoint> 380:
                        #pyautogui.moveTo(pyautogui.position()[0]-10,0,duration=1)
                    #else:
                        #pyautogui.moveTo(pyautogui.position()[0]-10,0,duration=1)
                        
                    """
                    # Top
                    if x_angle > 0 and y_angle > 0:
                        time.sleep(1)
                        current_x = current_x
                        current_y = current_y+5
                        pyautogui.moveTo(current_x,current_y,duration=1)
                    # Right
                    if x_angle < 0 and y_angle > 0: 
                        time.sleep(1)
                        current_x = current_x+5
                        current_y = current_y
                        pyautogui.moveTo(current_x,current_y,duration=1)
                    # Left
                    if x_angle > 10 and y_angle < 0:
                        time.sleep(1)
                        current_x = current_x - 5
                        current_y = current_y
                        pyautogui.moveTo(current_x,current_y,duration=1)
                    # Bottom
                    if x_angle < 10 and y_angle < -10:
                        time.sleep(1)
                        current_x = current_x
                        current_y = current_y-5
                        pyautogui.moveTo(current_x,current_y,duration=1)   
                    #pyautogui.drag(100,0, duration=2)
                    #pyautogui.position()
                    #pyautogui.moveTo(100,100,duration=2)
                    """
                    
                    # Top - 3,12                   0, 15
                    # left - 35,-10                35, 10
                    # right - -30,23               -30, 
                    # bottom - 2,-32
        cv2.imshow("Shopper Gaze Monitor", frame)

        if key_pressed == 27:
            print("Attempting to stop background threads")
            KEEP_RUNNING = False
            break

        if is_async_mode:
            # Swap infer request IDs
            cur_request_id, next_request_id = next_request_id, cur_request_id 

    infer_network.clean()
    infer_network_pose.clean()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    sys.exit()
    
    
