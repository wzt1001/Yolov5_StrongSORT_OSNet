import argparse

import os
import sys
import threading
import time
from uuid import uuid4
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import json

import numpy as np
import boto3
from pathlib import Path
import yaml
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn

from awscrt import mqtt

# import utils.command_line_utils as command_line_utils
# cmdUtils = command_line_utils.CommandLineUtils("PubSub - Send and recieve messages through an MQTT connection.")
# cmdUtils.add_common_mqtt_commands()
# cmdUtils.add_common_topic_message_commands()
# cmdUtils.add_common_proxy_commands()
# cmdUtils.add_common_logging_commands()
# cmdUtils.register_command("key", "<path>", "Path to your key in PEM format.", True, str)
# cmdUtils.register_command("cert", "<path>", "Path to your client certificate in PEM format.", True, str)
# cmdUtils.register_command("port", "<int>", "Connection port. AWS IoT supports 443 and 8883 (optional, default=auto).", type=int)
# cmdUtils.register_command("client_id", "<str>", "Client ID to use for MQTT connection (optional, default='test-*').", default="test-" + str(uuid4()))
# cmdUtils.register_command("count", "<int>", "The number of messages to send (optional, default='10').", default=10, type=int)
# cmdUtils.register_command("is_ci", "<str>", "If present the sample will run in CI mode (optional, default='None')")
# # Needs to be called so the command utils parse the commands
# cmdUtils.get_args()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker

# remove duplicated stream handler to avoid duplicated logging
#logging.getLogger().removeHandler(logging.getLogger().handlers[0])

# by zwang
with open("config.yaml", "r") as yaml_file:
    try:
        LOGGER.info('loading config yaml...')
        config = yaml.safe_load(yaml_file)
    except yaml.YAMLError as exc:
        LOGGER.error('config.yaml not exist', exc)
        assert 0, 'exiting due to the absense of config.yaml'

# by zwang, initiate boto3
s3 = boto3.client('s3', region_name=config['reid_settings']['region'])
                  # aws_access_key_id=CN_S3_AKI, aws_secret_access_key=CN_S3_SAK
BUCKET_NAME = config['reid_settings']['s3_bucket']
S3_LOCATION = config['reid_settings']['s3_location']
RESULT_ENDPOINT = config['reid_settings']['result_endpoint']
LOCAL_TEMP_PATH = config['reid_settings']['local_temp_path']
KINESIS_STREAM_ID = config['reid_settings']['kinesis_stream_id']

# by zwang, read from 

kinesis_client=boto3.client('kinesis')
npy_file = []

# by zwang, upload files to s3
def upload_files(path_local, path_s3):
    """
    :param path_local: local path
    :param path_s3: s3 path
    """
    LOGGER.info(path_local)
    LOGGER.info(path_s3)
    if not upload_single_file(path_local, path_s3):
        LOGGER.error(f'Upload files failed.')
 
    LOGGER.info(f'Upload files successful.')


def upload_single_file(src_local_path, dest_s3_path):
    """
    :param src_local_path:
    :param dest_s3_path:
    :return:
    """
    try:
        with open(src_local_path, 'rb') as f:
            s3.upload_fileobj(f, BUCKET_NAME, dest_s3_path)
    except Exception as e:
        LOGGER.error(f'Upload data failed. | src: {src_local_path} | dest: {dest_s3_path} | Exception: {e}')
        return False
    LOGGER.info(f'Uploading file successful. | src: {src_local_path} | dest: {dest_s3_path}')
    return True


# to do!!!! ascyronize the function to avoid blockage
def put_to_kinesis(track_id, max_score, max_size, best_image, large_image, frame_idx, current_time):
    kinesis_client.put_record(StreamName=KINESIS_STREAM_ID, Data=json.dumps({
        'track_id': track_id, 
        'max_score': str(max_score), 
        'max_size': str(max_size), 
        'best_image': best_image, 
        'large_image': large_image,
        'frame_idx': frame_idx,
        'current_time': current_time
    }), PartitionKey="partitionkey")

track_data = {}

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride,
        save_to_numpy_sample=False
):
    source = str(source)
    # save_img = not nosave and not source.endswith('.txt')  # save inference images
    
    save_img = False
    # is_file = Path(source).suffix[1:] in (VID_FORMATS)
    
    with open(source, 'r') as source_file:
        data = source_file.read()
    source_config = json.loads(data)
    
    if source_config['type'] == 'stream':
        webcam = True
        is_url = True
        is_file = False
    elif source_config['type'] == 'video_file':
        webcam = False
        is_url = False
        is_file = True
    else:
        assert "source type not supported"

    # is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    # if is_url and is_file:
    #     source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    print(source_config)
    # Dataloader
    if webcam:
        show_vid = check_imshow()
        dataset = LoadStreams([i['url'] for i in source_config['source']], img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages([i['url'] for i in source_config['source']], img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1

    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(nr_sources):
        tracker = create_tracker(tracking_method, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * nr_sources

    # Run tracking
    #model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    
    total_frame_cnt = 0

    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        
        total_frame_cnt += 1
        print('====== %s' % str(total_frame_cnt))
        # LOGGER.info(tracker_list)

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections, i is no. of cameras and det is dets from an single camera
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # xyxy

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i], removed_tracks = tracker_list[i].update(det.cpu(), im0)
                # outputs[0] = [[[x0, y0, x1, y1], tid, t.cls], ...]
                # LOGGER.info(outputs[i].tracker)

                # zwang, save detection results
                for out in outputs[i]:
                    if out[4] in track_data.keys():
                        # xyxy, frame id/timestamp, camera id
                        track_data[out[4]].append([out[0:4], frame_idx, i])
                    else:
                        # initiate new track
                        track_data[out[4]] = [[out[0:4], frame_idx, i]]
                        filepath = os.path.join(LOCAL_TEMP_PATH, f'{out[4]}.jpg')
                        save_one_box(out[0:4], imc, file=Path(filepath), BGR=True)
                        # path_s3 = os.path.join(S3_LOCATION, f'{out[4]}.jpg')
                        # upload_files(filepath, path_s3)

                # zwang, send track to endpoint when removed
                for t in removed_tracks:
                    print('------------ removed')
                    # {'ShardId': 'shardId-000000000001', 'SequenceNumber': '49637551416675302900361177921833649890934530126563508242', 'ResponseMetadata': {'RequestId': 'd5e21bdb-dd6d-455a-89b6-2cc9042594c0', 'HTTPStatusCode': 200, 'HTTPHeaders': {'x-amzn-requestid': 'd5e21bdb-dd6d-455a-89b6-2cc9042594c0', 'x-amz-id-2': 'FkjzFUpmWGtTPoPOf2Eg1ZpCQgoX8u/bOHrRyROf3OCndHWlbaT9lgYvBF+tRPyNNm072ZrvmVTOEf2Kee+j9FN4x+TShPKA', 'date': 'Wed, 01 Feb 2023 09:45:54 GMT', 'content-type': 'application/x-amz-json-1.1', 'content-length': '110'}, 'RetryAttempts': 0}}
                    # put_to_kinesis(track_id=t.track_id, max_score=str(t.max_score), max_size=str(t.max_size),
                    #                  best_image=t.best_image, large_image=t.large_image, frame_idx=frame_idx, 
                    #                  start_time=t.start_time, end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    npy_file.append({'track_id':t.track_id, 'max_score':str(t.max_score), 'max_size':str(t.max_size),
                                     'best_image':t.best_image, 'large_image':t.large_image, 'frame_idx':frame_idx, 
                                     'start_time': t.start_time, 'end_time':datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

                    # print(t.track_id)
                    # print(t.best_image)
                    # print(t.large_image)
                    # print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    # print(t.best_image)
                    # print(t.large_image)
                    # print(track_data)
                    # removed_track = track_data.pop(t.track_id)

                    ################################
                    # add post function here/zwang #
                    ################################

                    # r = requests.post(RESULT_ENDPOINT, data={'number': '12524', 'type': 'issue', 'action': 'show'})
                    # LOGGER.info(track_data)

                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], det[:, 4])):
    
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = str(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            annotator.box_label(bboxes, label, color=colors(c, True))

                # LOGGER.info(f'{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)')

            else:
                #strongsort_list[i].increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

    # save all results to a numpy file in root
    if save_to_numpy_sample:
        np.save("sample.npy", npy_file)

    # Print results
    # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_vid:
    #     s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # if update:
    #     strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=30, help='video frame-rate stride')
    parser.add_argument('--save-to-numpy-sample', default=False, action='store_true', help='save tracks to a numpy file for further testing')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
