import numpy as np
from collections import deque
import os
import os.path as osp
import uuid
import copy
import torch
import torch.nn.functional as F
import cv2
import base64
from datetime import datetime

from yolov5.utils.general import xywh2xyxy, xyxy2xywh

from trackers.bytetrack.kalman_filter import KalmanFilter
from trackers.bytetrack import matching
from trackers.bytetrack.basetrack import BaseTrack, TrackState

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, cls):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.trajectory = []

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def extract_image_patch(image, bbox):
        bbox = np.expand_dims(bbox, axis=0)
        bbox = xywh2xyxy(bbox)[0]
        bbox = bbox.astype(np.int)
        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        return image

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id, img):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        # self.track_id = self.next_id()
        self.track_id = str(uuid.uuid4())
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        
        img_crop = self.extract_image_patch(img, self.tlwh)
        if img_crop is not None:
            retval, buffer = cv2.imencode('.jpg', img_crop, encode_param)
            image_string = base64.b64encode(buffer).decode('utf-8')
        else:
            image_string = None
        self.best_image = image_string
        self.large_image = image_string

        # set max score, max size, append first trajectory entry
        self.max_score = self.score
        self.max_size = self._tlwh[2]*self._tlwh[3]
        self.trajectory.append(self._tlwh.tolist())

        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('-------- new track activated max_score=%s, max_size=%s' % (str(self.max_score), str(self.max_size)))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, img, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            # self.track_id = self.next_id()
            self.track_id = str(uuid.uuid4())
        if new_track.score > self.max_score:
            img_crop = self.extract_image_patch(img, new_track.tlwh)
            if img_crop is not None:
                retval, buffer = cv2.imencode('.jpg', img_crop, encode_param)
                image_string = base64.b64encode(buffer).decode('utf-8')
            else:
                image_string = None
            self.best_image = image_string
            self.max_score = new_track.score
            # print('!!!!!!!! track reactivated, new score: %s' % str(self.max_score))
        new_size = (max(0, new_track.tlwh[0]) + new_track.tlwh[2]) * (max(0, new_track.tlwh[1]) + new_track.tlwh[3])
        # print('------- new frame max_size %s, latest size %s' % (str(self.max_size), str(new_size)))
        if new_size > self.max_size:
            img_crop = self.extract_image_patch(img, new_track.tlwh)
            if img_crop is not None:
                retval, buffer = cv2.imencode('.jpg', img_crop, encode_param)
                image_string = base64.b64encode(buffer).decode('utf-8')
            else:
                image_string = None
            self.large_image = image_string
            self.max_size = new_size
            # print('!!!!!!!! track reactivated, new size: %s' % str(self.max_size))
        self.trajectory.append(self._tlwh.tolist())

        self.score = new_track.score
        self.cls = new_track.cls

    def update(self, new_track, frame_id, img):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        # self.cls = cls

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        # print('------- new frame max_score %s, latest score %s' % (str(self.max_score), str(new_track.score)))
        if new_track.score > self.max_score:
            img_crop = self.extract_image_patch(img, new_track.tlwh)

            if img_crop is not None:
                retval, buffer = cv2.imencode('.jpg', img_crop, encode_param)
                image_string = base64.b64encode(buffer).decode('utf-8')
            else:
                image_string = None
            
            self.best_image = image_string
            self.max_score = new_track.score
            # print('!!!!!!!! track updated, new score: %s' % str(self.max_score))

        new_size = (max(0, new_track.tlwh[0]) + new_track.tlwh[2]) * (max(0, new_track.tlwh[1]) + new_track.tlwh[3])
        # print('------- new frame max_size %s, latest size %s' % (str(self.max_size), str(new_size)))
        if new_size > self.max_size:
            img_crop = self.extract_image_patch(img, new_track.tlwh)
            if img_crop is not None:
                retval, buffer = cv2.imencode('.jpg', img_crop, encode_param)
                image_string = base64.b64encode(buffer).decode('utf-8')
            else:
                image_string = None
            self.large_image = image_string
            self.max_size = new_size
            # print('!!!!!!!! track updated, new size: %s' % str(self.max_size))
        self.trajectory.append(self._tlwh.tolist())

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, track_thresh=0.45, track_buffer=25, match_thresh=0.8, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.track_buffer=track_buffer
        
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        # self.det_thresh = track_thresh
        self.det_thresh = track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, dets, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        xyxys = dets[:, 0:4]
        xywh = xyxy2xywh(xyxys)
        confs = dets[:, 4]
        clss = dets[:, 5]
        
        classes = clss.numpy()
        xyxys = xyxys.numpy()
        confs = confs.numpy()

        remain_inds = confs > self.track_thresh
        inds_low = confs > 0.1
        inds_high = confs < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        
        dets_second = xywh[inds_second]
        dets = xywh[remain_inds]
        
        scores_keep = confs[remain_inds]
        scores_second = confs[inds_second]
        
        clss_keep = classes[remain_inds]
        clss_second = classes[remain_inds]
        

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores_keep, clss_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        #if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id, img)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, img, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(xywh, s, c) for (xywh, s, c) in zip(dets_second, scores_second, clss_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, img)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, img, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        #if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, img)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
            # print('####')

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id, img)
            activated_starcks.append(track)


        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
                # print('****')
                # print('lost_stracks len')
                # print(len(self.lost_stracks))

        # print('Remained match {} s'.format(t4-t3))
        # print(len(self.lost_stracks), len(self.tracked_stracks), len(lost_stracks), len(lost_stracks))
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        # huge bug fixed here, plz first extend strack then substract from lost
        self.removed_stracks.extend(removed_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        outputs = []
        for t in output_stracks:
            output= []
            tlwh = t.tlwh
            tid = t.track_id
            tlwh = np.expand_dims(tlwh, axis=0)
            xyxy = xywh2xyxy(tlwh)
            xyxy = np.squeeze(xyxy, axis=0)
            # trajectory = t.trajectory
            output.extend(xyxy)
            output.append(tid)
            output.append(t.cls)
            # output.append(trajectory)
            outputs.append(output)
        if len(removed_stracks) > 0:
            # assert removed_stracks[0].large_image == removed_stracks[0].best_image
            if removed_stracks[0].large_image == removed_stracks[0].best_image:
                print('!!!!!!!!!!!!!!!!!!')
            print(dets[0])

        return [outputs, removed_stracks]


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
