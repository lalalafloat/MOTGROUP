import numpy as np
from collections import deque, defaultdict
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        # mean = x, y, a, h, vx, vy, va, vh
        # 
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        
        # 编队
        self.form_id = 0
        self.oid = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0  # 这一步的效果是啥我不是很清楚 得复习一下卡尔曼滤波
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):  # 然后预测完更新一下strack的mean 和 var
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))  # [ATTN] 这里

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0  # 相比update这一步有点差别
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id  # 更新最新的id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))  # 更新kf的mean and var
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):  # [ATTN] 后续的kf的tlwh都是来自于mean的预测结果
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
    _qcount = 0
    _qid2cnt = defaultdict(int)
    _tid2oid = {}

    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
    
    @staticmethod
    def next_qid():
        BYTETracker._qcount += 1
        return BYTETracker._qcount
    
    @staticmethod
    def get_oid(track):
        if track.track_id in BYTETracker._tid2oid.keys():
            track.oid =  BYTETracker._tid2oid[track.track_id]
        else:
            BYTETracker._qid2cnt[track.form_id] += 1
            BYTETracker._tid2oid[track.track_id] = BYTETracker._qid2cnt[track.form_id]
            track.oid = BYTETracker._tid2oid[track.track_id]
        return track

    def update(self, output_results, img_info, img_size):  # 在这里面实现编批
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))  # 要么短边或者长边被缩放了
        bboxes /= scale
        # print('scores.shape = {}'.format(scores.shape))  # [N]

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)  # 0.1 ~ track_thresh
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]  # >track_thresh
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:  # 已经跟踪的一些轨迹
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)  # is_activated=True会被加入tracked_stracks

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)  # 去重合并
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)  # 1 - iou
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        # matches = [N_match, 2] 哪些track匹配上哪些det
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]  # 注意这里用的不是全部的strack_pool了，而是u_track
        dists = matching.iou_distance(r_tracked_stracks, detections_second)  # 1 - iou
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:  # 两次跟踪都没跟上的结果就标记为lost
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # 编队 form_id
        eps = 1e-9
        form_stracks = [track for track in output_stracks if abs(track.mean[4]) + abs(track.mean[5]) > eps]
        last_stracks = [track for track in output_stracks if abs(track.mean[4]) + abs(track.mean[5]) <= eps]

        # print(len(last_stracks))
        dist_scale = 4.0  # 两个超参
        sim_thresh = 0.7  # 先筛选满足条件的 最后用 sim / dist选择最大的
        q_list = []
        for track in form_stracks:
            if len(q_list) == 0:
                q = [track]
                q_list.append(q)
                continue

            max_q = -1
            max_score = -100
            for qidx,q in enumerate(q_list):
                for qtrack in q:
                    dist = np.sqrt(np.square(track.mean[:2] - qtrack.mean[:2]).sum())
                    # print('dist = {}'.format(dist))
                    siz = max(track.mean[2] * track.mean[3], track.mean[3])
                    qvxy = qtrack.mean[4:6] / np.linalg.norm(qtrack.mean[4:6], ord=2, axis=0, keepdims=True)
                    vxy = track.mean[4:6] / np.linalg.norm(track.mean[4:6], ord=2, axis=0, keepdims=True)
                    sim = np.dot(qvxy, vxy)
                    # print(vxy.shape, sim.shape)
                    # print(sim)
                    if dist < dist_scale * siz and sim > sim_thresh:
                        score = sim / dist
                        if score > max_score:
                            max_q = qidx
                            max_score = score
            if max_q >= 0:
                q_list[max_q].append(track)
            else:
                q_list.append([track])
            # vxy = np.stack([_.mean[4:6] for _ in form_stracks])  # [N, 2]
            # vxy_norm = np.linalg.norm(vxy, ord=2, axis=1, keepdims=True)  # [N, 1]
            # vxyn = vxy / vxy_norm

            # sim = np.matmul(vxyn, vxyn.T)  # [N, 2] x [2, N] = [N, N]
            # print(sim.max(), sim.min())
            # cost = 1.0 - sim
            # e = np.eye(*list(sim.shape)) * 100
            # cost = cost + e  # 不和自己匹配
            # matches, u_match, _ = matching.linear_assignment(cost, thresh=0.5)
        # 给编队打标
        # output_stracks = last_stracks # [TODO] 这个last如何处理，如果加上则不能从0开始
        output_stracks = []
        for q in q_list:
            flag_dict = defaultdict(int)
            for qtrack in q:
                if qtrack.form_id > 0:
                    flag_dict[qtrack.form_id] += 1
            max_flag = 0
            max_count = 0
            for k,v in flag_dict.items():
                if v > max_count:
                    max_count = v
                    max_flag = k
            if max_flag == 0:
                max_flag = self.next_qid()
            for qtrack in q:
                qtrack.form_id = max_flag
                output_stracks.append(qtrack)
        # 更新队内id
        output_stracks = [self.get_oid(_) for _ in output_stracks]
        # ddd = defaultdict(list)
        # for _ in output_stracks:
        #     ddd[_.form_id].append(_.oid)
        # for k,v in ddd.items():
        #     v.sort()
        #     print("{}: {}".format(k, v))
        return output_stracks


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
