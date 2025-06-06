#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python live_test.py --model best_model.pth --depth 16.9 --jitter 0.1  --rx 0 --ry -0.2 --rz -0.2 --jry 0.1 --jrz 0.1 --axis y --length 0.05 --speed 0.02 --auto_step --once

"""

import argparse
import math
import random
import signal
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import setting
import find_marker
import A_utility
from flow_processor      import FlowFeatureExtractor
from multimodal_network  import MultimodalFusionNet
from rtde_control        import RTDEControlInterface
from rtde_receive        import RTDEReceiveInterface

ROBOT_IP, RTDE_PORT  = "10.10.10.1", 50002
HOME_WITH_TOOL       = [-0.298569, -0.694446, 0.239335,
                        0.633457, -1.477861, 0.626266]
NEW_TCP              = (0, 0, 0.26, 0, 0, 0)
ROT                  = math.radians(45)
X_RANGE, Y_RANGE     = (-0.04, 0.04), (-0.04, 0.04)  # hover jitter
FLUSH_FRAMES         = 20

_rot   = lambda x, y, t: (math.cos(t)*x - math.sin(t)*y,
                           math.sin(t)*x + math.cos(t)*y)
old2new = lambda p: [*_rot(p[0], p[1], -ROT), *p[2:]]
new2old = lambda p: [*_rot(p[0], p[1],  ROT), *p[2:]]
class LiveTester:
    def __init__(self, a):
        self.a = a
        self.rtde_c = RTDEControlInterface(ROBOT_IP, RTDE_PORT)
        self.rtde_r = RTDEReceiveInterface(ROBOT_IP)
        self.rtde_c.setTcp(NEW_TCP)
        self.cam = cv2.VideoCapture(a.cam, cv2.CAP_V4L2)
        if not self.cam.isOpened():
            raise RuntimeError("Camera not found — check --cam index")
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,  800)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        setting.init()
        self.matcher = find_marker.Matching(
            N_=setting.N_, M_=setting.M_, fps_=setting.fps_,
            x0_=setting.x0_, y0_=setting.y0_,
            dx_=setting.dx_, dy_=setting.dy_)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(a.model, map_location=self.device)
        self.model = MultimodalFusionNet(num_classes=a.num_classes)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

        self.flow_extractor = FlowFeatureExtractor()
        self.img_tf = T.Compose([
            T.ToPILImage(),
            T.Resize((a.img_size, a.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.seq_len   = a.sequence_length
        self.home_new  = old2new(HOME_WITH_TOOL)

        self.base_rx, self.base_ry, self.base_rz = HOME_WITH_TOOL[3:]
        self.rx_off = math.radians(a.rx)
        self.ry_off = math.radians(a.ry)
        self.rz_off = math.radians(a.rz)
        self.jrx    = math.radians(a.jrx)
        self.jry    = math.radians(a.jry)
        self.jrz    = math.radians(a.jrz)

        signal.signal(signal.SIGINT, self._sig_exit)
    def _sig_exit(self, *_):
        print("\n[ABORT] Ctrl-C — stopping robot.")
        try:
            self.rtde_c.stopScript()
        finally:
            self.cam.release()
            cv2.destroyAllWindows()
            exit(1)

    def _flush_cam(self):
        for _ in range(FLUSH_FRAMES):
            self.cam.grab()
            _ = self.cam.retrieve()

    def _hover_rand(self):
        dx = random.uniform(*X_RANGE)
        dy = random.uniform(*Y_RANGE)
        p  = list(self.home_new)
        p[0] += dx; p[1] += dy
        return p

    def acquire_sequence(self):
        """Collect one press-slide sequence with all jitters."""
        a     = self.a
        rtde  = self.rtde_c
        steps = self.seq_len - 1
        step  = a.length / steps if a.auto_step else a.step

        rtde.moveL(HOME_WITH_TOOL, 0.25, 0.15)
        time.sleep(0.05)
        hover = self._hover_rand()

        rx = self.base_rx + self.rx_off + random.uniform(-self.jrx, self.jrx)
        ry = self.base_ry + self.ry_off + random.uniform(-self.jry, self.jry)
        rz = self.base_rz + self.rz_off + random.uniform(-self.jrz, self.jrz)
        pose_hover_old = new2old(hover[:3] + [rx, ry, rz])
        rtde.moveL(pose_hover_old, 0.20, 0.12)
        time.sleep(0.05)

        depth_mm = min(a.depth + random.uniform(-a.jitter, a.jitter),
                       a.max_depth)
        tgt = hover.copy()
        tgt[2] -= depth_mm / 1000.0
        rtde.moveL(new2old(tgt), 0.04, 0.03)
        time.sleep(0.12)

        self._flush_cam()
        fr0, flow0 = self._capture_first_valid()
        frames = [fr0]
        flows  = [np.array(flow0)]           
        ax   = 0 if a.axis == "x" else 1
        dirc = random.choice([1, -1])
        travelled = 0.0
        tgt_slide = tgt.copy()

        for _ in range(steps):
            travelled += step
            tgt_slide[ax] = tgt[ax] + dirc * travelled
            rtde.moveL(new2old(tgt_slide), a.speed, a.speed)
            time.sleep(0.04)

            fr  = A_utility.get_processed_frame(self.cam)
            cen = A_utility.marker_center(fr, debug=False)
            if len(cen) == setting.N_ * setting.M_:
                self.matcher.init(cen)
                self.matcher.run()
                flow = self.matcher.get_flow()
            else:
                print("[WARN] marker miss → reuse last flow")
                flow = flows[-1]

            frames.append(fr)
            flows.append(np.array(flow))      # <-- wrap as np.array

            cv2.imshow("GelSight", A_utility.draw_flow(fr.copy(), flow))
            if cv2.waitKey(1) == 27:
                return None

        # --- 7) retract & flush -------------------------------------------
        rtde.moveL(pose_hover_old, a.speed, a.speed)
        time.sleep(0.08)
        self._flush_cam()
        return frames, flows

    # ------------------------------------------------------------------
    def _capture_first_valid(self, attempts=20):
        for _ in range(attempts):
            fr = A_utility.get_processed_frame(self.cam)
            cen = A_utility.marker_center(fr, debug=False)
            if len(cen) == setting.N_ * setting.M_:
                self.matcher.init(cen)
                self.matcher.run()
                return fr, self.matcher.get_flow()
        raise RuntimeError("Marker grid not detected on first frame")

    # ------------------------------------------------------------------
    def run_inference(self, frames, flows):
        """Build tensors & run the multimodal fusion net."""
        flows = flows[: self.seq_len - 1]    # keep exactly 12 flows
        imgs  = torch.stack([self.img_tf(fr) for fr in frames]) \
                    .unsqueeze(0).to(self.device)  # [1,13,3,H,W]

        spatial_seq, global_seq = [], []
        for flow in flows:
            disp   = self.flow_extractor.extract_displacement_features(flow)
            deform = self.flow_extractor.extract_deformation_features(disp)
            grad   = self.flow_extractor.compute_spatial_gradients(disp)

            spatial_seq.append(torch.from_numpy(np.stack([
                disp['weighted_dx'], disp['weighted_dy'],
                disp['magnitude'],   disp['confidence'],
                grad['strain_xx'],   grad['strain_yy'],
                grad['strain_xy']
            ], axis=0)).float())

            global_seq.append(torch.from_numpy(np.array([
                deform['mean_displacement_x'],  deform['mean_displacement_y'],
                deform['std_displacement_x'],   deform['std_displacement_y'],
                deform['deformation_intensity'], deform['max_deformation'],
                deform['deformation_uniformity'], deform['angle_consistency'],
                deform['boundary_effect'],        deform['mean_confidence']
            ], dtype=np.float32)))

        spatial_seq = torch.stack(spatial_seq).unsqueeze(0).to(self.device)
        global_seq  = torch.stack(global_seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(imgs, spatial_seq, global_seq)
            return logits.argmax(1).item()

    # ------------------------------------------------------------------
    def run(self):
        try:
            while True:
                seq = self.acquire_sequence()
                if seq is None:
                    break
                frames, flows = seq
                pred = self.run_inference(frames, flows)
                print(f"➡  Predicted material ID: {pred}")
                if self.a.once:
                    break
        finally:
            try:
                self.rtde_c.moveL(HOME_WITH_TOOL, 0.25, 0.15)
            finally:
                self.cam.release()
                cv2.destroyAllWindows()


# -------------------------------------------------------------------------
if __name__ == "__main__":
    P = argparse.ArgumentParser("GelSight live tester with jitter")

    # model / camera / basic slide
    P.add_argument("--model", required=True, help="Path to .pth checkpoint")
    P.add_argument("--num_classes", type=int, default=10)
    P.add_argument("--cam",         type=int, default=0)
    P.add_argument("--axis", choices=["x", "y"], default="y")
    P.add_argument("--length", type=float, default=0.06,
                   help="Total slide length in metres")
    P.add_argument("--speed",  type=float, default=0.06,
                   help="TCP linear speed during slide (m/s)")
    P.add_argument("--sequence_length", type=int, default=13)
    P.add_argument("--img_size",        type=int, default=224)

    # press depth & jitter
    P.add_argument("--depth",  type=float, default=18.0,
                   help="Nominal press depth in mm before sliding")
    P.add_argument("--jitter", type=float, default=0.0,
                   help="± jitter around depth (mm)")
    P.add_argument("--max_depth", type=float, default=20.0,
                   help="Maximum allowable depth (mm)")

    # orientation offsets (deg) + jitter (deg)
    P.add_argument("--rx",  type=float, default=0.0)
    P.add_argument("--jrx", type=float, default=0.0)
    P.add_argument("--ry",  type=float, default=0.0)
    P.add_argument("--jry", type=float, default=0.0)
    P.add_argument("--rz",  type=float, default=0.0)
    P.add_argument("--jrz", type=float, default=0.0)

    # step control
    P.add_argument("--step", type=float, default=0.004,
                   help="Slide increment (m) if --auto_step off")
    P.add_argument("--auto_step", action="store_true",
                   help="Override --step so length/(seq_len-1) is used")

    # run flags
    P.add_argument("--once", action="store_true",
                   help="Run a single press-slide cycle then exit")

    args = P.parse_args()
    LiveTester(args).run()
