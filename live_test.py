#!/usr/bin/env python3
"""Live testing script for the trained multimodal GelSight model.

This script performs a single slide motion using a UR5 robot, captures
synchronized GelSight images and flow, and runs the multimodal neural
network to predict the contacted material. Most hardware constants are
borrowed from the data collection controller.
"""

import os
import cv2
import math
import time
import argparse
import signal
import torch
import torchvision.transforms as T
import numpy as np

from pathlib import Path

from flow_processor import FlowFeatureExtractor
from multimodal_network import MultimodalFusionNet

# Optional camera/flow utilities used during data collection.
# They are assumed to be available in the runtime environment.
import setting
import find_marker
import A_utility

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# ---------------------------------------------------------------------------
# Robot and camera constants (identical to the data collection script)
ROBOT_IP, RTDE_PORT = "10.10.10.1", 50002
HOME_WITH_TOOL = [-0.298569, -0.694446, 0.239335,
                  0.633457, -1.477861, 0.626266]
NEW_TCP = (0, 0, 0.26, 0, 0, 0)

ROT = math.radians(45)  # lab ↔ robot frame rotation
SLIDE_AXIS, SLIDE_LENGTH, STEP, SPEED = 'y', 0.06, 0.004, 0.06
FLUSH_FRAMES = 20

_rot  = lambda x,y,t:(math.cos(t)*x - math.sin(t)*y,
                      math.sin(t)*x + math.cos(t)*y)
old2new = lambda p:[*_rot(p[0],p[1],-ROT),*p[2:]]
new2old = lambda p:[*_rot(p[0],p[1], ROT),*p[2:]]

# ---------------------------------------------------------------------------
class LiveTester:
    def __init__(self, args):
        self.args = args

        # Robot init -------------------------------------------------------
        self.rtde_c = RTDEControlInterface(ROBOT_IP, RTDE_PORT)
        self.rtde_r = RTDEReceiveInterface(ROBOT_IP)
        self.rtde_c.setTcp(NEW_TCP)

        # Camera init ------------------------------------------------------
        self.cam = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
        if not self.cam.isOpened():
            raise RuntimeError("Camera not found")
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH , 800)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE , 1)

        setting.init()
        self.matcher = find_marker.Matching(
            N_=setting.N_, M_=setting.M_, fps_=setting.fps_,
            x0_=setting.x0_, y0_=setting.y0_,
            dx_=setting.dx_, dy_=setting.dy_)

        # neural network ---------------------------------------------------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(args.model, map_location=self.device)
        self.model = MultimodalFusionNet(num_classes=args.num_classes)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device).eval()

        self.flow_extractor = FlowFeatureExtractor()
        self.img_tf = T.Compose([
            T.ToPILImage(),
            T.Resize((args.img_size, args.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.seq_len = args.sequence_length
        self.home_new = old2new(HOME_WITH_TOOL)
        signal.signal(signal.SIGINT, self._sig_exit)

    def _sig_exit(self, *_):
        print("\n[ABORT] Ctrl-C pressed — stopping robot.")
        try:
            self.rtde_c.stopScript()
        except Exception:
            pass
        self.cam.release(); cv2.destroyAllWindows(); exit(1)

    def _flush_cam(self):
        for _ in range(FLUSH_FRAMES):
            self.cam.grab(); self.cam.retrieve()

    def _hover_point(self):
        return self.home_new.copy()

    # ------------------------------------------------------------------
    def acquire_sequence(self):
        """Perform a single slide motion and capture a sequence."""
        a = self.args
        rtde = self.rtde_c

        # reset to HOME
        rtde.moveL(HOME_WITH_TOOL, 0.25, 0.15)
        time.sleep(0.05)

        hover = self._hover_point()
        pose_hover = hover[:3] + list(HOME_WITH_TOOL[3:])
        rtde.moveL(new2old(pose_hover), 0.20, 0.12)
        time.sleep(0.05)

        # press down
        tgt = pose_hover.copy()
        tgt[2] -= a.depth/1000.0
        rtde.moveL(new2old(tgt), 0.04, 0.03)
        time.sleep(0.12)

        # slide and capture
        frames = []
        flows = []
        travelled = 0.0
        ax = 0 if a.axis=='x' else 1
        self._flush_cam()

        while travelled < a.length and len(frames) < self.seq_len:
            ret, fr = self.cam.read()
            if not ret:
                raise RuntimeError("Failed to capture frame")
            self.matcher.init(A_utility.marker_center(fr))
            self.matcher.run()
            flow = self.matcher.get_flow()

            frames.append(fr)
            flows.append(np.array(flow))

            travelled += a.step
            tgt_slide = tgt.copy()
            tgt_slide[ax] += travelled
            rtde.moveL(new2old(tgt_slide), a.speed, a.speed)
            cv2.imshow("GelSight", A_utility.draw_flow(fr.copy(), flow))
            if cv2.waitKey(1) == 27:
                return None

        rtde.moveL(new2old(pose_hover), a.speed, a.speed)
        time.sleep(0.08)
        self._flush_cam()
        return frames, flows

    # ------------------------------------------------------------------
    def run_inference(self, frames, flows):
        """Run the neural network on captured data."""
        if len(frames) < self.seq_len:
            raise ValueError("Not enough frames captured")
        T_required = self.seq_len
        F_required = self.seq_len - 1

        imgs = [self.img_tf(fr) for fr in frames[:T_required]]
        imgs = torch.stack(imgs).unsqueeze(0).to(self.device)

        spatial_seq = []
        global_seq = []
        for flow in flows[:F_required]:
            disp = self.flow_extractor.extract_displacement_features(flow)
            deform = self.flow_extractor.extract_deformation_features(disp)
            grad = self.flow_extractor.compute_spatial_gradients(disp)
            spatial = np.stack([
                disp['weighted_dx'],
                disp['weighted_dy'],
                disp['magnitude'],
                disp['confidence'],
                grad['strain_xx'],
                grad['strain_yy'],
                grad['strain_xy'],
            ], axis=0)
            spatial_seq.append(torch.from_numpy(spatial).float())
            global_seq.append(torch.from_numpy(np.array([
                deform['mean_displacement_x'],
                deform['mean_displacement_y'],
                deform['std_displacement_x'],
                deform['std_displacement_y'],
                deform['deformation_intensity'],
                deform['max_deformation'],
                deform['deformation_uniformity'],
                deform['angle_consistency'],
                deform['boundary_effect'],
                deform['mean_confidence'],
            ]).astype(np.float32)))

        spatial_seq = torch.stack(spatial_seq).unsqueeze(0).to(self.device)
        global_seq = torch.stack(global_seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(imgs, spatial_seq, global_seq)
            pred = logits.argmax(dim=1).item()
        return pred

    # ------------------------------------------------------------------
    def run(self):
        while True:
            seq = self.acquire_sequence()
            if seq is None:
                break
            frames, flows = seq
            pred = self.run_inference(frames, flows)
            print(f"Predicted material ID: {pred}")
            if self.args.once:
                break

        self.rtde_c.moveL(HOME_WITH_TOOL, 0.25, 0.15)
        self.cam.release(); cv2.destroyAllWindows(); self.rtde_c.stopScript()

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Live tester for trained multimodal GelSight model")
    parser.add_argument("--model", required=True, help="path to trained model checkpoint")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--depth", type=float, default=18.0)
    parser.add_argument("--axis", choices=["x","y"], default=SLIDE_AXIS)
    parser.add_argument("--length", type=float, default=SLIDE_LENGTH)
    parser.add_argument("--step", type=float, default=STEP)
    parser.add_argument("--speed", type=float, default=SPEED)
    parser.add_argument("--sequence_length", type=int, default=13)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--once", action="store_true", help="run only one cycle")
    args = parser.parse_args()

    tester = LiveTester(args)
    tester.run()
