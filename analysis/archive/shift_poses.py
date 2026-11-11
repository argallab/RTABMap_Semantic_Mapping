"""Write a pose CSV shifted along each node's odometry twist by delta seconds
(T' = T * SE2(v*delta, w*delta)), same format as the exporter dump.
Used to feed latency-corrected start poses to tools/icp_probe.

Usage: python3 shift_poses.py <db> <in.csv> <out.csv> --delta -0.15
"""
import argparse
import csv

import numpy as np
from scipy.spatial.transform import Rotation as R

import common as C
from a1b_motion import shifted, twists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--delta", type=float, required=True)
    a = ap.parse_args()

    con = C.connect(a.db)
    poses = C.load_pose_csv(a.inp)
    stamps = {}
    with open(a.inp) as f:
        for row in csv.DictReader(f):
            stamps[int(row["id"])] = row["stamp"]
    out = shifted(poses, twists(con), a.delta)
    with open(a.out, "w") as f:
        f.write("id,stamp,x,y,z,qx,qy,qz,qw\n")
        for i in sorted(out):
            T = out[i]
            q = R.from_matrix(T[:3, :3]).as_quat()
            f.write(f"{i},{stamps[i]},{T[0,3]:.9g},{T[1,3]:.9g},{T[2,3]:.9g},"
                    f"{q[0]:.9g},{q[1]:.9g},{q[2]:.9g},{q[3]:.9g}\n")
    print(f"wrote {len(out)} poses shifted by {a.delta:+.3f} s -> {a.out}")


if __name__ == "__main__":
    main()
