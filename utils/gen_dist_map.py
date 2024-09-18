import numpy as np
import cv2
import yaml
from pathlib import Path
import random
import argparse

def generate_dist_map(event_seq_path: Path, calib_seq_path: Path):
    calib_sequences = [x.stem for x in calib_seq_path.iterdir() if x.is_dir()]
    for seq in calib_sequences:
        calib_file_path = calib_seq_path / seq / "calibration/cam_to_cam.yaml"
        rectify_map_path = event_seq_path / seq / "events/left/rectify_map.h5"
        dist_map_path = event_seq_path / seq / "events/left/rect2dist_map.npy"
        if calib_file_path.exists() and rectify_map_path.exists():
            calibration = yaml.load(open(calib_file_path), Loader=yaml.FullLoader)
            intrinsic = calibration['intrinsics']['cam0']['camera_matrix']
            dist_params = calibration['intrinsics']['cam0']['distortion_coeffs']
            rect_intrinsic = calibration['intrinsics']['camRect0']['camera_matrix']
            R_rect0 = calibration['extrinsics']["R_rect0"]
            fx, fy, cx, cy = intrinsic
            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            fx, fy, cx, cy = rect_intrinsic
            rect_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            distortion_coeffs = np.array(dist_params)
            R = np.array(R_rect0)

            if not dist_map_path.is_file():
                rect2dist_map, _ = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeffs, R, rect_matrix, (640, 480),
                                                           cv2.CV_32FC2)
                np.save(str(dist_map_path), rect2dist_map)
                print("save", str(dist_map_path))
            else:
                rect2dist_map = np.load(str(dist_map_path))
                randx = random.randint(0, 639)
                randy = random.randint(0, 479)
                dist_point = rect2dist_map[randy, randx]
                rect_point = cv2.undistortPoints(dist_point, camera_matrix, distortion_coeffs, R=R, P=rect_matrix)
                rect_x = rect_point[0, 0, 0]
                rect_y = rect_point[0, 0, 1]
                max_error = 0.01
                if abs(rect_x-randx)<max_error and abs(rect_y-randy)<max_error:
                    error = max(abs(rect_y-randy), abs(rect_x-randx))
                    print(str(dist_map_path), np.array([randx, randy]), "check valid.", "error:", error)
                else:
                    print("!!!ERROR:", str(dist_map_path), np.array([randx, randy]), "check invalid")


def gen_DSEC_dist_map(path):
    data_path = Path(path)
    print("=======Begin to generate/check train sequences' dist map=======\n")
    event_seq_path = data_path / "Train/train_events/"
    calib_seq_path = data_path / "Train/train_calibration/"
    generate_dist_map(event_seq_path, calib_seq_path)

    print("\n=======Begin to generate/check test sequences' dist map=======\n")
    test_event_seq_path = data_path / "Test/test_events/"
    test_calib_seq_path = data_path / "Test/test_calibration/"
    generate_dist_map(test_event_seq_path, test_calib_seq_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate dist map for DSEC dataset")
    parser.add_argument("-d", "--dataset_path", default="/home/yyz/Dataset/DSEC/",
                        help="DSEC dataset path", type=str)
    args = parser.parse_args()
    gen_DSEC_dist_map(args.dataset_path)
