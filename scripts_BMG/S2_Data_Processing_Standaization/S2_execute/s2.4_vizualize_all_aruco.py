import os
import sys
import glob
import subprocess
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
S2_OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "S2_output_slam"))

def main():
    parser = argparse.ArgumentParser(description="Visualize SLAM trajectories relative to ArUco marker for an entire Date folder.")
    parser.add_argument("--path", type=str, required=True, help="Path to Date folder (e.g., Date_23092026)")
    args = parser.parse_args()

    date_dir = os.path.join(S2_OUTPUT_DIR, args.path)
    if not os.path.isdir(date_dir):
        print(f"[-] Lỗi: Không tìm thấy thư mục {date_dir}")
        sys.exit(1)

    datasets = [d for d in sorted(glob.glob(os.path.join(date_dir, "dataset_*"))) if os.path.isdir(d)]
    if not datasets:
        print(f"[-] Không có dataset nào trong {date_dir}")
        sys.exit(1)

    print(f"[+] Tìm thấy {len(datasets)} datasets trong {args.path}")
    
    eval_csv_path = os.path.join(date_dir, "dataset_evaluation.csv")
    evaluations = {}
    if os.path.exists(eval_csv_path):
        df_eval = pd.read_csv(eval_csv_path, header=None, names=["dataset", "status"])
        evaluations = dict(zip(df_eval["dataset"], df_eval["status"]))
        
    def save_evaluations():
        with open(eval_csv_path, 'w') as f:
            for ds_path in datasets:
                ds = os.path.basename(ds_path)
                st = evaluations.get(ds, "chưa đánh giá")
                f.write(f"{ds},{st}\n")

    # Lưu ngay file CSV lúc bắt đầu để liệt kê đầy đủ toàn bộ dataset
    save_evaluations()

    print("[!] Hướng dẫn sử dụng:")
    print("    - Phím 'Mũi tên Phải' (Right) : Xem dataset tiếp theo")
    print("    - Phím 'Mũi tên Trái' (Left)  : Xem dataset trước đó")
    print("    - Phím 'G'                    : Đánh dấu dataset TỐT (good)")
    print("    - Phím 'B'                    : Đánh dấu dataset XẤU (bad)")
    print("    - Phím 'Q' hoặc 'Esc'         : Thoát chương trình\n")

    current_idx = 0

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    def update_plot(idx):
        if idx < 0 or idx >= len(datasets):
            return
        
        ax.clear()
        ds_path = datasets[idx]
        ds_name = os.path.basename(ds_path)
        
        csv_path = os.path.join(ds_path, "Mapped_SLAM_Trajectory_Marker.csv")
        
        # Nếu chưa có CSV thì chạy s2.4_vizualize_trajectory_aruco.py với --no_show
        if not os.path.exists(csv_path):
            print(f"[*] Dataset {ds_name} chưa có file Mapped_SLAM_Trajectory_Marker.csv. Đang tạo...")
            s2_4_script = os.path.join(BASE_DIR, "s2.4_vizualize_trajectory_aruco.py")
            cmd = ["python", s2_4_script, "--path", f"{args.path}/{ds_name}", "--no_show"]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"[-] Lỗi khi chạy s2.4 cho dataset {ds_name}")
                ax.text2D(0.5, 0.5, f"Lỗi xử lý {ds_name}", transform=ax.transAxes, ha="center")
                ax.set_title(f"[{idx+1}/{len(datasets)}] {ds_name} - ERROR")
                fig.canvas.draw()
                return
                
        if not os.path.exists(csv_path):
            ax.text2D(0.5, 0.5, f"Không tìm thấy dữ liệu cho {ds_name}", transform=ax.transAxes, ha="center")
            ax.set_title(f"[{idx+1}/{len(datasets)}] {ds_name} - MISSING DATA")
            fig.canvas.draw()
            return
            
        status = evaluations.get(ds_name, "Chưa đánh giá")
        print(f"[+] Đang hiển thị: {ds_name} ({idx+1}/{len(datasets)}) - Trạng thái: {status}")
        df = pd.read_csv(csv_path)
        
        cam_points = df[['marker_cam_x', 'marker_cam_y', 'marker_cam_z']].to_numpy()
        tool_points = df[['marker_tool_x', 'marker_tool_y', 'marker_tool_z']].to_numpy()
        
        # Vẽ gốc tọa độ Marker
        ax.scatter(0, 0, 0, color='black', s=200, label='Marker Origin (0,0,0)', marker='*')
        # Vẽ các trục của Marker (5cm)
        ax.plot([0, 0.05], [0, 0], [0, 0], color='red', linewidth=3, label='Marker X-axis')
        ax.plot([0, 0], [0, 0.05], [0, 0], color='green', linewidth=3, label='Marker Y-axis')
        ax.plot([0, 0], [0, 0], [0, 0.05], color='blue', linewidth=3, label='Marker Z-axis')

        # Vẽ quỹ đạo Camera
        ax.plot(cam_points[:, 0], cam_points[:, 1], cam_points[:, 2],
                label='Camera Trajectory', color='green', linewidth=1)

        # Vẽ quỹ đạo Tool
        ax.plot(tool_points[:, 0], tool_points[:, 1], tool_points[:, 2],
                label='SLAM Predicted (Tool)', color='red', linewidth=2)

        # Điểm bắt đầu
        if len(tool_points) > 0 and len(cam_points) > 0:
            ax.scatter(*tool_points[0], color='cyan', s=100, label='Start (Tool)', marker='o', edgecolor='black', zorder=5)
            ax.scatter(*cam_points[0], color='yellow', s=100, label='Start (Camera)', marker='s', edgecolor='black', zorder=5)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f"[{idx+1}/{len(datasets)}] {ds_name}\nTrạng thái: {status.upper()}")
        ax.legend()

        # Equal aspect ratio trick cho 3D plot
        if len(cam_points) > 0:
            max_range = np.array([cam_points[:, 0].max()-cam_points[:, 0].min(),
                                  cam_points[:, 1].max()-cam_points[:, 1].min(),
                                  cam_points[:, 2].max()-cam_points[:, 2].min()]).max() / 2.0
            mid_x = (cam_points[:, 0].max()+cam_points[:, 0].min()) * 0.5
            mid_y = (cam_points[:, 1].max()+cam_points[:, 1].min()) * 0.5
            mid_z = (cam_points[:, 2].max()+cam_points[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
        fig.canvas.draw()

    def on_key(event):
        nonlocal current_idx
        if event.key == 'right':
            if current_idx < len(datasets) - 1:
                current_idx += 1
                update_plot(current_idx)
            else:
                print("[-] Đã là dataset cuối cùng.")
        elif event.key == 'left':
            if current_idx > 0:
                current_idx -= 1
                update_plot(current_idx)
            else:
                print("[-] Đã là dataset đầu tiên.")
        elif event.key == 'g':
            ds_name = os.path.basename(datasets[current_idx])
            evaluations[ds_name] = "tốt"
            save_evaluations()
            print(f"[v] Đã đánh dấu {ds_name} là TỐT")
            update_plot(current_idx)
        elif event.key == 'b':
            ds_name = os.path.basename(datasets[current_idx])
            evaluations[ds_name] = "xấu"
            save_evaluations()
            print(f"[x] Đã đánh dấu {ds_name} là XẤU")
            update_plot(current_idx)
        elif event.key in ['escape', 'q']:
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Hiển thị dataset đầu tiên
    update_plot(current_idx)
    plt.show()

if __name__ == "__main__":
    main()
