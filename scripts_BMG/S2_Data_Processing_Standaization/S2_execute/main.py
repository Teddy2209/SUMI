import argparse
import subprocess
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Run S2 SLAM pipeline sequentially")
    parser.add_argument("--path", required=True, help="Relative path to dataset (e.g. Date_27082026/dataset_104821)")
    parser.add_argument("--no_show", action="store_true", help="Do not show plots at the end of s2.3")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Danh sách các script cần chạy tuần tự
    scripts = [
        "s2_run_slam.py",
        "s2.1_create_trajectory.py",
        "s2.3_vizualize_aruco.py"
    ]

    for script in scripts:
        script_path = os.path.join(base_dir, script)
        if not os.path.exists(script_path):
            print(f"[LỖI] Không tìm thấy file script: {script_path}")
            sys.exit(1)

        cmd = [sys.executable, script_path, "--path", args.path]
        
        # Thêm --no_show cho s2.3_vizualize_aruco.py để chạy tự động hoàn toàn (không bị chặn ở bước xem ảnh)
        if script == "s2.3_vizualize_aruco.py" and args.no_show:
            cmd.append("--no_show")

        print(f"\n{'='*60}")
        print(f"🚀 ĐANG CHẠY BƯỚC: {script}")
        print(f"{'='*60}")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[LỖI CỰC KỲ NGHIÊM TRỌNG] Chạy {script} thất bại với mã lỗi {e.returncode}.")
            print("Đang dừng toàn bộ pipeline.")
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n[HỦY] Đã dừng pipeline bởi người dùng (Ctrl+C).")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"✅ ĐÃ HOÀN THÀNH TOÀN BỘ PIPELINE S2 CHO DATASET: {args.path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
