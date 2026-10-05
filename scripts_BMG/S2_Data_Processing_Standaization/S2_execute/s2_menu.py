#!/usr/bin/env python3
"""
Menu chọn dữ liệu dùng chung cho các script S2 (s2_run_slam, s2.1, s2.2, s2.3).

Dùng khi chạy script mà không truyền --path:
  1. Chọn thư mục Date
  2. Chọn chạy cả thư mục Date hoặc 1 dataset lẻ
"""

import os


def _pick(title, options):
    """In danh sách đánh số, trả về mục được chọn (None nếu không hợp lệ)."""
    print(f"\n=== {title} ===")
    for i, name in enumerate(options, 1):
        print(f"{i}. {name}")
    try:
        idx = int(input(f"Chọn số tương ứng (1-{len(options)}): ")) - 1
    except ValueError:
        return None
    return options[idx] if 0 <= idx < len(options) else None


def _list_dirs(parent, prefix):
    return sorted(d for d in os.listdir(parent)
                  if d.startswith(prefix) and os.path.isdir(os.path.join(parent, d)))


def choose_path(root_dir):
    """
    Menu tương tác trong root_dir (S1_output hoặc S2_output_slam).
    Trả về đường dẫn tương đối: 'Date_xxx' (cả thư mục) hoặc 'Date_xxx/dataset_xxx' (1 dataset), None nếu hủy.
    """
    dates = _list_dirs(root_dir, "Date_") if os.path.isdir(root_dir) else []
    if not dates:
        print(f"[!] Không tìm thấy thư mục Date_... nào trong {root_dir}")
        return None

    date = _pick("CHỌN THƯ MỤC DATE", dates)
    if date is None:
        return None

    mode = _pick(f"CHẾ ĐỘ CHẠY ({date})", ["Chạy cả thư mục Date", "Chạy 1 dataset lẻ"])
    if mode is None:
        return None
    if mode.startswith("Chạy cả"):
        return date

    datasets = _list_dirs(os.path.join(root_dir, date), "dataset_")
    dataset = _pick(f"CHỌN DATASET ({date})", datasets) if datasets else None
    return os.path.join(date, dataset) if dataset else None
