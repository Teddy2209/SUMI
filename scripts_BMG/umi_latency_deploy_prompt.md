# PROMPT GIAO VIỆC: Triển khai căn chỉnh độ trễ (latency matching) kiểu UMI cho pipeline của dự án

> Cách dùng: dán toàn bộ file này làm prompt đầu tiên cho một AI coding agent (ví dụ Claude Code) đang mở repo của dự án.
> Mọi khẳng định kỹ thuật bên dưới được gắn nhãn: **[ĐÃ KIỂM CHỨNG]** (đã đọc code/bài báo gốc), **[SUY LUẬN]** (suy ra từ cái đã đọc, chưa chạy), **[CHƯA KIỂM TRA]**. Nếu nhãn là [SUY LUẬN] hoặc [CHƯA KIỂM TRA], bạn phải tự kiểm tra trước khi dựa vào nó.

---

## 0. Vai trò và mục tiêu

Bạn là kỹ sư robotics/ML senior. Nhiệm vụ: đưa cơ chế **căn chỉnh độ trễ quan sát (PD1.1)** và **căn chỉnh độ trễ thực thi (PD1.2)** của UMI vào pipeline deploy của dự án, để policy huấn luyện bằng LeRobot chạy trên robot thật mà không bị lệch pha giữa ảnh, pose, độ mở gripper và hành động.

Bạn **không** được "xóa" độ trễ vật lý (không thể làm ảnh tươi hơn). Việc cần làm chỉ gồm hai thứ:
1. **Gán nhãn lại thời gian** cho mọi luồng cảm biến, rồi nội suy các luồng còn lại về đúng thời điểm của khung ảnh (PD1.1).
2. **Gửi lệnh sớm** đúng bằng độ trễ thực thi của phần cứng, và bỏ các action đã quá hạn (PD1.2).

## 1. Bối cảnh dự án (do người dùng cung cấp)

- Pipeline UMI cho **một gripper** điện 2 ngón (không bimanual). SLAM chạy chế độ RGB-D. Huấn luyện bằng **LeRobot**. Chạy trên robot thật.
- Hiện trạng deploy: **bỏ qua khâu khử trễ**; vòng lặp tuần tự, robot chạy hết `n_action_steps` (người dùng nói khoảng 8 điểm) rồi mới suy luận.
- **Chưa đo** bất kỳ độ trễ nào. Mọi giá trị trễ dưới đây chỉ là giá trị mặc định của UMI gốc, **không áp dụng được** cho phần cứng này cho tới khi đo.
- Chưa biết (phải hỏi người dùng ở mục 11): loại robot arm và driver, tần số/độ trễ trạng thái gripper, camera và đường truyền ảnh, phiên bản LeRobot.
- Mục tiêu cấp dự án: pick ≥ 95%, gấp vải ≥ 80%, cắm vít ≥ 60% (cắm vít cần độ chính xác mm); quan tâm thêm độ bền với thay đổi ánh sáng.

## 2. Nguồn chính thống và phiên bản

| Nguồn | Phiên bản đã đọc |
|---|---|
| Bài báo UMI, arXiv 2402.10329v3 | Fig. 5, mục III-B (PD1, PD2), phụ lục A (đo độ trễ) |
| `real-stanford/universal_manipulation_interface`, nhánh main | commit `d095ba9` (26/05/2026) |
| `huggingface/lerobot`, nhánh main | commit `ca69a20` (07/10/2026) |

Quy tắc: nếu bạn clone commit khác và thấy khác biệt, **báo lại** thay vì làm theo trí nhớ. Không dùng nguồn web ngoài trừ khi nói rõ đó là nguồn ngoài.

## 3. Ký hiệu (dùng đồng nhất trong code, comment, log, báo cáo)

| Ký hiệu | Nghĩa | Tên trong code UMI |
|---|---|---|
| Δt = 1/f | chu kỳ điều khiển (UMI mặc định f = 10 Hz) | `dt` |
| ℓ_c, ℓ_r, ℓ_g | trễ quan sát camera / robot / gripper | `camera_obs_latency`, `robot_obs_latency`, `gripper_obs_latency` |
| ℓ^a_r, ℓ^a_g | trễ thực thi hành động robot / gripper | `robot_action_latency`, `gripper_action_latency` |
| t^rx | timestamp lúc nhận | `t_recv`, `*_receive_timestamp` |
| t̃ = t^rx − ℓ | timestamp đã hiệu chỉnh | `t_cal`, trường `timestamp`, `robot_timestamp`, `gripper_timestamp` |
| t_obs | t̃ của khung camera mới nhất (neo) | `last_timestamp` |
| t_in, t_out, T_inf | bắt đầu / kết thúc suy luận, T_inf = t_out − t_in | |
| t_act = t_out + ℓ^a | thời điểm sớm nhất một lệnh còn kịp thành chuyển động thật | |
| H, K, n_o, d | action horizon (16), số bước thực thi mỗi lần suy luận (`steps_per_inference`, mặc định 6), obs horizon (UMI deploy: 2), bước down-sample | |
| a_k, τ_k = t_obs + kΔt | action thứ k và thời điểm vật lý nó nhắm tới | `action_timestamps` |
| k_min = ⌈(t_act − t_obs)/Δt⌉ | chỉ số action đầu tiên còn dùng được | |
| t_j = t_obs − (n_o − 1 − j)·d·Δt | thời điểm truy vấn quan sát, j = 0…n_o−1 | `camera_obs_timestamps`... |
| m ∈ {c, r, g}; T_m | chỉ số luồng; chu kỳ lấy mẫu của luồng | |
| t̃_m^new | t̃ của mẫu mới nhất của luồng m | |
| h, s | khoảng cách hai mẫu liền kề; trọng số nội suy s = (t − t_i)/(t_{i+1} − t_i) | |
| t_safe = min_m t̃_m^new | mốc muộn nhất mà mọi luồng đều đã có dữ liệu | |
| u_i, e_i = t̃_i − u_i, σ_e | thời điểm phơi sáng thật của khung i; sai số nhãn (jitter); độ lệch chuẩn | STD của script đo ℓ_c |
| δp ≈ v·e_i | lệch ảnh–pose do sai số nhãn (v là tốc độ tay) | |

## 4. Kiến trúc UMI gốc khi deploy [ĐÃ KIỂM CHỨNG]

Bài báo không nói về thread hay process; phần này đọc từ code.

- **Camera** (`umi/real_world/uvc_camera.py`, `UvcCamera`): một `mp.Process` riêng chạy ở `capture_fps` (60). Mỗi vòng: `grab()` → `retrieve()` → `t_recv = time.time()` → `t_cal = t_recv − receive_latency` → ghi **mọi** khung vào `SharedMemoryRingBuffer` (`put_downsample=False`). Trong `UmiEnv`, buffer giữ `get_max_k = max_obs_buffer_size = 60` khung (≈ 1 s). `cap_buffer_size` = 1 (60 fps) hoặc 3 (Cam Link 4K 30 fps) [mục đích suy đoán: giữ hàng đợi driver ngắn].
- **Robot controller** (`rtde_interpolation_controller.py`, `franka_interpolation_controller.py`): mỗi cái một `mp.Process`; UR5 `frequency=125`, Franka `frequency=1000`. Controller giữ một `PoseTrajectoryInterpolator`; mỗi nhịp tính `pose_interp(t_now)` rồi gọi `servoL`. Lệnh vào qua hàng đợi và **chỉ lấy tối đa 1 lệnh mỗi chu kỳ** (`get_k(1)`). Trạng thái robot được ghi với `robot_timestamp = t_recv − receive_latency`.
- **Gripper controller** (`wsg_controller.py`): process riêng. [CHƯA KIỂM TRA] có cùng giới hạn 1 lệnh/chu kỳ hay không.
- **Vòng suy luận** = vòng lặp chính của `eval_real.py` (không phải thread riêng): `get_obs` → `predict_action` → `exec_actions` → `precise_wait`. Chu kỳ là **K·Δt** (mặc định 6 × 0,1 = 0,6 s), không phải mỗi Δt.
- Đồng hồ chung của camera, robot state, gripper state và `exec_actions` là `time.time()`. Controller tự đổi sang `time.monotonic()` nội bộ (`target_time = time.monotonic() − time.time() + target_time`).
- Robot **không đứng chờ** policy: trong lúc suy luận nó tiếp tục đi theo quỹ đạo cũ.

## 5. PD1.1: căn chỉnh quan sát [ĐÃ KIỂM CHỨNG: bài báo + `UmiEnv.get_obs`]

Thuật toán mỗi lần gọi `get_obs()`:

1. Đưa mọi luồng về thời gian vật lý: `t̃ = t^rx − ℓ_stream`.
2. Lấy k khung camera mới nhất, k = ⌈n_o·d·(60/f)⌉. **Số 60 viết cứng** trong `UmiEnv.get_obs` (ví dụ n_o=2, d=1, f=10 → k=12).
3. Neo: `t_obs = t̃ của khung camera mới nhất`. Bài báo: neo vào luồng trễ nhiều nhất (thường là camera).
4. Lưới truy vấn: `t_j = t_obs − (n_o − 1 − j)·d·Δt`.
5. Với từng t_j:
   - **ảnh**: chọn khung gần nhất `argmin_i |t̃_i − t_j|` (không nội suy ảnh; sai số thời gian ≤ T_c/2 nếu không rớt khung);
   - **vị trí, độ mở gripper**: nội suy tuyến tính giữa hai mẫu bao quanh, `x = (1−s)·x_i + s·x_{i+1}` (`get_interp1d`, `PoseInterpolator`);
   - **quay**: slerp, `R(s) = R_i·Exp(s·Log(R_iᵀR_{i+1}))` (`scipy.Slerp`). Không được lerp rotvec (gập tại |r| = π), không lerp ma trận (mất tính trực giao).
6. Lỗi nội suy tuyến tính: `|e| ≤ h²/8 · max|p″|`. Phải dùng **hai mẫu liền kề**, không nối tới mẫu "now".

**Điều kiện không ngoại suy** (`m` là luồng nào đó khác camera): đảm bảo có dữ liệu bên phải t_obs khi `ℓ_c ≥ ℓ_m + T_m`. Nếu `0 ≤ ℓ_c − ℓ_m < T_m` thì xác suất đủ dữ liệu ở một chu kỳ là `(ℓ_c − ℓ_m)/T_m`, thời gian chờ tối đa `w_max = ℓ_m + T_m − ℓ_c`. Nếu `ℓ_c < ℓ_m` thì luôn thiếu.

**UMI không có `assert` cho điều kiện này.** `PoseInterpolator` dùng `np.clip(t, min_t, max_t)` và `get_interp1d` dùng `bounds_error=False, fill_value=(x[0], x[-1])`: khi truy vấn vượt mẫu cuối, code **âm thầm giữ mẫu cuối** (đã chạy thử: 57,0 mm và pose 0,2 đúng như vậy), không báo lỗi. Phần cứng gốc luôn thỏa điều kiện (ℓ_c = 0,125 ≥ ℓ_g + T_g ≈ 0,043) nên tác giả không gặp.

Bốn cách xử lý khi điều kiện vi phạm (ví dụ kẹp đóng 100 mm/s, đáp án đúng 52,0 mm):

| Cách | Kết quả | Cái giá |
|---|---|---|
| Giữ mẫu cuối (UMI) | 57,0 (sai +5,0) | sai ≈ vận tốc × khoảng thiếu, **không báo lỗi** |
| Ngoại suy tuyến tính | 55,5 (sai +3,5) | tệ khi vừa bắt đầu/dừng chuyển động |
| Chờ mẫu bên phải | 52,0 | tăng độ trễ chung (cộng vào k_min) |
| Lùi neo về `t_safe` | đúng | quan sát cũ hơn |

Yêu cầu cho dự án: **mặc định fail-loud** (log + đếm vi phạm, không giữ im lặng); chọn cách xử lý theo số đo thật, không chọn trước.

## 6. PD1.2: căn chỉnh hành động [ĐÃ KIỂM CHỨNG: bài báo + `eval_real.py` + `UmiEnv/BimanualUmiEnv.exec_actions` + controller]

Policy dự đoán chunk bắt đầu tại bước quan sát cuối (t_obs). Trong `eval_real.py`:

```
action_timestamps = arange(len(action)) * dt + obs_timestamps[-1]      # τ_k = t_obs + kΔt, k = 0…H−1
```

Chuỗi xử lý mỗi chu kỳ:

1. `curr_time = time.time()` sau khi suy luận xong (≈ t_out).
2. Lọc: `is_new = action_timestamps > curr_time + action_exec_latency` (`action_exec_latency = 0.01` viết cứng; **không phải** ℓ^a). Nếu không còn action nào: nhánh `Over budget`, chỉ thực thi action cuối ở bước lưới kế tiếp.
3. `env.exec_actions(actions, timestamps, compensate_latency=True)`. Trong env (cả `UmiEnv` lẫn `BimanualUmiEnv`), **ở process chính**: lọc `timestamps > time.time()`, rồi `schedule_waypoint(pose, target_time = τ_k − ℓ^a_r)` cho robot và `schedule_waypoint(pos, target_time = τ_k − ℓ^a_g)` cho gripper. Controller không biết ℓ^a.
4. Controller (`PoseTrajectoryInterpolator.schedule_waypoint`): bỏ waypoint nếu `time <= curr_time`; nếu `time <= last_waypoint_time` thì cắt quỹ đạo cũ tại `curr_time` rồi nối từ pose hiện tại tới waypoint mới (ghi đè). Vị trí liên tục, vận tốc **có thể gấp khúc** giữa hai chunk; code không trộn hai chunk với nhau.
5. `precise_wait(t_cycle_end − frame_latency)`, `frame_latency = 1/60`, `t_cycle_end = t_start + (iter_idx + K)·Δt`.

Công thức:

```
t_act − t_obs = ℓ_c + T_inf + ℓ^a          (xấp xỉ; code dùng đồng hồ thật)
k_min = ⌈(ℓ_c + T_inf + ℓ^a)/Δt⌉
thời điểm gửi action k: τ_k − ℓ^a           (pos(t) = cmd(t − ℓ^a) ⇒ pos(τ_k) = a_k)
chunk n sở hữu bước k_min … k_min+K−1 (steady state); phần còn lại bị chunk n+1 ghi đè
điều kiện chạy ổn: (a) H ≥ k_min + K ; (b) T_inf < K·Δt
độ dư quỹ đạo ≈ (H−1)Δt − ℓ^a − (KΔt + ℓ_c + T_inf)
```

Bài báo nói loại action cũ "cho từng phần cứng"; trong code việc này xảy ra ở từng controller (waypoint có thời gian ≤ `curr_time` bị bỏ), nên robot và gripper có ℓ^a khác nhau thì ngưỡng k_min thực tế khác nhau.

Pose gốc để đổi action tương đối sang tuyệt đối: `get_real_umi_action` dùng `env_obs[...][-1]`, tức pose đã căn chỉnh tại t_obs (không phải pose mới nhất lúc suy luận xong). Giữ nguyên bất biến này.

## 7. Đo các độ trễ [ĐÃ KIỂM CHỨNG: bài báo phụ lục A + `scripts/`]

| Đại lượng | Cách đo (bài báo A.1–A.4) | Script |
|---|---|---|
| ℓ_c | hiển thị QR chứa timestamp lên màn hình, quay lại bằng camera, ℓ_c = mean(t_recv − t_display) − ℓ_display; **t_recv chính là `camera_receive_timestamp` của process camera** | `scripts/calibrate_uvc_camera_latency.py` (in `AVG` và `STD`) |
| ℓ_r, ℓ_g | timestamp phần cứng nếu có (Franka); không có thì xấp xỉ ½ RTT của ping (UR5, WSG-50) | |
| ℓ^a_g | ra lệnh tín hiệu sin, ghi đáp ứng thật, tìm độ lệch bằng cross-correlation | `scripts/calibrate_gripper_latency.py` |
| ℓ^a_r | tương tự: pose mong muốn (teleop) so với pose đo | `scripts/calibrate_robot_latency.py` |

Điểm cần nhớ:
- ℓ_c được đo **xuyên qua đúng đường code của process camera**, nên phần trễ hằng số do process nằm sẵn trong ℓ_c. Phần còn lại là jitter e_i, và δp ≈ v·e_i (ví dụ jitter đều trên một chu kỳ 16,7 ms: σ_e ≈ 4,8 ms; tay 0,8 m/s → ≈ 3,9 mm; không bù thì ℓ_c·v = 0,125 × 0,8 = 100 mm).
- `get_latency` (`umi/common/latency_util.py`) dùng cross-correlation; **chỉ đo đúng khi ℓ < T_signal/2**. Với tín hiệu của `calibrate_gripper_latency.py` (chu kỳ ≈ 0,94 s) ℓ thật 0,6 s cho kết quả −0,341 s (alias). Muốn đo trễ lớn hơn phải kéo dài chu kỳ tín hiệu lệnh.
- `get_latency(..., force_positive=True)` có lỗi: trả −9,845 s khi ℓ thật là 0,12 s (index vào `t_lags` thay vì `t_lags[t_lags >= 0]`). Không script nào trong repo dùng tùy chọn này. Đừng dùng nó.
- Chạy mỗi phép đo ≥ 5 lần, ghi trung bình và độ lệch chuẩn.

## 8. Các bẫy và chỗ lệch giữa bài báo, config và code [ĐÃ KIỂM CHỨNG]

1. `eval_real.py` viết cứng `camera_obs_latency=0.17` khi dựng env; `UmiEnv` và `umi.yaml` dùng 0,125. Giá trị này **không** đọc từ config robot. Đổi camera mà quên sửa là sai ngầm.
2. Giá trị mặc định ℓ^a_r = ℓ^a_g = 0,1 s (`example/eval_robots_config.yaml`), trong khi bài báo (Fig. 5) ghi 100 ms cho robot và 120 ms cho gripper trên phần cứng của họ.
3. `command_latency` (mặc định 0,01) trong `eval_real.py` là độ trễ lệnh của SpaceMouse khi teleop, **không phải** ℓ^a. `action_exec_latency = 0.01` cũng không phải ℓ^a.
4. `umi.yaml` có `latency_steps = (camera_obs_latency − robot_obs_latency) × dataset_frequeny`, nhưng `dataset_frequeny: 0` nên mặc định **tắt**. Với demo UMI, pose SLAM đã khớp từng khung nên không cần; chỉ có ý nghĩa với dữ liệu có proprioception do robot ghi.
5. `camera_capture_timestamp` (timestamp driver V4L2) được lưu nhưng trong `umi/` không có chỗ nào dùng để căn chỉnh; nhãn dùng là `t_cal = t_recv − ℓ_c`.
6. Hằng số `60` trong tính k của `get_obs`; camera khác 60 fps phải sửa.
7. Controller chỉ nhận ≤ 1 lệnh mỗi chu kỳ (RTDE). Controller gripper của dự án nếu chạy tần số thấp có thể ingest chunk rất chậm (ví dụ 12 waypoint ở 30 Hz ≈ 0,4 s) [SUY LUẬN, cần kiểm tra driver của dự án].

## 9. LeRobot: đã có gì, thiếu gì [ĐÃ KIỂM CHỨNG tại `ca69a20`, trừ chỗ ghi khác]

- **Camera** (`cameras/opencv/camera_opencv.py`): một `Thread` (`_read_loop`), chỉ giữ **một khung mới nhất** (`latest_frame`) cùng `latest_timestamp = time.perf_counter()` đo **sau** khi đọc và hậu xử lý (tên biến `capture_time` nhưng thực chất là thời điểm nhận). **Không có ring buffer, không trừ ℓ_c.** `async_read()` chờ khung mới (timeout 200 ms) rồi xóa event; `read_latest()` không chờ và ném lỗi nếu khung già hơn `max_age_ms` (500). Docstring ví dụ viết `read_latest()` trả `(image, timestamp)` nhưng code trả về chỉ khung. Chỉ đọc `OpenCVCamera`; chưa đọc realsense/zmq.
- **Đồng hồ**: `perf_counter` có gốc tùy ý, khác `time.time()` của UMI; phải thống nhất miền đồng hồ trước khi trộn các luồng.
- **`DiffusionPolicy`** (`policies/diffusion/modeling_diffusion.py`): hai `deque` (quan sát `n_obs_steps`, hành động `n_action_steps`), `select_action` lấy dần từng action; **không có timestamp**, tức ngầm giả định trễ bằng 0 và thực thi tức thì.
- **RTC** (`policies/rtc`): có ở smolvla, pi0, pi05, pi0_fast, groot, molmoact2, evo1; **không có trong `policies/diffusion` và `policies/act`**. Theo lượt đọc ở commit `e0d5021`, RTC chỉ bù trễ suy luận (T_inf), không bù ℓ_c hay ℓ^a [chưa đọc lại ở `ca69a20`].
- **`async_inference/`**: có hàng đợi hành động chạy song song suy luận, `chunk_size_threshold` (0,5), `aggregate_fn`; hành động gắn nhãn bằng chỉ số bước (`timestep`), `TimedData` có trường `timestamp`. Chưa đọc `policy_server.py` nên chưa biết có dùng timestamp để bù trễ hay không.
- **Chunk bắt đầu từ bước quan sát cuối** (docstring `select_action`): về nguyên tắc tương tự UMI, nên các action đầu của chunk cũng đã "cũ" khi suy luận xong. [SUY LUẬN] UMI bỏ các action k < k_min; LeRobot mặc định thực thi ngay a_0.

## 10. Bất biến phải giữ giữa huấn luyện và deploy

1. Dữ liệu huấn luyện: ảnh, pose và độ mở gripper của mỗi mẫu phải cùng một thời điểm vật lý (UMI đạt được điều này vì mọi thứ nằm trong cùng file MP4 của GoPro cộng pose SLAM). Nếu dataset LeRobot của dự án được dựng lại, kiểm tra không có độ lệch thời gian giữa các cột.
2. `n_o`, `d`, `Δt` (tức `n_obs_steps`, bước giữa các quan sát, tần số điều khiển) lúc deploy phải bằng lúc train [SUY LUẬN].
3. Biểu diễn hành động tương đối: pose gốc để giải mã là pose tại t_obs (đã căn chỉnh).
4. Mô hình học "a_k nghĩa là pose tại thời điểm vật lý τ_k = t_obs + kΔt". Mọi thay đổi lịch gửi lệnh phải bảo toàn nghĩa này.

## 11. Quy trình làm việc bắt buộc

**Trước mọi thứ:** đọc repo của dự án ở chế độ chỉ đọc, rồi trả về (a) bản tóm tắt kiến trúc hiện tại (camera, robot, gripper, vòng lặp, LeRobot version), (b) danh sách những thứ còn chưa biết, (c) câu hỏi cho người dùng. **Không viết code trước khi người dùng trả lời.** Câu hỏi tối thiểu:
- Robot arm nào, driver nào, tần số và có timestamp phần cứng không?
- Gripper điện: giao thức, tần số trạng thái T_g, có timestamp không?
- Camera: loại, độ phân giải/fps, kết nối nào, hiện đọc bằng gì (OpenCV, RealSense...)?
- Vòng lặp deploy hiện tại nằm ở file nào, chạy trên máy nào, GPU nào, T_inf ước lượng?
- Dataset LeRobot được dựng thế nào (cột nào, tần số nào)?

**Nguyên tắc:**
- Giải thích khái niệm trước (5–10 dòng), chờ người dùng đồng ý, rồi mới viết code. Người dùng muốn hiểu bản chất, không chỉ nhận code chạy được.
- Không sửa repo UMI gốc hay LeRobot gốc; đặt code của dự án ở module riêng. Mọi giá trị trễ nằm trong **một file cấu hình** (không viết cứng như `0.17`).
- Mọi giả định phải gắn nhãn [ĐÃ KIỂM CHỨNG]/[SUY LUẬN]/[CHƯA KIỂM TRA] trong báo cáo.
- Làm theo giai đoạn nhỏ; mỗi giai đoạn có tiêu chí nghiệm thu và báo số đo.

## 12. Kế hoạch theo giai đoạn

**G0. Khảo sát.** Như mục 11. Đầu ra: sơ đồ luồng dữ liệu và thời gian của pipeline hiện tại.

**G1. Công cụ đo.** Chỉnh các script calibrate của UMI cho phần cứng của dự án (hoặc viết tương đương). Đo ℓ_c (AVG, STD), ℓ_g, ℓ_r, ℓ^a_g, ℓ^a_r, T_m của từng luồng, T_inf thực tế. Ghi vào file cấu hình. Nghiệm thu: mỗi đại lượng ≥ 5 lần đo; có báo cáo trung bình ± độ lệch chuẩn; kiểm tra `ℓ < T_signal/2` cho phép đo cross-correlation; kiểm tra điều kiện `ℓ_c ≥ ℓ_m + T_m` và nêu rõ đang ở trường hợp nào.

**G2. Lớp thu thập có nhãn thời gian.** Camera đọc nền (process hoặc thread) lưu **ring buffer** (nhãn `t̃ = t^rx − ℓ_c`, khung) với kích thước ≥ ℓ_c + (n_o − 1)·d·Δt + biên; mọi luồng dùng **cùng đồng hồ**. Nghiệm thu: log khoảng hở lớn nhất của `t̃` trong buffer (≤ 1,5·T_c), không rớt khung bất thường.

**G3. Căn chỉnh quan sát.** Hàm `get_obs_aligned()` theo mục 5, có kiểm tra điều kiện không ngoại suy (fail-loud, đếm vi phạm), slerp cho quay, lerp cho vị trí và độ mở, nearest-neighbor cho ảnh. Viết unit test: lerp/slerp trên tín hiệu giả lập, truy vấn ngoài khoảng dữ liệu phải bị phát hiện, trường hợp rotvec gập tại π. Nghiệm thu: chạy tín hiệu giả lập với độ trễ biết trước, sai số căn chỉnh nằm trong cận h²/8·max|p″|.

**G4. Căn chỉnh hành động.** Tính τ_k, k_min, lọc action quá hạn, gửi waypoint tại `τ_k − ℓ^a` cho từng phần cứng riêng; kiểm tra điều kiện `H ≥ k_min + K` và `T_inf < K·Δt` và in cảnh báo nếu vi phạm; xử lý tình huống hết quỹ đạo (không giật). Controller của dự án phải nhận waypoint có timestamp và nội suy theo `t_now` (không phải lệnh "gửi ngay"). Nghiệm thu: chạy mô phỏng có độ trễ biết trước, pos(τ_k) trùng a_k trong dung sai.

**G5. Tái cấu trúc vòng lặp.** Từ tuần tự (chạy hết rồi mới suy luận) sang song song: robot tiếp tục theo quỹ đạo trong lúc suy luận. Nghiệm thu: không còn khựng giữa các chunk; log `Over budget` bằng 0 trong 30 phút chạy.

**G6. Kiểm chứng trên robot thật.** So sánh có/không bù trễ trên các tác vụ của dự án (pick, gấp vải, cắm vít), ít nhất 20 lần mỗi điều kiện, báo tỉ lệ thành công. Log mỗi chu kỳ: `t_obs`, `min_m (t̃_m^new − t_obs)`, `k_min`, số action bị bỏ, T_inf, độ dư quỹ đạo. Ngưỡng nghiệm thu cụ thể do người dùng quyết định.

## 13. Báo cáo cuối mỗi giai đoạn

Một đoạn ngắn: đã làm gì, số đo (nếu có), điểm nào còn là [SUY LUẬN], rủi ro còn lại, và đúng một bước tiếp theo đề xuất. Không đưa recap dài.
