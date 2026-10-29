# SocialNav trong đám đông giờ tan tầm

Câu hỏi: model SocialNav đã huấn luyện (checkpoint `SocialNav-Qwen2.5-VL-SAFE-GRPO`) có điều hướng tốt tới mục tiêu khi có đám đông kiểu giờ tan tầm hay không? Từ đó xây dựng pipeline sinh dữ liệu huấn luyện point-goal có đám đông. Người đi bộ **không phản ứng** với robot, giống dữ liệu gốc và SocNav-Bench.

Báo cáo đầy đủ (tiếng Việt, gồm TN1–TN8, giới hạn, bước tiếp theo, phụ lục file và lệnh): https://claude.ai/artifact/Fzx5H2shFFBq22NM5TSL16

## Kết quả chính

Benchmark ABotN-PointBench Outdoor, scene `20260227170203`, 15 task × 3 seed. SR_social là SR_new và không va chạm người (khoảng cách tâm < 0.55 m). Khoảng tin cậy 95% lấy từ bootstrap hai tầng (task, rồi seed).

| Thí nghiệm | Kết quả |
|---|---|
| TN2: closed-loop, sim v1 | SR_social rush 0.022 so với original 0.467; hiệu ghép cặp −0.444 [−0.644, −0.244]. Phản thực tế crowd-blind: số va chạm/bước không giảm, −5.7% [−26.3, 11.8] (541 sự kiện thực so với 570 khi phát lại đường original) |
| TN3: probe open-loop | Language head nhận ra người: P(person) 0.75 với người, 0.07 với cột. Linear probe trên action feature phân biệt có/không vật thể với độ chính xác cân bằng 0.78. Waypoint vẫn đi xuyên vật thể: pass_through 0.88 với người (0.87 khi không có gì), 0.93 với bức tường 3 × 2 m |
| TN4: closed-loop, sim v2 | SR_social rush 0.067 so với 0.533; normal −0.311 [−0.533, −0.089]. Rush có 12.4 va chạm/task. Phản thực tế: −15.0% [−36.6, 3.3] (557 so với 471 sự kiện) |
| TN5: case study traj_6 | Đám đông làm kế hoạch lệch 0.13 m, trong khi nhiễu flow-matching là 0.91 m; khoảng hở bước kế tiếp chỉ tăng 6 mm. Trên toàn v2: 71% va chạm có người đã thấy trước trong camera, 30% là người đi cùng chiều từ phía sau |
| TN6: pipeline dataset (pilot) | Expert thành công 0.969 (test, 32 episode) và 0.944 (train, 36 episode); 1 078 / 1 014 sample, trong đó 384 / 307 sample có tương tác |
| TN7: A/B "patience" của expert | 59 cặp, phát lại tất định 59/59; không cải thiện, nên tắt |
| TN8: open-loop có đối chứng không người | Trên sample tương tác: response 0.184 m so với 0.170 m khi xoá người khỏi ảnh; perception 0.014 [−0.016, 0.046]; expert lệch 1.158 m |

Kết luận: model **thấy người nhưng không tránh**. Phần "phản ứng" đo được trong open-loop chủ yếu đến từ lịch sử waypoint (đường vòng của expert), không đến từ hình ảnh: đây là copycat / causal confusion. Model huấn luyện lại cần được đo bằng perception (có đối chứng) cộng closed-loop, không chỉ bằng ADE.

## Code

Phần lớn code nằm trong submodule `ABot-Navigation` (nhánh `crowd-rush-hour`), thư mục `abotn_evaluator/crowd/`:

| File | Vai trò |
|---|---|
| `make_sprites.py`, `make_probe_sprites.py` | Sprite người từ Penn-Fudan; bộ sprite probe (người, silhouette, cột, tường) |
| `sim.py`, `evaluator.py` | Mô phỏng đám đông v2 và evaluator `CrowdPointGoalEvaluator` (biến môi trường `CROWD_CONDITION`, `CROWD_SEED`, `CROWD_SPRITE_DIR`) |
| `sim_v1.py`, `evaluator_v1.py` | Bản v1 giữ nguyên để chạy lại TN2 |
| `analyze.py` | Bootstrap, hiệu ghép cặp, phản thực tế crowd-blind |
| `validate_sim.py` | Kiểm tra độ thực tế của sim (mật độ, tốc độ Weidmann, khoảng cách lân cận, xuyên thân) |
| `probe.py`, `probe_visibility.py`, `probe_analyze.py` | Probe TN3 |
| `case_study.py`, `contacts_all.py` | Video, phân tích một task, và thống kê nguồn gốc va chạm |
| `expert.py`, `datagen.py`, `export.py`, `qc.py` | Pipeline sinh dataset giờ tan tầm |
| `patience_ab.py` | A/B TN7 |
| `nocrowd.py`, `open_loop_eval.py` | Đối chứng không người và chấm open-loop TN8 |

Render server: `render_server/actor_gaussians.py` cùng cờ `--sprite_dir` của `render_sim_dynamic.py`. Khi không có cờ này, người bị bỏ qua mà không báo lỗi.

Trong repo này: `utils/abotn_agent.py` và `configs/abotn_socialnav.yaml` (agent cho runner của ABot), `utils/citywalker.py` (thêm `--seed`), `transformers/src/transformers/modeling_utils.py` (bản vá, xem dưới).

## Những gì có trong git và không có

Có trong git (dưới `outputs/`, thêm bằng `git add -f`):

- launcher `.sh` của mọi thí nghiệm;
- `comparison.json`, `analysis.txt`, `validate_sim.txt`;
- `result.json` của mọi task closed-loop;
- tóm tắt và bản ghi probe;
- `summary.json`, `open_loop.json` và 4 video của case study;
- QC, thống kê export, `open_loop_eval.json` của pilot, và `patience_ab.jsonl`.

Ảnh dùng trong báo cáo nằm ở `docs/crowd_rush_hour/img/`.

Không có trong git, vì quá lớn hoặc không phải của mình (`.gitignore`):

- `checkpoints/` (7.7 GB, ModelScope `zjugyn/SocialNav-Qwen2.5-VL-SAFE-GRPO`);
- `data/` (24 GB: benchmark, sprite, ảnh Penn-Fudan);
- `papers/`;
- phần còn lại của `outputs/` (2 GB): ảnh render, `crowd.npz`, `robot_xy_yaw.npy`, log, episode và ảnh của dataset pilot, dự đoán `pred_*.jsonl`.

Phản thực tế trong `analyze.py`, `contacts_all.py`, `case_study.py` và TN8 cần các file này. Chúng chỉ có trên máy chạy, hoặc phải chạy lại để tạo.

## Môi trường

- 3× A30 24 GB, driver 550 (CUDA 12.4), dùng chung với người khác. `requirements.txt` ghim torch 2.8/cu128, driver này không chạy được, nên `.venv` dùng torch 2.6.0+cu124 và numpy < 2.3.
- Repo HF của SocialNav rỗng; checkpoint lấy từ ModelScope `zjugyn/SocialNav-Qwen2.5-VL-SAFE-GRPO`.
- **Bản vá bắt buộc:** `from_pretrained` của bản transformers đi kèm (khối "# gyn") khởi tạo lại ngẫu nhiên `query_action`, head ActionFormer và pos embedding của `velocity_predictor` sau khi nạp trọng số. Khi đó policy bỏ qua mục tiêu. Bản vá chỉ khởi tạo các khóa bị thiếu. Thiếu bản vá này thì mọi con số đều sai.
- Render server: `render_sim_dynamic.py --port 7036 --gpus 0 --max_scenes_per_gpu 1 --scenes_root data/ABotN-PointBench/Outdoor/occmaps --sprite_dir data/probe_sprites`. Server chỉ liệt kê scene lúc khởi động, nên phải khởi động lại sau khi tải scene mới. Với `--max_scenes_per_gpu 1`, client phải làm lần lượt từng scene.
- `.venv` chỉ đủ cho suy luận. Huấn luyện còn thiếu trl, peft, deepspeed và liger_kernel; trl/peft có thể kéo bản transformers mới đè lên bản vá.

## Tái lập

Chạy với `PYTHONPATH=$PWD:$PWD/ABot-Navigation` và một render server như trên.

| Thí nghiệm | Lệnh |
|---|---|
| TN2 | `bash outputs/crowd_eval/run_all.sh`; `bash outputs/crowd_eval/run_seeds.sh <seed> <gpu>`; rồi `python -m abotn_evaluator.crowd.analyze outputs/crowd_eval` |
| TN3 | `python -m abotn_evaluator.crowd.probe --out outputs/crowd_probe --shard k --num-shards 2`; rồi `probe_visibility` và `probe_analyze` |
| TN4 | `bash outputs/crowd_eval_v2/run_v2.sh <seed> <gpu> original normal rush`; `analyze outputs/crowd_eval_v2`; `python -m abotn_evaluator.crowd.validate_sim` |
| TN5 | `python -m abotn_evaluator.crowd.case_study --task traj_6 --seed 0 --runs outputs/crowd_eval_v2 --out outputs/crowd_case`; `python -m abotn_evaluator.crowd.contacts_all outputs/crowd_eval_v2` |
| TN6 | `bash outputs/rush_dataset/pilot/run_pilot.sh`; rồi `python -m abotn_evaluator.crowd.qc --root outputs/rush_dataset/pilot --split test` |
| TN7 | `python -m abotn_evaluator.crowd.patience_ab --root outputs/rush_dataset/pilot` |
| TN8 | `nocrowd --root outputs/rush_dataset/pilot --split test`; `utils/citywalker.py --seed 0` trên `citywalker_test.jsonl` và `citywalker_test_nocrowd.jsonl`; `open_loop_eval --pred ... --control ...` |
| Full-scale (chưa chạy) | `bash outputs/rush_dataset/run_full.sh outputs/rush_dataset/v1 train 160 0,1,2`, rồi `... test 24 0,1,2` |

Lệnh đầy đủ có trong phụ lục của báo cáo.
