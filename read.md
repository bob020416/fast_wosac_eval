# wosac_eval

## Install

```bash
cd /home/hcis-s26/Yuhsiang/wosac_eval
pip install -r requirements.txt
```

如果要跑 `2025` WOSAC，Waymo package 一定要更新到支援 `traffic_light_violation` 的版本。

建議直接安裝：

```bash
pip install --no-deps --upgrade waymo-open-dataset-tf-2-12-0==1.6.7
```

安裝後請確認這個欄位存在：

```bash
python - <<'PY'
from waymo_open_dataset.protos import sim_agents_metrics_pb2
print(sorted(sim_agents_metrics_pb2.SimAgentMetricsConfig().DESCRIPTOR.fields_by_name.keys()))
PY
```

輸出裡必須要有：

```bash
traffic_light_violation
```

## 1. Prepare GT

```bash
cd /home/hcis-s26/Yuhsiang/wosac_eval
TF_CPP_MIN_LOG_LEVEL=2 python prepare_gt.py \
  /home/hcis-s26/Yuhsiang/catk/cache/SMART/validation_tfrecords_splitted_full \
  --output-dir /home/hcis-s26/Yuhsiang/wosac_eval/data/waymo_processed/validation_gt
```

## 2. Evaluate Standard Rollouts

如果 rollout pickle 直接包含：

- `agent_id`
- `simulated_states`

就直接跑：

```bash
cd /home/hcis-s26/Yuhsiang/wosac_eval
python wosac_eval.py /home/hcis-s26/Yuhsiang/catk/vis/catk_sketch_1000 \
  --gt-dir /home/hcis-s26/Yuhsiang/wosac_eval/data/waymo_processed/validation_gt \
  --version 2025 \
  --rollout-key baseline

cd /home/hcis-s26/Yuhsiang/wosac_eval
python wosac_eval.py /home/hcis-s26/Yuhsiang/catk/vis/catk_latent_statictime_200 \
  --gt-dir /home/hcis-s26/Yuhsiang/wosac_eval/data/waymo_processed/validation_gt \
  --version 2025 \
  --rollout-key controlnet
  
```

## 3. Evaluate CATK Rollouts

CATK 這種 `_closedloop_wosac.pkl` 格式也可以直接吃。

跑 `baseline`：

```bash
cd /home/hcis-s26/Yuhsiang/wosac_eval
python wosac_eval.py /home/hcis-s26/Yuhsiang/catk/vis/catk_sketch_1000 \
  --gt-dir /home/hcis-s26/Yuhsiang/wosac_eval/data/waymo_processed/validation_gt \
  --version 2025 \
  --rollout-key baseline
```

跑 `controlnet`：

```bash
cd /home/hcis-s26/Yuhsiang/wosac_eval
python wosac_eval.py /home/hcis-s26/Yuhsiang/catk/vis/catk_sketch_1000 \
  --gt-dir /home/hcis-s26/Yuhsiang/wosac_eval/data/waymo_processed/validation_gt \
  --version 2025 \
  --rollout-key controlnet
```

如果你目前環境還沒升級 Waymo，也可以先跑 `2024`：

```bash
python wosac_eval.py /home/hcis-s26/Yuhsiang/catk/vis/catk_sketch_1000 \
  --gt-dir /home/hcis-s26/Yuhsiang/wosac_eval/data/waymo_processed/validation_gt \
  --version 2024 \
  --rollout-key controlnet
```

## 4. Output

輸出 report 預設會寫到：

```bash
<rollout_dir>/wosac_metrics_report.json
```
