# Helmet Model Results

## Run Summary

- Run directory: `runs/train/helmet_detector`
- Dataset config: `configs/helmet_dataset.yaml`
- Weights source: `yolov5s.pt`
- Best weights file: `runs/train/helmet_detector/weights/best.pt`
- Best weights size: `56,745,269` bytes
- Configured epochs: `80`
- Logged epochs in `results.csv`: `7` (`epoch 0` through `epoch 6`)

## Dataset Classes

- `Helmet`
- `No Helmet`
- `Worker`

## Best Logged Metrics

These are the best values found in `runs/train/helmet_detector/results.csv`:

- Best precision: `0.8752`
- Best recall: `0.5727`
- Best mAP@0.5: `0.57987`
- Best mAP@0.5:0.95: `0.3193`

## Last Logged Epoch

The final row currently recorded in `results.csv` is `epoch 6`:

- Train box loss: `0.034749`
- Train obj loss: `0.027923`
- Train cls loss: `0.0027395`
- Validation box loss: `0.034909`
- Validation obj loss: `0.019862`
- Validation cls loss: `0.0010377`
- Precision: `0.86861`
- Recall: `0.5727`
- mAP@0.5: `0.57987`
- mAP@0.5:0.95: `0.3193`

## Interpretation

- Precision is fairly strong, which means the model is reasonably careful when it predicts detections.
- Recall is noticeably lower than precision, which means it is still missing a meaningful number of true objects.
- `mAP@0.5 = 0.57987` suggests the model is learning, but it is not yet especially strong for production-quality helmet monitoring.
- `mAP@0.5:0.95 = 0.3193` shows box quality and consistency across stricter IoU thresholds still need improvement.
- Because the CSV only contains `7` logged epochs even though the run was configured for `80`, this result looks like an early/interrupted training run rather than a fully completed one.

## Files Used For This Summary

- `runs/train/helmet_detector/results.csv`
- `runs/train/helmet_detector/opt.yaml`
- `runs/train/helmet_detector/weights/best.pt`
