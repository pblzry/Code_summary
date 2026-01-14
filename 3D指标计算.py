def dice_score(pred, gt, label):
    pred_bin = (pred == label).astype(np.uint8)
    gt_bin = (gt == label).astype(np.uint8)
    inter = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin)
    if union == 0:
        return 1.0
    return 2 * inter / union

def hd95(pred, gt, label):
    pred_bin = (pred == label).astype(np.uint8)
    gt_bin = (gt == label).astype(np.uint8)
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 100.0
    dt_pred = distance_transform_edt(1 - pred_bin)
    dt_gt = distance_transform_edt(1 - gt_bin)
    pred_border = pred_bin - binary_erosion(pred_bin, structure=np.ones((3,3,3)))
    gt_border = gt_bin - binary_erosion(gt_bin, structure=np.ones((3,3,3)))
    pred_to_gt = dt_gt[pred_border.astype(bool)]
    gt_to_pred = dt_pred[gt_border.astype(bool)]
    return np.percentile(np.hstack([pred_to_gt, gt_to_pred]), 95)

def assd(pred, gt, label):
    pred_bin = (pred == label).astype(np.uint8)
    gt_bin = (gt == label).astype(np.uint8)
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 100.0
    dt_pred = distance_transform_edt(1 - pred_bin)
    dt_gt = distance_transform_edt(1 - gt_bin)
    pred_border = pred_bin - binary_erosion(pred_bin, structure=np.ones((3,3,3)))
    gt_border = gt_bin - binary_erosion(gt_bin, structure=np.ones((3,3,3)))
    pred_to_gt = dt_gt[pred_border.astype(bool)]
    gt_to_pred = dt_pred[gt_border.astype(bool)]
    return (pred_to_gt.sum() + gt_to_pred.sum()) / (len(pred_to_gt) + len(gt_to_pred))

def nsd(pred, gt, label, tau=1):
    pred_bin = (pred == label).astype(np.uint8)
    gt_bin = (gt == label).astype(np.uint8)
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 1.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 0.0
    dt_pred = distance_transform_edt(1 - pred_bin)
    dt_gt = distance_transform_edt(1 - gt_bin)
    pred_border = pred_bin - binary_erosion(pred_bin, structure=np.ones((3,3,3)))
    gt_border = gt_bin - binary_erosion(gt_bin, structure=np.ones((3,3,3)))
    pred_in_gt = np.sum(dt_gt[pred_border.astype(bool)] <= tau)
    gt_in_pred = np.sum(dt_pred[gt_border.astype(bool)] <= tau)
    denom = np.sum(pred_border) + np.sum(gt_border)
    if denom == 0:
        return 1.0
    return (pred_in_gt + gt_in_pred) / denom