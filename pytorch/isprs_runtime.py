"""Shared Stage 5A configuration, metric and checkpoint helpers."""
from __future__ import annotations
import ast
from pathlib import Path
import numpy as np

def load_config(path):
    out={}
    for raw in Path(path).read_text().splitlines():
        line=raw.split('#',1)[0].strip()
        if not line: continue
        key,value=line.split(':',1); value=value.strip()
        normalized={"true":"True","false":"False","null":"None"}.get(value,value)
        try: out[key]=ast.literal_eval(normalized)
        except (ValueError,SyntaxError): out[key]=value
    return out

def confusion_matrix(target, pred, classes=9):
    target=np.asarray(target).reshape(-1); pred=np.asarray(pred).reshape(-1)
    if target.shape != pred.shape: raise ValueError("target/prediction shapes differ")
    if target.size and ((target<0).any() or (target>=classes).any() or (pred<0).any() or (pred>=classes).any()):
        raise ValueError("metric labels outside class range")
    return np.bincount(classes*target+pred,minlength=classes**2).reshape(classes,classes)

def metric_report(cm):
    cm=np.asarray(cm,dtype=np.float64); tp=np.diag(cm); actual=cm.sum(1); predicted=cm.sum(0)
    union=actual+predicted-tp
    div=lambda a,b: np.divide(a,b,out=np.zeros_like(a),where=b!=0)
    iou=div(tp,union); precision=div(tp,predicted); recall=div(tp,actual); f1=div(2*precision*recall,precision+recall)
    return {"per_class_iou":iou,"miou":float(iou.mean()),"overall_accuracy":float(tp.sum()/cm.sum()) if cm.sum() else 0.,
            "precision":precision,"recall":recall,"f1":f1}
