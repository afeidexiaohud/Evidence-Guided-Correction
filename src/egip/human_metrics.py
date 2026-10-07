"""Frozen independent Human-Gold status and character-span evaluation."""
from collections import Counter
import numpy as np

STATUSES = {"CONFLICT", "NO_CONFLICT", "UNCERTAIN", "UNSCORABLE"}

def spans(text, items, *, disjoint=False):
    out=[]
    for s in items:
        a,b=s['start'],s['end']
        if type(a) is not int or type(b) is not int or not 0<=a<b<=len(text):
            raise ValueError('Invalid Unicode half-open span')
        if s['text']!=text[a:b]: raise ValueError('Substring mismatch')
        out.append((a,b))
    if len(out)!=len(set(out)): raise ValueError('Duplicate spans')
    out.sort()
    if disjoint and any(b>c for (a,b),(c,d) in zip(out,out[1:])):
        raise ValueError('Gold spans must be disjoint')
    return out

def chars(ss):return {i for a,b in ss for i in range(a,b)}

def iou(a,b):
    inter=max(0,min(a[1],b[1])-max(a[0],b[0]))
    return inter/(a[1]-a[0]+b[1]-b[0]-inter)

def prf(tp,p,g):
    return {'precision':tp/p if p else 0.,'recall':tp/g if g else 0.,'f1':2*tp/(p+g) if p+g else 0.}

def matching(pred,gold,threshold=.5):
    """Maximum cardinality bipartite matching; never match one span twice."""
    edges=[[j for j,g in enumerate(gold) if iou(p,g)>=threshold] for p in pred];owners={}
    def visit(i,seen):
        for j in edges[i]:
            if j in seen:continue
            seen.add(j)
            if j not in owners or visit(owners[j],seen):owners[j]=i;return True
        return False
    for i in range(len(pred)):visit(i,set())
    return sorted((i,j) for j,i in owners.items())

def pair_metrics(pred,gold):
    p,g=chars(pred),chars(gold);tp=len(p&g)
    matched=matching(pred,gold)
    return {'counts':[tp,len(p),len(g)],'character':prf(tp,len(p),len(g)),
            'exact_span_set':pred==gold,'best_gold_span_iou_mean':sum(max((iou(x,y) for x in pred),default=0.) for y in gold)/len(gold) if gold else None,
            'overreach_characters':len(p-g),'underreach_characters':len(g-p),
            'overreach_fraction':len(p-g)/len(p) if p else 0.,'underreach_fraction':len(g-p)/len(g) if g else None,
            'matched_spans':len(matched),'gold_spans':len(gold),'predicted_spans':len(pred),
            'span_prf_iou_0_5':prf(len(matched),len(pred),len(gold))}

def ranking_metrics(text,candidates,rank,gold,k=1):
    if type(k) is not int or k<1:raise ValueError('Positive integer k required')
    if any(type(x) is not int for x in rank) or sorted(rank)!=list(range(len(candidates))):raise ValueError('Full rank required for MRR')
    cs=[spans(text,[c])[0] for c in candidates];gs=spans(text,gold,disjoint=True)
    if not gs:raise ValueError('Ranking needs localizable CONFLICT gold')
    hits=[any(iou(cs[i],g)>=.5 for g in gs) for i in rank]
    first=next((i for i,h in enumerate(hits,1) if h),None)
    result=pair_metrics(sorted(cs[i] for i in rank[:k]),gs)
    result.update(hit_at_k=bool(any(hits[:k])),mrr=1/first if first else 0.,k=k)
    return result

def evaluate(labels,predictions):
    """One dataset per call; require even excluded items' frozen predictions."""
    ids=[r['item_id'] for r in labels]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate label IDs')
    if set(predictions)!=set(ids):raise ValueError('Prediction ID mismatch; missing output is not silently dropped')
    if len({r['dataset'] for r in labels})>1:raise ValueError('Primary evaluation must be dataset-wise')
    counts=Counter();items=[];nonlocalizable=0
    for r in labels:
        status=r['factual_status']
        if status not in STATUSES:raise ValueError('Unknown or missing factual status')
        counts[status]+=1
        gs=spans(r['claim'],r['spans'],disjoint=True);ps=spans(r['claim'],predictions[r['item_id']])
        reason=r.get('span_unlocalizable_reason','')
        if status!='CONFLICT':
            if gs or reason:raise ValueError('Only CONFLICT has spans/unlocalizable reason')
            continue
        if not gs:
            if not reason:raise ValueError('CONFLICT needs spans or an explicit unlocalizable reason')
            nonlocalizable+=1;continue
        if reason:raise ValueError('Localizable CONFLICT cannot also be unlocalizable')
        items.append(dict(item_id=r['item_id'],cluster_id=r['cluster_id'],**pair_metrics(ps,gs)))
    total=[sum(x['counts'][i] for x in items) for i in range(3)]
    return {'sample_records':len(labels),'factual_status_counts':{s:counts[s] for s in sorted(STATUSES)},
            'conflict_unlocalizable':nonlocalizable,'localization_records':len(items),
            'character_micro':prf(*total) if items else None,'items':items,
            'note':'NO_CONFLICT/UNCERTAIN/UNSCORABLE excluded from error-conditioned F1; never reclassified as model false positives'}

def paired_cluster_ci(a,b,*,seed=20260910,resamples=5000):
    """Percentile CI for paired micro-F1 delta. Inputs are scored localizable items."""
    def index(rows):
        d={r['item_id']:r for r in rows}
        if len(d)!=len(rows):raise ValueError('Duplicate scored IDs')
        return d
    aa,bb=index(a),index(b)
    if not aa or set(aa)!=set(bb):raise ValueError('Nonempty identical paired IDs required')
    grouped={}
    for key in sorted(aa):
        if aa[key]['cluster_id']!=bb[key]['cluster_id']:raise ValueError('Cluster mismatch')
        grouped.setdefault(aa[key]['cluster_id'],[]).append(key)
    keys=sorted(grouped);arr=np.array([[ [sum(d[i]['counts'][j] for i in grouped[k]) for j in range(3)] for d in [aa,bb]] for k in keys],float)
    def delta(x):
        f=lambda c:2*c[0]/(c[1]+c[2]) if c[1]+c[2] else 0.
        return f(x[0])-f(x[1])
    point=delta(arr.sum(axis=0))
    if len(keys)<2:return {'delta':point,'ci95':None,'clusters':len(keys),'reason':'Fewer than two clusters'}
    if resamples<2:raise ValueError('At least two bootstrap resamples')
    rng=np.random.default_rng(seed);values=[delta(arr[rng.integers(0,len(keys),len(keys))].sum(axis=0)) for _ in range(resamples)]
    return {'delta':point,'ci95':np.quantile(values,[.025,.975]).tolist(),'clusters':len(keys),'seed':seed,'resamples':resamples,'scope':'fixed checkpoints, sampled source clusters; not training randomness'}

def status_agreement(a,b):
    if set(a)!=set(b) or not a:raise ValueError('Nonempty paired IDs required')
    if any(v not in STATUSES for v in list(a.values())+list(b.values())):raise ValueError('Invalid status')
    n=len(a);pa=Counter(a.values());pb=Counter(b.values());po=sum(a[k]==b[k] for k in a)/n;pe=sum(pa[s]*pb[s] for s in STATUSES)/n**2
    return {'n':n,'raw_agreement':po,'cohen_kappa':(po-pe)/(1-pe) if pe<1 else None,'confusion':{s:{t:sum(a[k]==s and b[k]==t for k in a) for t in sorted(STATUSES)} for s in sorted(STATUSES)}}

def boundary_agreement(pred,gold):
    """Maximum total IoU assignment; unmatched/zero-overlap are not hidden."""
    from scipy.optimize import linear_sum_assignment
    if not pred or not gold:
        return {'pairs':[],'unmatched_predicted':len(pred),'unmatched_gold':len(gold)}
    costs=np.array([[iou(p,g) for g in gold] for p in pred])
    rows,cols=linear_sum_assignment(-costs)
    pairs=[{'pred_index':int(i),'gold_index':int(j),'iou':float(costs[i,j]),
            'left_boundary_absolute_error':abs(pred[i][0]-gold[j][0]),
            'right_boundary_absolute_error':abs(pred[i][1]-gold[j][1])}
           for i,j in zip(rows,cols) if costs[i,j]>0]
    return {'pairs':pairs,'unmatched_predicted':len(pred)-len(pairs),'unmatched_gold':len(gold)-len(pairs)}
