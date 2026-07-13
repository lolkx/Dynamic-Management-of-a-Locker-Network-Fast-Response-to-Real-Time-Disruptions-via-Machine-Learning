"""
ml_common.py
------------
Shared ML plumbing used by train_model_C.py / train_model_D.py and by
generate_instances_B.py's --report overflow-rate check. Relocated from
train_model_B.py (now deleted) so neither script depends on a "Model B"
training script that no longer exists.
"""

from __future__ import annotations


# =============================================================================
# REAL OVERFLOW LABEL (used by generate_instances_B.py --report)
# =============================================================================

def saturation_label_overflow(orders: list[dict],
                               locker_cap: dict) -> dict[int, int]:
    """
    Binary overflow label: 1 if delivery orders at locker k cannot all fit
    in its compartments (greedy largest-first bin assignment).

    Returns {location_id: 0|1} for every location with at least one delivery.
    """
    del_by_loc: dict[int, dict[int, int]] = {}
    for o in orders:
        if o['type'] == 'delivery':
            k = o['location']
            s = int(o.get('size', 1))
            by_size = del_by_loc.setdefault(k, {})
            by_size[s] = by_size.get(s, 0) + 1

    labels: dict[int, int] = {}
    for k, by_size in del_by_loc.items():
        cap = locker_cap.get(k)
        if not cap:
            labels[k] = 0
            continue
        avail, overflow = dict(cap), False
        for s in [3, 2, 1]:
            n_ord = by_size.get(s, 0)
            for comp_s in range(s, 4):
                if n_ord <= 0: break
                use = min(n_ord, avail.get(comp_s, 0))
                avail[comp_s] = avail.get(comp_s, 0) - use
                n_ord -= use
            if n_ord > 0:
                overflow = True; break
        labels[k] = 1 if overflow else 0
    return labels


# =============================================================================
# MODEL FACTORIES
# =============================================================================

def _make_rf(**kw):
    from sklearn.ensemble import RandomForestClassifier
    p = dict(n_estimators=300, min_samples_leaf=5,
             class_weight='balanced', random_state=42, n_jobs=-1)
    p.update(kw); return RandomForestClassifier(**p)

def _make_gbm(**kw):
    from sklearn.ensemble import GradientBoostingClassifier
    p = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
             subsample=0.8, random_state=42)
    p.update(kw); return GradientBoostingClassifier(**p)

def _make_lr(**kw):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    p = dict(max_iter=1000, class_weight='balanced', random_state=42)
    p.update(kw)
    return Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(**p))])

def _make_xgb(**kw):
    from xgboost import XGBClassifier
    p = dict(n_estimators=300, max_depth=5, learning_rate=0.1,
             subsample=0.8, colsample_bytree=0.8, tree_method='hist',
             n_jobs=-1, random_state=42, eval_metric='logloss')
    p.update(kw); return XGBClassifier(**p)

MODEL_FACTORIES = {'rf': _make_rf, 'gbm': _make_gbm, 'lr': _make_lr, 'xgb': _make_xgb}


# =============================================================================
# CV
# =============================================================================

CV_FOLDS = 5

def _cv_scores(X, y, model_type, cv):
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import make_scorer, f1_score
    clf = MODEL_FACTORIES[model_type]()
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    res = cross_validate(clf, X, y, cv=skf, n_jobs=-1, scoring={
        'accuracy': 'accuracy',
        'f1':       make_scorer(f1_score, zero_division=0),
        'roc_auc':  'roc_auc',
    })
    return {k.replace('test_', '') + '_mean': v.mean()
            for k, v in res.items() if k.startswith('test_')} | \
           {k.replace('test_', '') + '_std':  v.std()
            for k, v in res.items() if k.startswith('test_')}
