#!/usr/bin/env python3
"""Reproducible, dependency-free benchmark for data-driven oil-water forecasting.

The workflow generates heterogeneous one-dimensional waterflood cases with a
finite-volume Buckley--Leverett solver, trains two light-weight surrogate models
(ridge regression and inverse-distance k-nearest neighbours), and exports tables
and SVG figures for the accompanying EI-style conference paper.
"""
from __future__ import annotations
import csv, math, random, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS, FIGS = ROOT/'data', ROOT/'results', ROOT/'figures'
for p in (DATA, RESULTS, FIGS): p.mkdir(exist_ok=True)

def relperm(s, nw=2.0, no=2.0, swc=0.2, sor=0.2, muw=1.0, muo=5.0):
    se = min(1.0, max(0.0, (s-swc)/(1-swc-sor)))
    krw, kro = se**nw, (1-se)**no
    lw, lo = krw/muw, kro/muo
    return lw/(lw+lo+1e-12)

def simulate(phi, perm, mu_ratio, inj, nx=96, steps=170):
    swc, sor = 0.2, 0.2
    s = [swc]*nx
    dt = 0.23/inj
    breakthrough, cum_oil = None, 0.0
    watercut_curve=[]
    for n in range(steps):
        flux=[0.0]*(nx+1); flux[0]=inj
        for i in range(1,nx):
            flux[i]=inj*relperm(s[i-1], muo=mu_ratio)
        flux[nx]=inj*relperm(s[-1], muo=mu_ratio)
        old=s[:]
        for i in range(nx):
            hetero = 1.0 + 0.35*math.sin(2*math.pi*i/nx)* (1-perm)
            s[i] = min(1-sor, max(swc, old[i] - dt/(phi*hetero)*(flux[i+1]-flux[i])))
        wc = flux[nx]/max(inj,1e-12)
        oil = max(0.0, 1-wc)*inj*dt
        cum_oil += oil
        watercut_curve.append(wc)
        if breakthrough is None and wc > 0.05:
            breakthrough = (n+1)*dt
    if breakthrough is None: breakthrough = steps*dt
    return breakthrough, cum_oil, watercut_curve[-1], watercut_curve

def features(case):
    phi, perm, mu, inj = case
    return [1.0, phi, perm, mu, inj, phi*perm, math.log(mu), inj/phi]

def solve_linear(a,b):
    n=len(b)
    for i in range(n):
        piv=max(range(i,n), key=lambda r: abs(a[r][i])); a[i],a[piv]=a[piv],a[i]; b[i],b[piv]=b[piv],b[i]
        div=a[i][i] or 1e-12
        a[i]=[v/div for v in a[i]]; b[i]/=div
        for r in range(n):
            if r==i: continue
            f=a[r][i]
            a[r]=[av-f*iv for av,iv in zip(a[r],a[i])]; b[r]-=f*b[i]
    return b

def ridge_fit(x,y,lam=1e-3):
    m=len(x[0]); xtx=[[0.0]*m for _ in range(m)]; xty=[0.0]*m
    for row,yi in zip(x,y):
        for i in range(m):
            xty[i]+=row[i]*yi
            for j in range(m): xtx[i][j]+=row[i]*row[j]
    for i in range(1,m): xtx[i][i]+=lam
    return solve_linear(xtx,xty)

def dot(a,b): return sum(i*j for i,j in zip(a,b))
def knn_predict(train_x, train_y, q, k=7):
    ds=[]
    for x,y in zip(train_x,train_y):
        d=math.sqrt(sum((a-b)**2 for a,b in zip(x[1:],q[1:])))
        ds.append((d,y))
    ds.sort(); vals=ds[:k]
    w=[1/(d+1e-6) for d,_ in vals]
    return sum(wi*yi for wi,(_,yi) in zip(w,vals))/sum(w)
def rmse(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/len(a))
def mae(a,b): return sum(abs(x-y) for x,y in zip(a,b))/len(a)
def r2(a,b):
    mu=sum(a)/len(a); return 1-sum((x-y)**2 for x,y in zip(a,b))/(sum((x-mu)**2 for x in a)+1e-12)

def svg_scatter(path, truth, preds, title):
    w,h,pad=640,420,55; mn=min(truth+preds); mx=max(truth+preds)
    def xy(x,y): return pad+(x-mn)/(mx-mn)*(w-2*pad), h-pad-(y-mn)/(mx-mn)*(h-2*pad)
    pts='\n'.join(f'<circle cx="{xy(t,p)[0]:.1f}" cy="{xy(t,p)[1]:.1f}" r="4" fill="#1f77b4" opacity="0.72"/>' for t,p in zip(truth,preds))
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/><text x="{w/2}" y="25" text-anchor="middle" font-size="18">{title}</text><line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{pad}" stroke="#d62728" stroke-width="2"/><line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="black"/><line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="black"/><text x="{w/2}" y="{h-12}" text-anchor="middle">Simulator truth</text><text x="18" y="{h/2}" transform="rotate(-90 18,{h/2})" text-anchor="middle">Surrogate prediction</text>{pts}</svg>')

def main():
    random.seed(20260615)
    cases=[]
    for _ in range(180):
        c=(random.uniform(0.16,0.32), random.uniform(0.05,1.0), random.uniform(1.5,12.0), random.uniform(0.55,1.45))
        bt, oil, wc, curve=simulate(*c); cases.append((c,bt,oil,wc,curve))
    with (DATA/'synthetic_waterflood_cases.csv').open('w',newline='') as f:
        wr=csv.writer(f); wr.writerow(['porosity','perm_contrast','oil_water_viscosity_ratio','injection_rate','breakthrough_pv','cumulative_oil_pv','terminal_watercut'])
        for c,bt,oil,wc,_ in cases: wr.writerow([*map(lambda z:f'{z:.6f}',c),f'{bt:.6f}',f'{oil:.6f}',f'{wc:.6f}'])
    random.shuffle(cases); train=cases[:135]; test=cases[135:]
    trX=[features(c) for c,_,_,_,_ in train]; teX=[features(c) for c,_,_,_,_ in test]
    rows=[]; best_preds={}
    for name,idx in [('breakthrough_pv',1),('cumulative_oil_pv',2),('terminal_watercut',3)]:
        ytr=[r[idx] for r in train]; yte=[r[idx] for r in test]
        beta=ridge_fit(trX,ytr); pr=[dot(beta,x) for x in teX]
        pk=[knn_predict(trX,ytr,x) for x in teX]
        for model,pred in [('Physics-feature ridge',pr),('Local analog kNN',pk)]:
            rows.append([name,model,rmse(yte,pred),mae(yte,pred),r2(yte,pred)])
        best_preds[name]=pk if rmse(yte,pk)<rmse(yte,pr) else pr
        svg_scatter(FIGS/(name+'_parity.svg'), yte, best_preds[name], name.replace('_',' ').title())
    with (RESULTS/'metrics.csv').open('w',newline='') as f:
        wr=csv.writer(f); wr.writerow(['target','model','rmse','mae','r2']); wr.writerows([[a,b,f'{c:.5f}',f'{d:.5f}',f'{e:.4f}'] for a,b,c,d,e in rows])
    # representative saturation profile figure
    _,_,_,_,curve = max(cases, key=lambda r:r[2])
    points=' '.join(f'{40+i*(560/(len(curve)-1)):.1f},{360-v*300:.1f}' for i,v in enumerate(curve))
    (FIGS/'watercut_curve.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="420"><rect width="100%" height="100%" fill="white"/><text x="320" y="25" text-anchor="middle" font-size="18">Water-cut evolution for a high-recovery case</text><polyline points="{points}" fill="none" stroke="#2ca02c" stroke-width="3"/><line x1="40" y1="360" x2="600" y2="360" stroke="black"/><line x1="40" y1="60" x2="40" y2="360" stroke="black"/><text x="320" y="405" text-anchor="middle">Time step</text><text x="18" y="210" transform="rotate(-90 18,210)" text-anchor="middle">Water cut</text></svg>')
    print('Wrote dataset, metrics, and SVG figures.')
if __name__ == '__main__': main()