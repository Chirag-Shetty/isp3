import json, pickle, numpy as np
from pathlib import Path

with open('fall_rf_model.pkl','rb') as f:
    payload = pickle.load(f)
clf = payload['model']

SNR = 10.0; FRAME_DT = 0.055; WS = 40

def exfeat(pc, td, hd, prev=None):
    if not pc or len(pc)==0:
        return np.zeros(20,np.float32), np.zeros(3,np.float32)
    pts=np.array(pc,dtype=np.float32)
    if pts.shape[1]>4:
        m=pts[:,4]>=SNR
        if m.sum()>0: pts=pts[m]
    if len(pts)==0:
        return np.zeros(20,np.float32), np.zeros(3,np.float32)
    x,y,z=pts[:,0],pts[:,1],pts[:,2]
    d=pts[:,3] if pts.shape[1]>3 else np.zeros(len(pts))
    xm,ym,zm=x.mean(),y.mean(),z.mean()
    r=float(np.sqrt(xm**2+ym**2+zm**2))+1e-8
    vx,vy,vz=float(d.mean())*(xm/r),float(d.mean())*(ym/r),float(d.mean())*(zm/r)
    cv=np.array([vx,vy,vz],dtype=np.float32)
    acc=(cv-prev)/FRAME_DT if prev is not None else np.zeros(3,dtype=np.float32)
    sp=float(np.sqrt(x.var()+y.var())) if len(pts)>1 else 0.0
    hr=float(z.max()-z.min()) if len(pts)>1 else 0.0
    pf=np.array([xm,ym,zm,vx,vy,vz,acc[0],acc[1],acc[2],float(len(pts)),sp,hr],dtype=np.float32)
    tf=np.zeros(6,dtype=np.float32)
    if td and len(td)>0 and len(td[0])>=7:
        tf=np.array(td[0][1:7],dtype=np.float32)
    hf=np.zeros(2,dtype=np.float32)
    if hd and len(hd)>0 and len(hd[0])>=3:
        hf=np.array(hd[0][1:3],dtype=np.float32)
    return np.concatenate([pf,tf,hf]),cv

def w2f(w):
    T,F=w.shape
    xs=np.arange(T,dtype=np.float64)
    fts=[]
    for c in range(F):
        v=w[:,c].astype(np.float64)
        fts+=[v.mean(),v.std(),v.min(),v.max(),v.max()-v.min()]
        fts.append(float(np.polyfit(xs,v,1)[0]) if v.std()>1e-8 else 0.0)
    fts.append(float(np.polyfit(xs,w[:,2].astype(np.float64),1)[0]))
    fts.append(float(np.polyfit(xs,w[:,11].astype(np.float64),1)[0]))
    fts.append(float(np.max(np.abs(w[:,5]))))
    fts.append(float(np.max(np.abs(w[:,8]))))
    return np.array(fts,dtype=np.float32)

def test_folder(folder_name, label):
    p = Path(f'Dataset1/{folder_name}')
    print(f'\n--- {folder_name.upper()} (true={label}) ---')
    for jf in sorted(p.glob('*.json'))[:3]:
        raw=json.load(open(jf,'r',encoding='utf-8'))
        frames=raw['data'] if isinstance(raw,dict) and 'data' in raw else raw
        buf,pv=[],None
        for fr in frames:
            fd=fr.get('frameData',fr)
            feat,pv=exfeat(fd.get('pointCloud',[]),fd.get('trackData',[]),fd.get('heightData',[]),pv)
            buf.append(feat)
        if len(buf)<WS: continue
        win=np.array(buf[:WS],dtype=np.float32)
        fv=w2f(win).reshape(1,-1)
        pr=clf.predict_proba(fv)[0]
        xs=np.arange(WS,dtype=np.float64)
        zslope=float(np.polyfit(xs,win[:,2].astype(np.float64),1)[0])
        print(f'  {jf.name}: P(FALL)={pr[1]:.4f}  pred={"FALL" if pr[1]>0.5 else "NO-FALL"}  '
              f'z_mean={win[:,2].mean():.3f}  z_slope={zslope:.5f}  '
              f'n_pts={win[:,9].mean():.1f}  height_range={win[:,11].mean():.3f}')

test_folder('standing_still', 'NO-FALL')
test_folder('sitting_chair',  'NO-FALL')
test_folder('fall_stand',     'FALL')
test_folder('fall_walk',      'FALL')
