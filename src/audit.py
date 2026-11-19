"""Audit original images; conservative duplicate grouping, deterministic held-out split."""
import hashlib,json,time,zipfile,itertools
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('data/raw/COVID-19_Radiography_Dataset'); OUT=Path('results/audit')
CLASSES=['Normal','COVID','Viral Pneumonia']; SEED=20261115
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def metadata(path):
    z=zipfile.ZipFile(path)
    strings=[''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',NS)]
    result={}
    for r in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:row',NS)[1:]:
        vals={c.get('r')[0]:strings[int(c.find('s:v',NS).text)] if c.get('t')=='s' else ''.join(c.itertext()) for c in r}
        result[vals['A'].casefold()]=vals
    return result

def inspect(p):
    r=dict(path=str(p),label=p.parents[1].name,filename=p.name,patient_id='',error='')
    try:
        b=p.read_bytes(); im=Image.open(p); im.load(); a=np.asarray(im); gray=im.convert('L')
        small=np.asarray(gray.resize((32,32),Image.Resampling.LANCZOS),dtype=float)
        v=dctn(small,norm='ortho')[:8,:8].ravel(); bits=v>np.median(v[1:]); bits[0]=False
        r.update(width=im.width,height=im.height,mode=im.mode,channels=1 if a.ndim==2 else a.shape[-1],dtype=str(a.dtype),intensity_min=int(a.min()),intensity_max=int(a.max()),intensity_mean=float(a.mean()),sha256=hashlib.sha256(b).hexdigest(),pixel_sha256=hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest(),phash=f'{int.from_bytes(np.packbits(bits).tobytes()):016x}')
    except Exception as e:r['error']=repr(e)
    return r

def ssim(a,b):
    ux,uy=uniform_filter(a,7),uniform_filter(b,7)
    vx,vy=uniform_filter(a*a,7)-ux*ux,uniform_filter(b*b,7)-uy*uy
    cov=uniform_filter(a*b,7)-ux*uy
    v=((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
    return float(v[3:-3,3:-3].mean())

def main():
    start=time.time();OUT.mkdir(parents=True,exist_ok=True)
    paths=sorted(ROOT.glob('*/images/*')); print('Audit images',len(paths),flush=True)
    with ThreadPoolExecutor(8) as pool: rows=list(pool.map(inspect,paths))
    df=pd.DataFrame(rows); unmatched=[]
    for label in df.label.unique():
        meta=metadata(ROOT/f'{label}.metadata.xlsx')
        for i in df.index[df.label==label]:
            m=meta.get(Path(df.at[i,'path']).stem.casefold(),{})
            df.at[i,'source']=m.get('D','unknown');df.at[i,'metadata_size']=m.get('C','')
            if not m:unmatched.append(df.at[i,'path'])
    print('Images read; hashes ready',flush=True)
    parent=list(range(len(df)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    exact=[]
    for key,group in df[df.error==''].groupby('pixel_sha256'):
        ids=group.index.tolist()
        for i in ids[1:]:union(ids[0],i);exact.append(dict(a=ids[0],b=i,kind='exact_pixels'))
    # All 64-bit pHash pairs at Hamming <=4: one of five disjoint bit blocks must match.
    buckets=defaultdict(list); pairs=set(); hashes={i:int(df.at[i,'phash'],16) for i in df.index[df.error=='']}
    for i,h in hashes.items():
        for k,(shift,width) in enumerate([(0,13),(13,13),(26,13),(39,13),(52,12)]):
            key=(k,(h>>shift)&((1<<width)-1))
            for j in buckets[key]:
                if (h^hashes[j]).bit_count()<=4:pairs.add((j,i))
            buckets[key].append(i)
    candidates=[];cache={}
    def thumbnail(i):
        if i not in cache:cache[i]=np.asarray(Image.open(df.at[i,'path']).convert('L').resize((64,64),Image.Resampling.LANCZOS),dtype=np.float32)/255
        return cache[i]
    for a,b in sorted(pairs):
        if df.at[a,'pixel_sha256']==df.at[b,'pixel_sha256']:continue
        x,y=thumbnail(a),thumbnail(b);score=ssim(x,y);mad=float(np.abs(x-y).mean())
        confirmed=score>=.98 and mad<=.02
        # Keep high similarity candidates together too; low-SSIM pHash collisions are distinct radiographs.
        if score>=.90: union(a,b)
        candidates.append(dict(a=a,b=b,path_a=df.at[a,'path'],path_b=df.at[b,'path'],label_a=df.at[a,'label'],label_b=df.at[b,'label'],hamming=(hashes[a]^hashes[b]).bit_count(),ssim=score,mean_absolute_difference=mad,confirmed_visual=confirmed))
    pd.DataFrame(candidates,columns=['a','b','path_a','path_b','label_a','label_b','hamming','ssim','mean_absolute_difference','confirmed_visual']).to_csv(OUT/'visual_candidates.csv',index=False)
    pd.DataFrame(exact,columns=['a','b','kind']).to_csv(OUT/'exact_pairs.csv',index=False)
    df['duplicate_group']=[f'dup_{find(i):05d}' for i in df.index]
    conflicts=df.groupby('duplicate_group').label.nunique();conflicts=set(conflicts[conflicts>1].index)
    df['split']='excluded_lung_opacity';df.loc[df.label.isin(CLASSES),'split']='unassigned'
    df.loc[df.duplicate_group.isin(conflicts),'split']='excluded_label_conflict'
    df.loc[df.error!='','split']='excluded_unreadable'
    df[df.duplicate_group.isin(conflicts)].to_csv(OUT/'label_conflicts.csv',index=False)
    rng=np.random.default_rng(SEED)
    for label in CLASSES:
        sub=df[(df.label==label)&(df.split=='unassigned')]
        groups=list(sub.groupby('duplicate_group').groups.items());rng.shuffle(groups)
        # Larger groups first; random tie order. Greedy fill proportional deficit.
        groups.sort(key=lambda x:-len(x[1]));target=np.array([.7,.15,.15])*len(sub);counts=np.zeros(3,int)
        for _,ids in groups:
            slot=int(np.argmax((target-counts)/np.maximum(target,1)))
            df.loc[list(ids),'split']=['train','val','test'][slot];counts[slot]+=len(ids)
    used=df[df.split.isin(['train','val','test'])]
    assert used.groupby('duplicate_group').split.nunique().max()==1
    df.to_csv('data/processed/manifest.csv',index=False)
    used[['path','label','duplicate_group','split']].to_csv('data/processed/splits.csv',index=False)
    sources=pd.crosstab(df.source,df.label);sources.to_csv(OUT/'source_by_class.csv')
    pd.crosstab([used.source,used.label],used.split).to_csv(OUT/'source_class_split.csv')
    counts=pd.crosstab(df.label,df.split);counts.to_csv(OUT/'split_counts.csv')
    for label in CLASSES:
        sample=df[(df.label==label)&(df.error=='')].sample(12,random_state=SEED)
        fig,axes=plt.subplots(3,4,figsize=(10,8))
        for ax,(_,r) in zip(axes.ravel(),sample.iterrows()):ax.imshow(Image.open(r.path),cmap='gray');ax.set_title(r.filename,fontsize=8);ax.axis('off')
        fig.suptitle(label);fig.tight_layout();fig.savefig(OUT/f'sample_{label.replace(" ","_")}.png');plt.close(fig)
    show=sorted(candidates,key=lambda r:abs(r["ssim"]-.98))[:12]
    if show:
        fig,axs=plt.subplots(len(show),2,figsize=(6,2.5*len(show)))
        for axrow,r in zip(np.atleast_2d(axs),show):
            for ax,key in zip(axrow,['path_a','path_b']):ax.imshow(Image.open(r[key]),cmap='gray');ax.set_title(Path(r[key]).name,fontsize=8);ax.axis('off')
        fig.tight_layout();fig.savefig(OUT/'candidate_contact_sheet.png',dpi=65);plt.close(fig)
    summary=dict(seed=SEED,total_images=len(df),counts=df.label.value_counts().to_dict(),unreadable=df[df.error!=''].path.tolist(),unmatched_metadata=unmatched,dimensions=df.groupby(['label','width','height']).size().to_dict().__str__(),modes=df.groupby(['label','mode']).size().to_dict().__str__(),intensity_range=[int(df.intensity_min.min()),int(df.intensity_max.max())],exact_byte_duplicate_excess=int(df.sha256.duplicated().sum()),exact_pixel_duplicate_excess=int(df.pixel_sha256.duplicated().sum()),exact_pairs=len(exact),visual_candidates=len(candidates),confirmed_visual=sum(r['confirmed_visual'] for r in candidates),conflicting_groups=len(conflicts),excluded_conflicts=int(df.duplicate_group.isin(conflicts).sum()),split_counts=counts.to_dict(),mask_counts={p.parent.name:len(list(p.glob('*'))) for p in ROOT.glob('*/masks')},seconds=time.time()-start)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    delta=[];info={}
    for r in df[df['mode']=='RGB'].itertuples():
        a=np.asarray(Image.open(r.path)).astype(int);delta.append(int(np.max(a.max(-1)-a.min(-1))))
    for label in df.label.unique():
        for r in df[df.label==label].head(5).itertuples():info[r.path]=Image.open(r.path).info
    details=dict(phash_radius=4,visual_confirmed_threshold={'ssim64':.98,'mean_absolute_difference':.02},conservative_group_threshold_ssim=.90,grouped_candidate_pairs=sum(r['ssim']>=.90 for r in candidates),exact_excess_by_class=df.groupby('label').pixel_sha256.apply(lambda x:int(x.duplicated().sum())).to_dict(),group_sizes=df.groupby('duplicate_group').size().value_counts().sort_index().to_dict(),rgb_count=len(delta),rgb_channels_identical=all(d==0 for d in delta),rgb_max_channel_difference=max(delta,default=0),png_info_sample=info)
    (OUT/'duplicate_details.json').write_text(json.dumps(details,indent=2,default=str))
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
