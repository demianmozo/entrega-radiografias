"""Reproducible ResNet-18 baseline with fixed test; audit.py must run first."""
import os,time,json,random,hashlib,platform,subprocess,sys,argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/baseline')

def seed_worker(worker):
    seed=torch.initial_seed()%2**32;np.random.seed(seed);random.seed(seed)

class Radiographs(Dataset):
    def __init__(self,frame,cfg,train=False):
        self.rows=frame.reset_index(drop=True);self.labels=[cfg['classes'].index(x) for x in self.rows.label]
        ops=[transforms.Grayscale(3),transforms.Resize((224,224),antialias=True)]
        if train:ops+=[transforms.RandomAffine(7,translate=(.03,.03),scale=(.95,1.05),interpolation=transforms.InterpolationMode.BILINEAR,fill=0)]
        ops += [transforms.ToTensor(),transforms.Normalize(cfg['normalization_mean'],cfg['normalization_std'])]
        self.transform=transforms.Compose(ops)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows.iloc[i].path) as im:x=self.transform(im)
        return x,self.labels[i]

def metrics(y,p,classes):
    return dict(accuracy=accuracy_score(y,p),balanced_accuracy=balanced_accuracy_score(y,p),macro_f1=f1_score(y,p,average='macro',labels=list(range(3)),zero_division=0),per_class=classification_report(y,p,labels=list(range(3)),target_names=classes,output_dict=True,zero_division=0),confusion_matrix=confusion_matrix(y,p,labels=list(range(3))).tolist())

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/baseline.json');parser.add_argument('--out',default='results/baseline');parser.add_argument('--epochs-per-invocation',type=int,default=2);args=parser.parse_args()
    start=time.time();OUT=Path(args.out);cfg=json.loads(Path(args.config).read_text());OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'test_metrics.json').exists():raise RuntimeError('Completed experiment exists; use a new output directory')
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(cfg['cpu_threads']);torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    df=pd.read_csv('data/processed/manifest.csv',keep_default_na=False)
    sets={k:df[df.split==k].copy() for k in ['train','val','test']}
    for a in sets:
        for b in sets:
            if a!=b:assert not set(sets[a].duplicate_group)&set(sets[b].duplicate_group)
    generator=torch.Generator().manual_seed(seed)
    loaders={k:DataLoader(Radiographs(v,cfg,k=='train'),batch_size=cfg['batch_size'],shuffle=k=='train',num_workers=cfg['workers'],worker_init_fn=seed_worker,generator=generator,persistent_workers=cfg["workers"]>0) for k,v in sets.items()}
    counts=np.bincount(loaders['train'].dataset.labels,minlength=3);weights=len(sets['train'])/(3*counts)
    cfg.update(train_counts=counts.tolist(),class_weights=weights.tolist(),manifest_sha256=hashlib.sha256(Path('data/processed/manifest.csv').read_bytes()).hexdigest(),split_sha256=hashlib.sha256(Path('data/processed/splits.csv').read_bytes()).hexdigest())
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2))
    env=dict(python=sys.version,torch=torch.__version__,torchvision=__import__('torchvision').__version__,numpy=np.__version__,pandas=pd.__version__,sklearn=__import__('sklearn').__version__,platform=platform.platform(),cpu=platform.processor(),cuda_available=torch.cuda.is_available(),threads=torch.get_num_threads())
    (OUT/'environment.json').write_text(json.dumps(env,indent=2))
    (OUT/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    model=models.resnet18(weights=None);model.fc=nn.Linear(model.fc.in_features,3);model=model.to(memory_format=torch.channels_last)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
    resume=None
    if (OUT/'resume.pt').exists():
        resume=torch.load(OUT/'resume.pt',map_location='cpu',weights_only=False)
        assert resume['config']==cfg,'Configuration changed during resumed experiment'
        model.load_state_dict(resume['model'])
        print('RESUME after epoch',len(resume['history']),flush=True)
    else:
        # Independent smoke; restore weights/RNG and recreate data loaders afterwards.
        state={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state()
        x,y=next(iter(loaders['train']));x=x.to(memory_format=torch.channels_last);tick=time.time()
        with torch.autocast('cpu',dtype=torch.bfloat16):z=model(x);loss=criterion(z,y)
        loss.backward();grad=float(model.conv1.weight.grad.norm())
        assert z.shape==(len(y),3) and torch.isfinite(loss) and np.isfinite(grad) and grad>0
        smoke=dict(shape=list(x.shape),label_counts=torch.bincount(y,minlength=3).tolist(),logits_shape=list(z.shape),loss=loss.item(),first_layer_gradient_norm=grad,seconds=time.time()-tick)
        (OUT/'smoke.json').write_text(json.dumps(smoke,indent=2));print('SMOKE',smoke,flush=True)
        model.load_state_dict(state);model.zero_grad(set_to_none=True);torch.set_rng_state(rng);del state,x,y,z,loss
    generator=torch.Generator().manual_seed(seed)
    del loaders
    loaders={k:DataLoader(Radiographs(v,cfg,k=='train'),batch_size=cfg['batch_size'],shuffle=k=='train',num_workers=cfg['workers'],worker_init_fn=seed_worker,generator=generator,persistent_workers=cfg['workers']>0) for k,v in sets.items()}
    if resume:
        history=resume['history'];best=resume['best'];best_epoch=resume['best_epoch']
        random.setstate(resume['python_rng']);np.random.set_state(resume['numpy_rng']);torch.set_rng_state(resume['torch_rng']);generator.set_state(resume['loader_rng'])
    else:
        history=[];best=-1;best_epoch=0;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    invocation_epochs=0
    phases=[('head',cfg['head_epochs'],cfg['head_lr']),('finetune',cfg['finetune_epochs'],cfg['finetune_lr'])]
    for phase,epochs,lr in phases:
        if resume and phase=='head' and resume['phase']=='finetune':continue
        local_start=resume['phase_epoch'] if resume and resume['phase']==phase else 0
        if local_start>=epochs or (resume and resume['phase']==phase=='finetune' and resume['bad']>=cfg['patience']):continue
        for name,param in model.named_parameters():param.requires_grad=(phase=='finetune' or name.startswith('fc.'))
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2)
        bad=0
        if resume and resume['phase']==phase:
            optimizer.load_state_dict(resume['optimizer']);scheduler.load_state_dict(resume['scheduler']);bad=resume['bad']
        for local in range(local_start,epochs):
            tick=time.time();record=dict(epoch=len(history)+1,phase=phase,lr=optimizer.param_groups[0]['lr'])
            for split in ['train','val']:
                training=split=='train';model.train(training)
                if training and phase=='head':model.eval();model.fc.train()
                ys=[];ps=[];total=0.;denominator=0.
                for x,y in loaders[split]:
                    x=x.to(memory_format=torch.channels_last)
                    with torch.set_grad_enabled(training):
                        with torch.autocast('cpu',dtype=torch.bfloat16):z=model(x);loss=criterion(z,y)
                        if training:
                            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                    assert torch.isfinite(loss)
                    norm=float(criterion.weight[y].sum());total+=loss.item()*norm;denominator+=norm
                    ys.extend(y.tolist());ps.extend(z.argmax(1).tolist())
                record[f'{split}_loss']=total/denominator;record[f'{split}_macro_f1']=f1_score(ys,ps,average='macro',zero_division=0);record[f'{split}_accuracy']=accuracy_score(ys,ps)
            score=record['val_macro_f1'];scheduler.step(score)
            if score>best+cfg['min_delta']:
                best=score;best_epoch=record['epoch'];bad=0
                torch.save(dict(model=model.state_dict(),config=cfg,epoch=best_epoch,val_macro_f1=best),OUT/'best.pt')
            else:bad+=1
            record['seconds']=time.time()-tick;history.append(record);pd.DataFrame(history).to_csv(OUT/'history.csv',index=False);print(json.dumps(record),flush=True)
            checkpoint=dict(model=model.state_dict(),config=cfg,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),phase=phase,phase_epoch=local+1,history=history,best=best,best_epoch=best_epoch,bad=bad,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),loader_rng=generator.get_state())
            torch.save(checkpoint,OUT/'resume.tmp.pt');os.replace(OUT/'resume.tmp.pt',OUT/'resume.pt')
            invocation_epochs+=1
            if phase=='finetune' and bad>=cfg['patience']:print('EARLY STOP',flush=True);break
            if args.epochs_per_invocation>0 and invocation_epochs>=args.epochs_per_invocation and not (phase=='finetune' and local+1>=epochs):
                print('CHUNK_COMPLETE: rerun same command to continue; test remains closed',flush=True);return
    # All optimization and model selection are complete, even across chunked invocations.
    # All model decisions are now frozen. Test is evaluated exactly once.
    checkpoint=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model']);model.eval()
    (OUT/'selection.json').write_text(json.dumps(dict(best_epoch=best_epoch,val_macro_f1=best,test_opened_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')),indent=2))
    ys=[];probs=[]
    with torch.no_grad():
        for x,y in loaders['test']:
            with torch.autocast('cpu',dtype=torch.bfloat16):z=model(x.to(memory_format=torch.channels_last))
            probs.extend(z.float().softmax(1).numpy());ys.extend(y.tolist())
    probs=np.asarray(probs);pred=probs.argmax(1);result=metrics(ys,pred,cfg['classes']);result.update(best_epoch=best_epoch,best_val_macro_f1=best,training_evaluation_seconds=sum(r['seconds'] for r in history)+(time.time()-start if invocation_epochs==0 else max(0,time.time()-start-sum(r['seconds'] for r in history[-invocation_epochs:]))),final_invocation_seconds=time.time()-start,initialization=cfg['initialization'])
    majority=int(counts.argmax());reference=metrics(ys,np.full(len(ys),majority),cfg['classes'])
    predictions=sets['test'][['path','label','duplicate_group','source']].reset_index(drop=True);predictions['true_index']=ys;predictions['predicted_index']=pred;predictions['predicted_label']=[cfg['classes'][x] for x in pred]
    for i,label in enumerate(cfg['classes']):predictions[f'p_{label}']=probs[:,i]
    predictions.to_csv(OUT/'test_predictions.csv',index=False)
    (OUT/'test_metrics.json').write_text(json.dumps(result,indent=2));(OUT/'majority_metrics.json').write_text(json.dumps(reference,indent=2))
    h=pd.DataFrame(history);fig,axs=plt.subplots(1,2,figsize=(11,4))
    for split in ['train','val']:axs[0].plot(h.epoch,h[f'{split}_loss'],label=split);axs[1].plot(h.epoch,h[f'{split}_macro_f1'],label=split)
    for ax in axs:ax.legend();ax.set_xlabel('Epoch');ax.axvline(best_epoch,color='gray',ls='--')
    axs[0].set_ylabel('Weighted CE');axs[1].set_ylabel('Macro-F1');fig.tight_layout();fig.savefig(OUT/'learning_curves.png');plt.close(fig)
    cm=np.asarray(result['confusion_matrix'])
    for normalized in [False,True]:
        v=cm/cm.sum(1,keepdims=True) if normalized else cm
        fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(v,cmap='Blues');fig.colorbar(im,ax=ax)
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{v[i,j]:.3f}' if normalized else str(v[i,j]),ha='center',color='white' if v[i,j]>v.max()/2 else 'black')
        ax.set_xticks(range(3),cfg['classes'],rotation=15);ax.set_yticks(range(3),cfg['classes']);ax.set_xlabel('Predicted');ax.set_ylabel('True');fig.tight_layout();fig.savefig(OUT/('confusion_normalized.png' if normalized else 'confusion_absolute.png'));plt.close(fig)
    errors=predictions[predictions.true_index!=predictions.predicted_index].copy();errors['confidence']=probs.max(1)[errors.index];errors=errors.sort_values('confidence',ascending=False);errors.to_csv(OUT/'errors.csv',index=False)
    sample=pd.concat([errors[errors.label==c].head(4) for c in cfg['classes']])
    if len(sample):
        fig,axs=plt.subplots(3,4,figsize=(12,9))
        for ax in axs.ravel():ax.axis('off')
        for ax,(_,r) in zip(axs.ravel(),sample.iterrows()):ax.imshow(Image.open(r.path),cmap='gray');ax.set_title(f'{Path(r.path).name}\n{r.label} → {r.predicted_label}\np={r.confidence:.2f}',fontsize=8)
        fig.tight_layout();fig.savefig(OUT/'error_examples.png');plt.close(fig)
    print('RESULT',json.dumps(result),flush=True)
if __name__=='__main__':main()
