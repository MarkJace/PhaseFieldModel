"""仅根据已有能垒、临界尺寸和gamma表生成一个含N个核的初始态；不求相场。"""
import csv,json
from pathlib import Path
import numpy as np
from scipy.spatial import HalfspaceIntersection,ConvexHull
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def write_field_vtk(path, fields, spacing):
    shape=next(iter(fields.values())).shape
    grid=vtk.vtkRectilinearGrid();grid.SetDimensions(*(np.array(shape)+1))
    for axis,setter in enumerate([grid.SetXCoordinates,grid.SetYCoordinates,grid.SetZCoordinates]):setter(numpy_to_vtk(np.arange(shape[axis]+1)*spacing,deep=True))
    for name,values in fields.items():
        valid=np.isfinite(values)
        array=numpy_to_vtk(np.where(valid,values,0).ravel(order='F'),deep=True);array.SetName(name);grid.GetCellData().AddArray(array)
        if not valid.all():
            array=numpy_to_vtk(valid.astype(np.uint8).ravel(order='F'),deep=True);array.SetName(name+'_isfinite');grid.GetCellData().AddArray(array)
    grid.GetCellData().SetActiveScalars('P_C_m2')
    writer=vtk.vtkRectilinearGridWriter();writer.SetFileName(str(path));writer.SetInputData(grid);writer.SetFileTypeToBinary()
    if writer.Write()!=1:raise RuntimeError('VTK写入失败')


def existing_shape(x, critical_h, critical_d, table_x, table_gamma, angles):
    """只插值已有gamma重建形状，使用已保存H*/D*，不重求任何形核势垒。"""
    theta=np.deg2rad(angles[:-1]);normals=np.column_stack([np.cos(theta),np.sin(theta)])
    gamma=np.array([np.interp(x,table_x,table_gamma[:,j]) for j in range(len(angles)-1)])
    support=gamma/gamma.min()
    vertices=HalfspaceIntersection(np.column_stack([normals,-support]),np.zeros(2)).intersections
    vertices=vertices[ConvexHull(vertices).vertices]
    # 表中几何插值和gamma插值可能有微小不一致；分别匹配原文件H*和D*。
    radial_scale=critical_d/(2*np.max(vertices[:,0]))
    vertical_scale=critical_h/np.ptp(vertices[:,1])
    scaled=vertices*np.array([radial_scale,vertical_scale])
    return normals,support,radial_scale,vertical_scale,scaled


def seed_blend(shape,spacing,center,normals,support,rscale,zscale,epsilon,interface_nm):
    length=np.array(shape)*spacing
    axes=[((np.arange(n)+.5)*spacing-center[j]+length[j]/2)%length[j]-length[j]/2 for j,n in enumerate(shape)]
    radius=np.sqrt(axes[0][:,None,None]**2+axes[1][None,:,None]**2);z=axes[2][None,None,:]
    rscale*=1+epsilon;zscale*=1+epsilon
    distance=np.full(shape,np.inf)
    for normal,b in zip(normals,support):
        nx,nz=normal[0]/rscale,normal[1]/zscale
        distance=np.minimum(distance,(b-nx*radius-nz*z)/np.hypot(nx,nz))
    return .5*(1+np.tanh(distance/interface_nm)),distance>=0


def initialize_nuclei(source,out,n_seeds,seed,eps_range,interface_nm,exclusion_blend=.05,eps=None):
    if eps is not None:
        eps=np.atleast_1d(eps).astype(float)
        if len(eps) not in [1,n_seeds] or not np.isfinite(eps).all() or np.any(eps<=-1):raise ValueError('eps须为一个或N个有限值，且>-1')
    elif eps_range is None or len(eps_range)!=2 or not np.isfinite(eps_range).all() or not -1<eps_range[0]<=eps_range[1]:raise ValueError('eps_range无效')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with np.load(source) as d:
        x=d['x_Sc'];ps=d['Ps_C_m2'];barrier=d['barrier_eV'];h=d['critical_height_nm'];diam=d['critical_diameter_nm'];valid=d['valid'].astype(bool)
        table_x=d['table_x_sc'];table_gamma=d['table_gamma_J_m2'];angles=d['angles_deg'];meta=json.loads(str(d['metadata_json']))
    spacing=meta['delta_nm'];shape=x.shape;length=np.array(shape)*spacing
    eligible=valid & np.isfinite(barrier)&np.isfinite(h)&np.isfinite(diam)&(ps>0)
    if n_seeds and not eligible.any():raise RuntimeError('没有可抽样的有效CNT节点')
    logweight=np.full(shape,-np.inf)
    logweight[eligible]=-barrier[eligible]*1.602176634e-19/(1.380649e-23*meta['temperature'])
    weights=np.zeros(shape)
    if eligible.any():weights[eligible]=np.exp(logweight[eligible]-logweight[eligible].max());weights/=weights.sum()
    log_probability=np.full(shape,np.nan)
    if eligible.any():
        normalizer=logweight[eligible].max()+np.log(np.exp(logweight[eligible]-logweight[eligible].max()).sum())
        log_probability[eligible]=(logweight[eligible]-normalizer)/np.log(10)
    available=weights.copy();rng=np.random.default_rng(seed)
    parent=-np.nan_to_num(ps,nan=0.);p=parent.copy();labels=np.zeros(shape,dtype=np.int32)
    occupied=np.zeros(shape,dtype=bool);records=[]
    for sid in range(1,n_seeds+1):
        for _ in range(2000):
            total=available.sum()
            if total<=0:raise RuntimeError('无可用位置，不能生成请求的N个不重叠核')
            index=np.unravel_index(rng.choice(available.size,p=(available/total).ravel()),shape)
            probability=available[index]/total;available[index]=0
            epsilon=float(rng.uniform(*eps_range)) if eps is None else float(eps[0] if len(eps)==1 else eps[sid-1]);center=(np.array(index)+.5)*spacing
            normals,support,rs,zs,vertices=existing_shape(x[index],h[index],diam[index],table_x,table_gamma,angles)
            actual_vertices=vertices*(1+epsilon)
            if np.max(abs(actual_vertices[:,1]))>=length[2]/2 or np.max(abs(actual_vertices[:,0]))>=min(length[:2])/2:continue
            blend,mask=seed_blend(shape,spacing,center,normals,support,rs,zs,epsilon,interface_nm)
            footprint=blend>exclusion_blend
            if not mask.any() or np.any(occupied&footprint):continue
            before=p.copy();p=p*(1-blend)+np.nan_to_num(ps,nan=0.)*blend
            labels[mask]=sid;occupied|=footprint;available[occupied]=0
            records.append(dict(seed_id=sid,i=int(index[0]),j=int(index[1]),k=int(index[2]),x_nm=float(center[0]),y_nm=float(center[1]),z_nm=float(center[2]),x_Sc=float(x[index]),barrier_eV=float(barrier[index]),draw_probability=float(probability),epsilon=epsilon,critical_height_nm=float(h[index]),critical_diameter_nm=float(diam[index]),seed_height_nm=float(h[index]*(1+epsilon)),seed_diameter_nm=float(diam[index]*(1+epsilon)),core_grid_count=int(mask.sum()),core_grid_volume_nm3=float(mask.sum()*spacing**3),new_positive_volume_nm3=float(np.sum((p>0)&(before<=0))*spacing**3)))
            print(f'初始核 {sid}/{n_seeds}: {index}, eps={epsilon:+.4f}',flush=True)
            break
        else:raise RuntimeError('重试后仍无法放入N个核；可减少N、缩小eps范围或扩大区域')
    seed_meta=dict(source=str(Path(source).resolve()),source_metadata=meta,n_seeds=n_seeds,seed_nuc=seed,epsilon_range=list(eps_range) if eps is None else None,epsilon_mode='uniform_range' if eps is None else ('fixed_all' if len(eps)==1 else 'per_seed'),epsilon_values_input=eps.tolist() if eps is not None else None,actual_epsilons=[r['epsilon'] for r in records],interface_nm=interface_nm,exclusion_blend=exclusion_blend,spacing_nm=spacing,shape=list(shape),mode='one initial state with N nuclei, independent epsilon per nucleus',uses_saved_barrier_and_critical_dimensions=True)
    np.savez_compressed(out/'initial_seeded.npz',P_initial_C_m2=p,P_parent_C_m2=parent,x_Sc=x,Ps_C_m2=ps,seed_id=labels,position_weight=weights,log10_position_weight=log_probability,eligible=eligible,metadata_json=np.array(json.dumps(seed_meta)))
    write_field_vtk(out/'initial_seeded.vtk',dict(P_C_m2=p,x_Sc=x,seed_id=labels,position_weight=weights,log10_position_weight=log_probability),spacing)
    with (out/'seed_catalog.csv').open('w',newline='') as f:
        keys=list(records[0]) if records else ['seed_id'];writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(records)
    print(f'初始化完成：同一个极化场包含{n_seeds}个随机核；未执行相场演化',flush=True)
