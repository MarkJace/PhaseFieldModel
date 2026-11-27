#!/usr/bin/env python3
"""方法一：WZ原子构型 -> 重叠计数浓度场 -> 局域均匀参考形核能垒。"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter,uniform_filter
import vtk
from vtk.util.numpy_support import numpy_to_vtk
# 上级稳态/CNT核心只在需要新计算能垒时加载；读取已有数据置核时不加载。


def neighbor_offsets():
    """正交周期盒中的三角层AB堆垛：j,k奇偶性共同决定12近邻。"""
    a,c=.32,.51;dy=np.sqrt(3)*a/2
    def position(i,j,k):
        return np.array([a*(i+(j%2)/2+(k%2)/2),dy*(j+(k%2)/3),c*k/2])
    result={}
    for pj in [0,1]:
        for pk in [0,1]:
            j,k=2+pj,2+pk;origin=position(0,j,k);pairs=[]
            for di in range(-2,3):
                for dj in range(-2,3):
                    for dk in [-1,0,1]:
                        if (di,dj,dk)!=(0,0,0):pairs.append((np.linalg.norm(position(di,j+dj,k+dk)-origin),(di,dj,dk)))
            result[pj,pk]=[offset for distance,offset in sorted(pairs)[:12]]
    return result


def chemical_structure(shape,x,target,seed,tolerance):
    rng=np.random.default_rng(seed);noise=rng.normal(size=shape)
    count=round(noise.size*x)
    if not 0<count<noise.size:raise ValueError('原子体系至少需要一个Al和一个Sc')
    offsets=neighbor_offsets();history=[]
    def build(sigma):
        field=gaussian_filter(noise,sigma=sigma,mode='wrap') if sigma else noise
        species=np.zeros(field.size,dtype=np.uint8)
        species[np.argpartition(field.ravel(),field.size-count)[-count:]]=1
        species=species.reshape(shape)
        cross=0
        for (pj,pk),neighbors in offsets.items():
            center=species[:,pj::2,pk::2]
            total=np.zeros(center.shape,dtype=np.int16)
            for di,dj,dk in neighbors:total+=np.roll(species,(-di,-dj,-dk),axis=(0,1,2))[:,pj::2,pk::2]
            cross+=int(total[center==0].sum())
        eta=1-cross/(12*(species.size-count)*species.mean())
        history.append([sigma,eta]);return species,eta
    best,eta=build(0.)
    if abs(eta-target)<=tolerance:return best,eta,history
    lo,hi=0.,1.
    for _ in range(12):
        candidate,value=build(hi)
        if abs(value-target)<abs(eta-target):best,eta=candidate,value
        if value>=target:break
        hi*=2
    for _ in range(40):
        if abs(eta-target)<=tolerance:return best,eta,history
        mid=(lo+hi)/2;candidate,value=build(mid)
        if abs(value-target)<abs(eta-target):best,eta=candidate,value
        if value<target:lo=mid
        else:hi=mid
    raise RuntimeError(f'目标SRO未匹配，最优{eta}; 可适当增大sro-tol')


def count_field(species,grid_shape,delta,window):
    """原子投到正交体素，周期重叠立方窗计数。不是每格独立分配原子。"""
    nx,ny,nz=species.shape;lengths=np.array(grid_shape)*delta
    ax=lengths[0]/nx;dy=lengths[1]/ny;dz=lengths[2]/nz
    ii,jj=np.indices((nx,ny));all_count=np.zeros(grid_shape,dtype=np.int16);sc_count=np.zeros_like(all_count)
    for k in range(nz):
        xx=np.mod(ax*(ii+(jj%2)/2+(k%2)/2),lengths[0])
        yy=np.mod(dy*(jj+(k%2)/3),lengths[1])
        ix=np.floor(xx/delta).astype(int)%grid_shape[0]
        iy=np.floor(yy/delta).astype(int)%grid_shape[1]
        iz=int(np.floor(k*dz/delta))%grid_shape[2]
        ids=np.ravel_multi_index((ix,iy,np.full(ix.shape,iz)),grid_shape)
        np.add.at(all_count.ravel(),ids.ravel(),1)
        np.add.at(sc_count.ravel(),ids.ravel(),species[:,:,k].ravel())
    size=window**3
    total=np.rint(uniform_filter(all_count.astype(float),size=window,mode='wrap')*size).astype(np.int32)
    sc=np.rint(uniform_filter(sc_count.astype(float),size=window,mode='wrap')*size).astype(np.int32)
    if np.any(total==0):raise RuntimeError('统计窗中无阳离子，请增大window-cells')
    return sc,total,np.array([ax,dy,dz])


def finish_initialization(source,args):
    from nucleus_initialization import initialize_nuclei
    initialize_nuclei(source,args.out,args.n_seeds,args.seed_nuc,args.eps_range,args.interface_nm,eps=args.eps)
    if args.run_growth:
        import subprocess
        subprocess.run([sys.executable,str(Path(__file__).with_name('phasefield_evolve.py')),'--source',str(args.out/'initial_seeded.npz'),'--out',str(args.out/'evolution')],check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid',nargs=3,type=int,default=[128,128,256])
    parser.add_argument('--delta-nm',type=float,default=.28)
    parser.add_argument('--x-sc',type=float,default=.25)
    parser.add_argument('--sro',type=float,default=.5)
    parser.add_argument('--sro-tol',type=float,default=.001)
    parser.add_argument('--seed',type=int,default=20261127)
    parser.add_argument('--window-cells',type=int,default=5,help='奇数；默认5，对应1.4nm重叠立方窗')
    parser.add_argument('--composition-step',type=float,default=.002,help='畴壁/参考核几何查表成分间距；实际局域Landau不量化')
    parser.add_argument('--field-mv-cm',type=float,default=3.)
    parser.add_argument('--temperature',type=float,default=298.)
    parser.add_argument('--angle-step',type=int,default=15)
    parser.add_argument('--half-width-nm',type=float,default=200.)
    parser.add_argument('--tol',type=float,default=1e-5)
    parser.add_argument('--donors-cm3',type=float,default=1e18)
    parser.add_argument('--hole-mass',type=float,default=1.)
    parser.add_argument('--out',type=Path,default=Path('results_nucleation'))
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--from-existing',type=Path,help='读取已有initial_nucleation.npz，跳过构型和能垒计算')
    parser.add_argument('--n-seeds',type=int,default=3)
    parser.add_argument('--seed-nuc',type=int,default=12345)
    eps_group=parser.add_mutually_exclusive_group()
    eps_group.add_argument('--eps',nargs='+',type=float,help='单值用于全部核；或按核顺序给N个值，必须大于-1')
    eps_group.add_argument('--eps-range',nargs=2,type=float,help='每核在[min,max]内随机取值；默认[-0.15,0.15]')
    parser.add_argument('--interface-nm',type=float,default=.14)
    parser.add_argument('--run-growth',action='store_true',help='生成含N个核的初态后，再调用独立相场程序')
    args=parser.parse_args();grid=tuple(args.grid)
    if args.n_seeds<0 or not np.isfinite(args.interface_nm) or args.interface_nm<=0:parser.error('N>=0，interface-nm必须有限且>0')
    if args.eps is not None:
        if len(args.eps) not in [1,args.n_seeds] or not np.isfinite(args.eps).all() or min(args.eps)<=-1:parser.error('--eps须为一个值或N个值，每个值有限且>-1')
    else:
        if args.eps_range is None:args.eps_range=[-.15,.15]
        if not np.isfinite(args.eps_range).all() or args.eps_range[0]<=-1 or args.eps_range[0]>args.eps_range[1]:parser.error('--eps-range要求-1<min<=max，且均有限')
    if not args.overwrite and any((args.out/f).exists() for f in ['initial_seeded.npz','initial_seeded.vtk','seed_catalog.csv']):parser.error('置核输出已存在，请另选out或加--overwrite')
    if args.from_existing:
        finish_initialization(args.from_existing,args)
        return
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from sro_nucleation import wall_energies,wulff_polygon,reference_geometry,bulk_driving
    positive=[args.delta_nm,args.composition_step,args.field_mv_cm,args.temperature,args.half_width_nm,args.tol,args.hole_mass,args.sro_tol]
    if min(grid)<4 or not np.isfinite(positive).all() or min(positive)<=0 or not 0<args.x_sc<1 or not 0<=args.sro<1 or args.window_cells<3 or args.window_cells%2!=1 or args.window_cells>min(grid) or args.angle_step<=0 or 90%args.angle_step or not 0<=args.donors_cm3<np.inf:parser.error('尺寸/数值参数无效；窗口须为不大于最小网格数的奇数，角度步长须整除90')
    if not args.overwrite and any((args.out/f).exists() for f in ['initial_nucleation.npz','initial_nucleation.vtk']):parser.error('输出已存在，请另选out或加--overwrite')
    lengths=np.array(grid)*args.delta_nm
    # 选取最近的正交WZ周期胞数；小幅调整平均晶格间距使原子周期盒与相场盒重合。
    atomic_shape=(max(3,round(lengths[0]/.32)),2*max(2,round(lengths[1]/(np.sqrt(3)*.32))),2*max(2,round(lengths[2]/.51)))
    print(f'相场网格{grid}, 尺寸{lengths} nm; WZ阳离子数组{atomic_shape}',flush=True)
    species,achieved,history=chemical_structure(atomic_shape,args.x_sc,args.sro,args.seed,args.sro_tol)
    print(f'原子Sc比例{species.mean():.8f}, SRO={achieved:.6f}',flush=True)
    nsc,ntotal,atomic_spacing=count_field(species,grid,args.delta_nm,args.window_cells)
    concentration=nsc/ntotal
    fractions,inverse=np.unique(concentration,return_inverse=True)
    # 每种真实计数比仅做一次电焓根求解；不会四舍五入局域Landau系数。
    bulk=[bulk_driving(x,args.field_mv_cm*1e8) for x in fractions]
    good=np.array([b is not None and b['status']=='ok' for b in bulk])
    table_x=np.array([]);table_gamma=np.empty((0,0));diagnostics={};table_geometry=np.empty((0,4))
    angles=np.arange(0,361,args.angle_step,dtype=float)
    if good.any():
        low=max(0,np.floor(fractions[good].min()/args.composition_step)*args.composition_step)
        high=min(.61149,np.ceil(fractions[good].max()/args.composition_step)*args.composition_step)
        table_x=np.unique(np.r_[np.arange(low,high,args.composition_step),high])
        base_angles=np.arange(-90,91,args.angle_step,dtype=float)
        print(f'局域计数比{len(fractions)}种；畴壁几何查表{len(table_x)}种成分，步长{args.composition_step}',flush=True)
        base,diagnostics=wall_energies(table_x,base_angles,args)
        if not np.isfinite(base).all():raise RuntimeError('查表畴壁求解失败；请调整求解参数后重跑，不以插值掩盖失败')
        equivalent=np.round(np.rad2deg(np.arcsin(np.sin(np.deg2rad(angles))))/args.angle_step)*args.angle_step
        indices=[np.flatnonzero(base_angles==a)[0] for a in equivalent];table_gamma=base[:,indices]
        geometry=[]
        for gamma in table_gamma:
            vertices,_=wulff_polygon(angles,gamma);v,w,area,h,d=reference_geometry(vertices,angles,gamma)
            geometry.append([v,w,h,d])
        table_geometry=np.array(geometry)
    names=['Ps_C_m2','intrinsic_field_MV_cm','driving_J_m3','lambda_star','barrier_eV','critical_height_nm','critical_diameter_nm','log10_R_over_R0']
    unique_fields={name:np.full(len(fractions),np.nan) for name in names};status=np.full(len(fractions),2,dtype=np.uint8)
    for i,b in enumerate(bulk):
        if b is None:continue
        unique_fields['Ps_C_m2'][i]=b['Ps'];unique_fields['intrinsic_field_MV_cm'][i]=b['intrinsic']/1e8
        if b['status']!='ok':status[i]=1;continue
        v,w,h,d=[np.interp(fractions[i],table_x,table_geometry[:,j]) for j in range(4)]
        dg=b['driving'];star=2*w/(3*v*dg);barrier=4*w**3/(27*(v*dg)**2)
        for name,value in dict(driving_J_m3=dg,lambda_star=star,barrier_eV=barrier/1.602176634e-19,critical_height_nm=star*h*1e9,critical_diameter_nm=star*d*1e9,log10_R_over_R0=-barrier/(1.380649e-23*args.temperature*np.log(10))).items():unique_fields[name][i]=value
        status[i]=0
    mapping=inverse.reshape(grid)
    fields={key:value[mapping] for key,value in unique_fields.items()}
    fields.update(x_Sc=concentration,N_Sc_window=nsc,N_cations_window=ntotal,status_id=status[mapping],valid=(status[mapping]==0).astype(np.uint8),landau_a2_SI=5.50378e9*(concentration-.6115)/2)
    for name,value in [('landau_a4_SI',.149031e9/4),('landau_a6_SI',.928609e9/6)]:fields[name]=np.full(grid,value)
    args.out.mkdir(parents=True,exist_ok=True)
    metadata=vars(args).copy();metadata.update(out=str(args.out),atomic_shape=list(atomic_shape),actual_atomic_x_sc=float(species.mean()),actual_sro=float(achieved),box_lengths_nm=lengths.tolist(),atomic_spacing_nm=atomic_spacing.tolist(),window_side_nm=args.window_cells*args.delta_nm,mean_grid_x_sc=float(concentration.mean()),count_weighted_x_sc=float(nsc.sum()/ntotal.sum()),status_codes={'0':'valid local uniform CNT','1':'metastable_absent','2':'nonferroelectric'},table_approximation='linear interpolation of V0,W0,H0,D0; exact local composition for Landau and bulk driving',not_time_evolution=True)
    np.savez_compressed(args.out/'initial_nucleation.npz',**fields,species_sc=species,unique_local_x_sc=fractions,table_x_sc=table_x,table_gamma_J_m2=table_gamma,table_geometry_V_W_H_D=table_geometry,angles_deg=angles,sro_history=history,metadata_json=np.array(json.dumps(metadata)),**diagnostics)
    image=vtk.vtkRectilinearGrid();image.SetDimensions(*(np.array(grid)+1))
    for axis,setter in enumerate([image.SetXCoordinates,image.SetYCoordinates,image.SetZCoordinates]):setter(numpy_to_vtk(np.arange(grid[axis]+1)*args.delta_nm,deep=True))
    for key,value in fields.items():
        finite=np.isfinite(value);array=numpy_to_vtk(np.where(finite,value,0).ravel(order='F'),deep=True);array.SetName(key);image.GetCellData().AddArray(array)
        if not finite.all():
            mask=numpy_to_vtk(finite.astype(np.uint8).ravel(order='F'),deep=True);mask.SetName(key+'_isfinite');image.GetCellData().AddArray(mask)
    image.GetCellData().SetActiveScalars('barrier_eV')
    writer=vtk.vtkRectilinearGridWriter();writer.SetFileName(str(args.out/'initial_nucleation.vtk'));writer.SetInputData(image);writer.SetFileTypeToBinary()
    if writer.Write()!=1:raise RuntimeError('VTK写出失败')
    print(f'完成：有效CNT网格{fields["valid"].sum()}/{np.prod(grid)}；结果{args.out}',flush=True)
    finish_initialization(args.out/'initial_nucleation.npz',args)


if __name__=='__main__':main()
