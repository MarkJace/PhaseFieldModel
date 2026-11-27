#!/usr/bin/env python3
"""只读取已置核极化初态进行演化；不抽样、不重算形核能垒。"""
import argparse,csv,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/sro_wall_matplotlib')
import numpy as np
from scipy.fft import rfftn,irfftn,rfftfreq,fftfreq
from scipy.sparse.linalg import LinearOperator,cg
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import roots_legendre
from scipy.constants import elementary_charge as q,Boltzmann as kb,h,m_e
from nucleus_initialization import write_field_vtk

def make_carriers(temperature, donors_cm3, hole_mass):
    """归一化 F_1/2 数值表；其原函数保证载流子势与电荷相容。"""
    kt = kb * temperature
    nc = 2 * (2 * np.pi * 0.27 * m_e * kt / h**2)**1.5
    nv = 2 * (2 * np.pi * hole_mass * m_e * kt / h**2)**1.5
    gap = 3.3 * q
    donors = donors_cm3 * 1e6  # cm^-3 -> m^-3，完全离化净施主
    eta = np.linspace(-60, 1000, 42401)
    nodes, weights = roots_legendre(160)
    # t=u² 消去积分起点的平方根；截断处超过费米能至少 50 kT。
    upper = np.sqrt(np.maximum(eta, 0) + 50)
    u = upper[:, None] * (nodes[None, :] + 1) / 2
    occupation = 1 / (1 + np.exp(np.clip(u*u - eta[:, None], -700, 700)))
    values = 4 / np.sqrt(np.pi) * upper / 2 * ((u*u * occupation) @ weights)
    spline = CubicSpline(eta, values)
    primitive = spline.antiderivative()

    def fd(z):
        z = np.asarray(z)
        bounded = np.clip(z, -60, 1000)
        result = spline(bounded)
        # 简并极限展开；通常物理解不需要走到 eta>1000。
        high = 4 / (3*np.sqrt(np.pi)) * np.maximum(z, 1)**1.5 * (1 + np.pi**2/(8*np.maximum(z, 1)**2))
        return np.where(z < -60, np.exp(np.clip(z, -745, -60)), np.where(z > 1000, high, result))

    def integral_fd(z):
        z = np.asarray(z)
        bounded = np.clip(z, -60, 1000)
        result = primitive(bounded)
        low = primitive(-60) + np.exp(np.clip(z, -745, -60)) - np.exp(-60)
        zz = np.maximum(z, 1)
        asym = lambda t: 8/(15*np.sqrt(np.pi))*t**2.5 + np.pi**2/(3*np.sqrt(np.pi))*np.sqrt(t)
        high = primitive(1000) + asym(zz) - asym(1000)
        return np.where(z < -60, low, np.where(z > 1000, high, result))

    ef = brentq(lambda e: nv*fd(-e*q/kt) - nc*fd((e*q-gap)/kt) + donors, -5, 8) * q
    en0, ep0 = (ef-gap)/kt, -ef/kt

    def carriers(phi):
        return nc*fd(en0+q*phi/kt), nv*fd(ep0-q*phi/kt)

    def charge(phi):
        n, p = carriers(phi)
        return q * (p-n+donors)

    def carrier_potential(phi):
        # U'(phi)=rho(phi)；U(0)=0，包含固定施主及电子/空穴巨势。
        return q*donors*phi - kt*(nc*(integral_fd(en0+q*phi/kt)-integral_fd(en0)) + nv*(integral_fd(ep0-q*phi/kt)-integral_fd(ep0)))

    return carriers, charge, carrier_potential, ef/q


def spectral_operators(shape,spacing):
    kx=2*np.pi*fftfreq(shape[0],spacing)[:,None,None]
    ky=2*np.pi*fftfreq(shape[1],spacing)[None,:,None]
    kz=2*np.pi*rfftfreq(shape[2],spacing)[None,None,:]
    k2=kx*kx+ky*ky+kz*kz
    return k2,kz


def solve_potential(p,phi,k2,kz,charge,potential,args):
    """-eps lap(phi)=rho_b+rho(phi)，周期FD平衡；常数模也求解以满足总电中性。"""
    shape=p.shape;eps=8.8541878128e-12*5
    rhob=-irfftn(1j*kz*rfftn(p),s=shape).real
    if args.electrostatics=='linear':
        mass=max(float(-(charge(1e-4)-charge(-1e-4))/2e-4),1e-12)
        phi=irfftn(rfftn(rhob)/(eps*k2+mass),s=shape).real
        residual=eps*irfftn(k2*rfftn(phi),s=shape).real+mass*phi-rhob
        # 线性模式仅用于调试/敏感性，不等同于CNT的非线性屏蔽。
        energy=.5*np.mean(rhob*phi)
        return phi,irfftn(1j*kz*rfftn(phi),s=shape).real,energy,float(np.linalg.norm(residual.ravel())/np.sqrt(p.size))
    def residual(v):return eps*irfftn(k2*rfftn(v),s=shape).real-charge(v)-rhob
    threshold=args.poisson_atol+args.poisson_tol*np.linalg.norm(rhob.ravel())/np.sqrt(p.size)
    for iteration in range(args.poisson_maxiter):
        residual_value=residual(phi);norm=np.linalg.norm(residual_value.ravel())/np.sqrt(p.size)
        if norm<=threshold:break
        mass=np.maximum(-(charge(phi+1e-4)-charge(phi-1e-4))/2e-4,1e-10)
        def matvec(v):
            v=v.reshape(shape)
            return (eps*irfftn(k2*rfftn(v),s=shape).real+mass*v).ravel()
        diagonal=eps*k2+float(np.mean(mass))
        def precondition(v):return irfftn(rfftn(v.reshape(shape))/diagonal,s=shape).real.ravel()
        operator=LinearOperator((p.size,p.size),matvec=matvec,dtype=float)
        pre=LinearOperator(operator.shape,matvec=precondition,dtype=float)
        step,info=cg(operator,-residual_value.ravel(),M=pre,rtol=1e-4,atol=0,maxiter=args.cg_maxiter)
        if info!=0:raise RuntimeError(f'Poisson CG未收敛: {info}；不输出伪收敛结果')
        step=step.reshape(shape);factor=min(1.,.5/max(float(np.max(abs(step))),1e-30))
        for _ in range(20):
            candidate=phi+factor*step
            if np.linalg.norm(residual(candidate).ravel())/np.sqrt(p.size)<norm:phi=candidate;break
            factor*=.5
        else:raise RuntimeError('Poisson Newton线搜索失败')
    else:raise RuntimeError('Poisson达到最大迭代次数；尝试加大poisson-maxiter')
    dphi=irfftn(1j*kz*rfftn(phi),s=shape).real
    grad_energy=.5*eps*np.mean(phi*irfftn(k2*rfftn(phi),s=shape).real)
    energy=-grad_energy+np.mean(p*dphi+potential(phi))
    return phi,dphi,float(energy),float(norm)


def total_energy(p,a2,k2,electro_energy,args):
    lap_positive=irfftn(k2*rfftn(p),s=p.shape).real
    density=a2*p*p+(.149031e9/4)*p**4+(.928609e9/6)*p**6-args.field*1e8*p
    return float((np.mean(density)+.5*5.71e-12*np.mean(p*lap_positive)+electro_energy)*p.size*(args.spacing_nm*1e-9)**3)


def advance(p,phi,a2,k2,kz,charge,potential,args):
    phi,dphi,electro_energy,residual=solve_potential(p,phi,k2,kz,charge,potential,args)
    a4=.149031e9/4;a6=.928609e9/6;scale=1e9;g=5.71e-12
    force=(2*a2*p+4*a4*p**3+6*a6*p**5-args.field*1e8+dphi)/scale
    stabilization=max(float(np.max((2*a2+12*a4*p*p+30*a6*p**4)/scale)),1.)
    new=irfftn(rfftn(p-args.dtau*force+args.dtau*stabilization*p)/(1+args.dtau*(g/scale*k2+stabilization)),s=p.shape).real
    if not np.isfinite(new).all() or np.max(abs(new))>3:raise RuntimeError('极化更新失稳；请减小dtau')
    return new,phi,residual,total_energy(p,a2,k2,electro_energy,args)



def run_evolution(initial,x,labels,spacing,source_metadata,args,folder):
    folder.mkdir(parents=True,exist_ok=True)
    # 只从局域Sc场构造Landau参数，不从任何形核表或上级脚本获取。
    a2=5.50378e9*(x-.6115)/2
    args.field=source_metadata['field_mv_cm'];args.spacing_nm=spacing
    _,charge,potential,_=make_carriers(source_metadata['temperature'],source_metadata['donors_cm3'],source_metadata['hole_mass'])
    p=initial.copy();phi=np.zeros(p.shape);k2,kz=spectral_operators(p.shape,spacing*1e-9)
    history=[];saved=[];cell_volume=spacing**3
    for step in range(args.steps+1):
        entry=dict(step=step,tau=step*args.dtau,time_s=step*args.dtau/(args.mobility*1e9) if args.mobility else np.nan,P_mean_C_m2=float(p.mean()),positive_volume_nm3=float(np.sum((p>0)&(a2<0))*cell_volume),positive_total_volume_nm3=float(np.sum(p>0)*cell_volume))
        if step%args.save_every==0 or step==args.steps:
            path=folder/f'P_{step:05d}.vtk'
            write_field_vtk(path,dict(P_C_m2=p,x_Sc=x,initial_seed_id=labels),spacing);saved.append((step*args.dtau,path.name))
        if step<args.steps:
            p,phi,residual,energy=advance(p,phi,a2,k2,kz,charge,potential,args)
        else:
            phi,_,electro_energy,residual=solve_potential(p,phi,k2,kz,charge,potential,args)
            energy=total_energy(p,a2,k2,electro_energy,args)
        entry.update(free_energy_J=energy,poisson_residual_C_m3=residual);history.append(entry)
        if step%10==0 or step==args.steps:print(f'{folder.name}: {step}/{args.steps}, V+={entry["positive_volume_nm3"]:.6g} nm3',flush=True)
    with (folder/'history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=history[0]);writer.writeheader();writer.writerows(history)
    np.savez_compressed(folder/'evolution.npz',P_initial_C_m2=initial,P_final_C_m2=p,phi_final_V=phi,x_Sc=x,initial_seed_id=labels,landau_a2_SI=a2,landau_a4_SI=.149031e9/4,landau_a6_SI=.928609e9/6,history_columns=np.array(list(history[0])),history=np.array([list(row.values()) for row in history]))
    text='<VTKFile type="Collection" version="0.1" byte_order="LittleEndian"><Collection>\n'
    for t,name in saved:text+=f'<DataSet timestep="{t}" group="" part="0" file="{name}"/>\n'
    (folder/'evolution.pvd').write_text(text+'</Collection></VTKFile>\n')
    return history


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('seeded_N3/initial_seeded.npz'))
    parser.add_argument('--out',type=Path,default=Path('results_phasefield'))
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--dtau',type=float,default=.01)
    parser.add_argument('--save-every',type=int,default=10)
    parser.add_argument('--mobility',type=float)
    parser.add_argument('--with-control',action='store_true',help='可选：额外演化同一来源的无核母相，不重新置核')
    parser.add_argument('--electrostatics',choices=['fd','linear'],default='fd')
    parser.add_argument('--poisson-tol',type=float,default=1e-4)
    parser.add_argument('--poisson-atol',type=float,default=.1)
    parser.add_argument('--poisson-maxiter',type=int,default=60)
    parser.add_argument('--cg-maxiter',type=int,default=300)
    args=parser.parse_args()
    if args.steps<1 or args.save_every<1 or not np.isfinite([args.dtau,args.poisson_tol,args.poisson_atol]).all() or min(args.dtau,args.poisson_tol,args.poisson_atol)<=0:parser.error('步数/时间步/容差无效')
    if args.mobility is not None and (not np.isfinite(args.mobility) or args.mobility<=0):parser.error('mobility须有限且为正')
    if args.out.exists() and any(args.out.iterdir()):parser.error('输出目录非空，请选择新目录')
    with np.load(args.source) as data:
        if 'P_initial_C_m2' not in data:raise ValueError('请先用method1_initial_nucleation.py --from-existing生成initial_seeded.npz；相场端不读取能垒原文件置核')
        initial=data['P_initial_C_m2'];parent=data['P_parent_C_m2'];x=data['x_Sc'];labels=data['seed_id'];meta=json.loads(str(data['metadata_json']))
    if initial.shape!=x.shape or labels.shape!=x.shape:raise ValueError('初态/浓度/标签尺寸不匹配')
    args.out.mkdir(parents=True,exist_ok=True)
    config=vars(args).copy();config.update(source=str(args.source.resolve()),out=str(args.out),initialization=meta,state='running',workflow='evolve supplied initial P only; no nucleation calculations or sampling')
    (args.out/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    seeded=run_evolution(initial,x,labels,meta['spacing_nm'],meta['source_metadata'],args,args.out/'seeded')
    control=None
    if args.with_control:control=run_evolution(parent,x,np.zeros_like(labels),meta['spacing_nm'],meta['source_metadata'],args,args.out/'control')
    config['state']='complete';(args.out/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot([r['tau'] for r in seeded],[r['positive_volume_nm3'] for r in seeded],label='Supplied N-seed initial state')
    if control:axes[0].plot([r['tau'] for r in control],[r['positive_volume_nm3'] for r in control],label='Unseeded control')
    axes[0].set(xlabel='tau = L * 1e9 * t',ylabel='Positive ferroelectric volume (nm3)');axes[0].legend()
    axes[1].plot([r['tau'] for r in seeded],[r['free_energy_J'] for r in seeded]);axes[1].set(xlabel='tau',ylabel='Free energy (J)')
    for ax in axes:ax.grid(alpha=.25)
    fig.savefig(args.out/'evolution.png',dpi=180);plt.close(fig)
    print('演化完成：初态未重新抽样，所有核在同一个场内共同演化',flush=True)


if __name__=='__main__':main()
