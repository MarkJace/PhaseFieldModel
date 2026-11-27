#!/usr/bin/env python3
"""WZ SRO -> 局域成分 -> 畴壁/Wulff/CNT -> 正交微区相对速率。仅输出NPZ和VTK。"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.integrate import solve_bvp, simpson, quad
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import roots_legendre, logsumexp
from scipy.constants import epsilon_0, elementary_charge as q, Boltzmann as kb, h, m_e
from scipy.spatial import HalfspaceIntersection, ConvexHull
import vtk
from vtk.util.numpy_support import numpy_to_vtk

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


def neighbors(s):
    """返回同层6个和层间6个邻居的Sc数量，周期边界、偶数层。"""
    inside = np.zeros(s.shape, dtype=np.int16)
    between = np.zeros_like(inside)
    for di, dj in [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]:
        inside += np.roll(s, (-di,-dj), axis=(0,1))
    even = (np.arange(s.shape[2]) % 2 == 0)[None,None,:]
    for dk in [-1,1]:
        base = np.roll(s, -dk, axis=2)
        between += base
        for axis in [0,1]:
            between += np.where(even, np.roll(base, 1, axis=axis), np.roll(base, -1, axis=axis))
    return inside, between


def wulff_polygon(angles_deg, gamma):
    """二维Wulff半平面交：x*cos(theta)+z*sin(theta)<=gamma(theta)。"""
    from scipy.spatial import HalfspaceIntersection, ConvexHull
    if not np.isfinite(gamma).all() or np.any(gamma <= 0):
        raise ValueError('Wulff构造需要有限且严格为正的畴壁能')
    # 去掉重复的360度；不插值，也不将gamma极图直接当作形状。
    angles = np.mod(angles_deg, 360)
    angles, indices = np.unique(np.round(angles, 10), return_index=True)
    theta = np.deg2rad(angles)
    normals = np.column_stack([np.cos(theta), np.sin(theta)])
    scale = float(np.min(gamma))
    support = gamma[indices]/scale
    halfspaces = np.column_stack([normals, -support])
    vertices = HalfspaceIntersection(halfspaces, np.zeros(2)).intersections
    vertices = vertices[ConvexHull(vertices).vertices]
    if np.max(vertices @ normals.T-support) > 1e-7:
        raise RuntimeError('Wulff顶点不满足输入半平面约束')
    # 统一以z向半高度归一化，保留Wulff原点，不平移非对称构型。
    half_height = np.ptp(vertices[:,1])/2
    return vertices/half_height, scale*half_height


def reference_geometry(vertices, angles, gamma, length_nm=1.):
    """旋转右半截面，逐圆台积分面积/体积。顶点必须逆时针。"""
    poly=np.asarray(vertices)
    if np.sum(poly[:,0]*np.roll(poly[:,1],-1)-poly[:,1]*np.roll(poly[:,0],-1))<0: poly=poly[::-1]
    volume=0.; surface=0.; area=0.
    scale=length_nm*1e-9
    for start,end in zip(poly,np.roll(poly,-1,axis=0)):
        # 裁剪每条边到x>=0，旋转轴上的边不计为外表面。
        if max(start[0],end[0])<=1e-12: continue
        p1,p2=start.copy(),end.copy()
        if p1[0]<0: p1=start+(end-start)*(-start[0])/(end[0]-start[0])
        if p2[0]<0: p2=start+(end-start)*(-start[0])/(end[0]-start[0])
        delta=p2-p1
        if np.linalg.norm(delta)<1e-12: continue
        normal=np.array([delta[1],-delta[0]])/np.linalg.norm(delta)
        theta=np.rad2deg(np.arctan2(normal[1],normal[0]))%360
        distance=abs((angles-theta+180)%360-180)
        index=int(np.argmin(distance))
        if distance[index]>1e-5: raise ValueError('顶点面法向与gamma角度网格不匹配')
        r1,r2=max(p1[0],0)*scale,max(p2[0],0)*scale
        dz=abs(delta[1])*scale
        face_area=np.pi*(r1+r2)*np.linalg.norm(delta)*scale
        volume+=np.pi*dz*(r1*r1+r1*r2+r2*r2)/3
        surface+=face_area*gamma[index]
        area+=face_area
    if volume<=0 or surface<=0: raise ValueError('核的参考体积/表面能无效')
    return volume,surface,area,np.ptp(poly[:,1])*scale,2*np.max(poly[:,0])*scale


def bulk_driving(x, field):
    """初态P<0，反转核P>0，E>0；计算电焓差并检查亚稳态。"""
    a2=5.50378e9*(x-.6115)/2
    a4=.149031e9/4; a6=.928609e9/6
    if a2>=0: return None
    ps=np.sqrt((-4*a4+np.sqrt((4*a4)**2-48*a6*a2))/(12*a6))
    # 曲率=0给出初始负极化分支失稳的内禀场。
    pspin=-np.sqrt((-12*a4+np.sqrt((12*a4)**2-240*a6*a2))/(60*a6))
    intrinsic=2*a2*pspin+4*a4*pspin**3+6*a6*pspin**5
    if field>=intrinsic: return dict(status='metastable_absent',Ps=ps,intrinsic=intrinsic)
    roots=np.roots([6*a6,0,4*a4,0,2*a2,-field])
    real=roots.real[abs(roots.imag)<1e-7]
    minima=real[2*a2+12*a4*real**2+30*a6*real**4>0]
    negative=minima[minima<0]; positive=minima[minima>0]
    if len(negative)!=1 or len(positive)!=1: raise RuntimeError('未找到预期稳定和亚稳极小值')
    pmeta,pstable=negative[0],positive[0]
    enthalpy=lambda p:a2*p*p+a4*p**4+a6*p**6-field*p
    return dict(status='ok',Ps=ps,intrinsic=intrinsic,Pmeta=pmeta,Pstable=pstable,driving=enthalpy(pmeta)-enthalpy(pstable),approx=2*ps*field)



def generate_structure(shape, fraction, target, seed, tolerance):
    """周期相关随机场合成占位；固定Sc数量，真实12近邻统计匹配SRO。"""
    rng=np.random.default_rng(seed)
    count=round(np.prod(shape)*fraction)
    history=[]
    def build(noise,sigma):
        field=gaussian_filter(noise,sigma=sigma,mode='wrap') if sigma else noise
        ids=np.argpartition(field.ravel(),field.size-count)[-count:]
        species=np.zeros(field.size,dtype=np.uint8);species[ids]=1;species=species.reshape(shape)
        within,between=neighbors(species)
        eta=1-float((within+between)[species==0].sum())/((species.size-count)*12*species.mean())
        history.append([sigma,eta])
        return species,eta
    noise=rng.normal(size=shape)
    best,eta=build(noise,0.)
    # 小体系的随机SRO有统计起伏；目标0时允许可复现地重新抽样。
    if target==0:
        for _ in range(200):
            if abs(eta)<=tolerance:return best,eta,np.array(history)
            best,eta=build(rng.normal(size=shape),0.)
        raise RuntimeError('随机SRO在给定容差内未匹配；增大体系或sro-tol')
    if abs(eta-target)<=tolerance:return best,eta,np.array(history)
    low,high=0.,1.
    for _ in range(12):
        candidate,value=build(noise,high)
        if abs(value-target)<abs(eta-target):best,eta=candidate,value
        if value>=target:break
        high*=2
        if high>max(shape):raise RuntimeError('给定体系无法匹配目标SRO')
    for _ in range(45):
        if abs(eta-target)<=tolerance:return best,eta,np.array(history)
        middle=(low+high)/2
        candidate,value=build(noise,middle)
        if abs(value-target)<abs(eta-target):best,eta=candidate,value
        if value<target:low=middle
        else:high=middle
    raise RuntimeError(f'SRO未匹配：实际{eta:.8g}，可适当放宽sro-tol或增大体系')


def local_statistics(species, block):
    """不重叠索引块统计；SRO按全局x归一化，邻居允许跨块。"""
    shape=tuple(np.array(species.shape)//block)
    def reduce(values):
        return values.reshape(shape[0],block[0],shape[1],block[1],shape[2],block[2]).sum(axis=(1,3,5))
    counts=reduce(species).astype(int)
    within,between=neighbors(species)
    al=reduce(1-species)
    cross=reduce((within+between)*(1-species))
    eta=np.full(shape,np.nan)
    np.divide(cross,al*12*species.mean(),out=eta,where=al>0)
    return counts,1-eta


def wall_energies(fractions, angles, args):
    """每种成分仅求一次；所有中间数据留在内存，不写出剖面文件。"""
    _,charge,potential,ef=make_carriers(args.temperature,args.donors_cm3,args.hole_mass)
    length=args.half_width_nm
    g,eps=5.71e-12,epsilon_0*5
    a4,a6=.149031e9/4,.928609e9/6
    mesh=np.r_[0,np.geomspace(min(1e-4,length/1000),length,650)]
    sampling=np.r_[0,np.geomspace(min(1e-5,length/10000),length,1800),np.linspace(0,length,2001)]
    dimensions=(len(fractions),len(angles))
    gamma=np.full(dimensions,np.nan)
    residual=np.full(dimensions,np.nan);edge_field=np.full(dimensions,np.nan)
    identity=np.full(dimensions,np.nan);gauss=np.full(dimensions,np.nan)
    neutral_check=np.full(len(fractions),np.nan)
    messages=np.full(dimensions,'not_computed',dtype='<U200')
    previous={}
    for i,x in enumerate(fractions):
        a2=5.50378e9*(x-.6115)/2
        if a2>=0:
            messages[i,:]='nonferroelectric';continue
        ps=np.sqrt((-4*a4+np.sqrt((4*a4)**2-48*a6*a2))/(12*a6))
        landau=lambda p:a2*p*p+a4*p**4+a6*p**6
        derivative=lambda p:2*a2*p+4*a4*p**3+6*a6*p**5
        def solve(angle,initial=None):
            sine=np.sin(np.deg2rad(angle))
            def ode(t,y):
                return np.array([y[1],1e-18/(g*ps)*(derivative(ps*y[0])+sine*y[3]*1e9),y[3],(sine*ps*y[1]*1e-9-charge(y[2])*1e-18)/eps])
            if initial is None:
                width=np.sqrt(g/abs(2*a2))/1e-9
                u=-np.tanh(mesh/width)
                guess=np.array([u,-(1-u*u)/width,np.zeros_like(mesh),np.zeros_like(mesh)])
            else:guess=initial.sol(mesh)
            result=solve_bvp(ode,lambda l,r:np.array([l[0],l[3],r[0]+1,r[2]]),mesh,guess,tol=args.tol,max_nodes=40000)
            if not result.success:raise RuntimeError(result.message)
            return result
        current={}
        for j in sorted(range(len(angles)),key=lambda j:abs(angles[j])):
            angle=angles[j]
            try:
                if angle in previous:
                    try:result=solve(angle,previous[angle])
                    except RuntimeError:result=None
                else:result=None
                if result is None:
                    if 0. not in current:current[0.]=solve(0.)
                    result=current[0.]
                    for aa in np.linspace(0,angle,max(1,int(np.ceil(abs(angle)/2)))+1)[1:]:result=solve(aa,result)
                current[angle]=result;previous[angle]=result
                coordinates=np.unique(np.r_[sampling,result.x])
                u,ux,phi,phix=result.sol(coordinates)
                p,dp,dphi=ps*u,ps*ux*1e9,phix*1e9
                sine=np.sin(np.deg2rad(angle))
                base=landau(p)-landau(ps)+.5*g*dp*dp
                density=base-.5*eps*dphi*dphi+sine*p*dphi+potential(phi)
                value=2*simpson(density,x=coordinates*1e-9)
                if not np.isfinite(value) or value<=0:raise RuntimeError('畴壁能非正或非有限')
                gamma[i,j]=value
                residual[i,j]=max(result.rms_residuals)
                edge_field[i,j]=-dphi[-1]
                alternative=2*simpson(base+.5*eps*dphi*dphi+potential(phi)-phi*charge(phi),x=coordinates*1e-9)
                identity[i,j]=value-alternative
                gauss[i,j]=2*simpson(charge(phi),x=coordinates*1e-9)+2*ps*sine+2*eps*dphi[-1]
                if angle==0:
                    exact=quad(lambda p:np.sqrt(max(0,2*g*(landau(p)-landau(ps)))),-ps,ps,epsabs=1e-10)[0]
                    neutral_check[i]=value-exact
                messages[i,j]='ok'
            except RuntimeError as exc:
                messages[i,j]=str(exc);previous.pop(angle,None)
        if i%10==0 or i==len(fractions)-1:
            print(f'畴壁成分 {i+1}/{len(fractions)}, x={x:.4f}',flush=True)
    return gamma,dict(wall_residual=residual,wall_edge_field_V_m=edge_field,wall_energy_identity_error_J_m2=identity,wall_gauss_error_C_m2=gauss,neutral_energy_error_J_m2=neutral_check,wall_messages=messages,fermi_level_eV=ef)


def nucleation(fractions, angles, gamma, args):
    """半平面Wulff -> 轴对称三维核 -> 经典形核；无亚稳态独立标记。"""
    names=['Ps_C_m2','intrinsic_field_MV_cm','Pmeta_C_m2','Pstable_C_m2','driving_J_m3','approx_2PsE_J_m3','W0_J','V0_m3','A0_m2','lambda_star','barrier_J','barrier_eV','critical_height_nm','critical_diameter_nm','critical_volume_nm3','log10_R_over_R0','wall_at_star_J','drive_at_star_J']
    fields={name:np.full(len(fractions),np.nan) for name in names}
    status=np.full(len(fractions),3,dtype=np.int32)
    polygons=[]
    for i,x in enumerate(fractions):
        bulk=bulk_driving(x,args.field_mv_cm*1e8)
        if bulk is None:
            status[i]=2;polygons.append(np.empty((0,2)));continue
        fields['Ps_C_m2'][i]=bulk['Ps'];fields['intrinsic_field_MV_cm'][i]=bulk['intrinsic']/1e8
        if not np.isfinite(gamma[i]).all():polygons.append(np.empty((0,2)));continue
        vertices,_=wulff_polygon(angles,gamma[i])
        polygons.append(vertices)
        volume,wall,area,height,width=reference_geometry(vertices,angles,gamma[i])
        for key,value in [('W0_J',wall),('V0_m3',volume),('A0_m2',area)]:fields[key][i]=value
        if bulk['status']!='ok':status[i]=1;continue
        status[i]=0
        driving=bulk['driving'];star=2*wall/(3*volume*driving)
        barrier=4*wall**3/(27*(volume*driving)**2)
        values=dict(Pmeta_C_m2=bulk['Pmeta'],Pstable_C_m2=bulk['Pstable'],driving_J_m3=driving,approx_2PsE_J_m3=bulk['approx'],lambda_star=star,barrier_J=barrier,barrier_eV=barrier/q,critical_height_nm=star*height*1e9,critical_diameter_nm=star*width*1e9,critical_volume_nm3=star**3*volume*1e27,log10_R_over_R0=-barrier/(kb*args.temperature*np.log(10)),wall_at_star_J=wall*star**2,drive_at_star_J=-driving*volume*star**3)
        for key,value in values.items():fields[key][i]=value
    padded=np.full((len(fractions),max(1,max(map(len,polygons))),2),np.nan)
    for i,polygon in enumerate(polygons):padded[i,:len(polygon)]=polygon
    return fields,status,padded,np.array([len(p) for p in polygons])


def save_vtk(path, shape, spacing, fields, metadata):
    """规则体单元VTK，非有限数据以0占位并附掩码；元数据也写入Field Data。"""
    image=vtk.vtkRectilinearGrid();image.SetDimensions(*(np.array(shape)+1))
    for axis,setter in enumerate([image.SetXCoordinates,image.SetYCoordinates,image.SetZCoordinates]):
        setter(numpy_to_vtk(np.arange(shape[axis]+1,dtype=float)*spacing[axis],deep=True))
    for key,values in fields.items():
        finite=np.isfinite(values)
        array=numpy_to_vtk(np.where(finite,values,0).ravel(order='F'),deep=True)
        array.SetName(key);image.GetCellData().AddArray(array)
        if not finite.all():
            mask=numpy_to_vtk(finite.astype(np.uint8).ravel(order='F'),deep=True);mask.SetName(key+'_isfinite');image.GetCellData().AddArray(mask)
    image.GetCellData().SetActiveScalars('log10_relative_rate_max')
    # JSON以UTF-8字节保存，避免legacy VTK字符串Field Data的读取兼容性问题。
    encoded=np.frombuffer(json.dumps(metadata,ensure_ascii=True).encode('utf-8'),dtype=np.uint8)
    config=numpy_to_vtk(encoded,deep=True);config.SetName('configuration_json_utf8');image.GetFieldData().AddArray(config)
    for key in ['seed','actual_sro','actual_x_sc','field_mv_cm','temperature']:
        array=numpy_to_vtk(np.array([metadata[key]],dtype=float),deep=True);array.SetName(key);image.GetFieldData().AddArray(array)
    writer=vtk.vtkRectilinearGridWriter();writer.SetInputData(image);writer.SetFileName(str(path));writer.SetFileTypeToBinary()
    if writer.Write()!=1:raise RuntimeError('VTK写入失败')


def triplet(values,name):
    if len(values)==1:return (values[0],)*3
    if len(values)==3:return tuple(values)
    raise ValueError(name+'需要一个整数或三个整数')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,nargs='+',default=[100],help='阳离子数组尺寸，单值或Nx Ny Nz；Nz必须为偶数')
    parser.add_argument('--block',type=int,nargs='+',default=[10],help='不重叠微区尺寸，单值或Bx By Bz')
    parser.add_argument('--x-sc',type=float,default=.25)
    parser.add_argument('--sro',type=float,default=.5)
    parser.add_argument('--sro-tol',type=float,default=.001)
    parser.add_argument('--seed',type=int,default=20261126)
    parser.add_argument('--field-mv-cm',type=float,default=3.)
    parser.add_argument('--temperature',type=float,default=298.)
    parser.add_argument('--angle-step',type=int,default=15)
    parser.add_argument('--half-width-nm',type=float,default=200.)
    parser.add_argument('--tol',type=float,default=1e-5)
    parser.add_argument('--donors-cm3',type=float,default=1e18)
    parser.add_argument('--hole-mass',type=float,default=1.)
    parser.add_argument('--lattice-a-nm',type=float,default=.32)
    parser.add_argument('--lattice-c-nm',type=float,default=.51)
    parser.add_argument('--y-copies',type=int,default=2,help='正交重排后的Y向周期副本数量')
    parser.add_argument('--subdiv',type=int,default=1,help='每微区每方向的相场体素细分数')
    parser.add_argument('--out',type=Path,default=Path('run_default'),help='最终两文件的输出目录')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    try:
        size=triplet(args.size,'size');block=triplet(args.block,'block')
        if min(size)<3 or size[2]%2 or min(block)<1 or any(n%b for n,b in zip(size,block)):raise ValueError('各方向尺寸>=3、Nz为偶数、block须为size的约数')
        if not 0<args.x_sc<1 or not 0<=args.sro<1:raise ValueError('0<x-sc<1，目前合成器支持0<=SRO<1')
        if not 0<round(np.prod(size)*args.x_sc)<np.prod(size):raise ValueError('体系至少需一个Al和一个Sc')
        if args.angle_step<=0 or 90%args.angle_step:raise ValueError('angle-step必须为90的正整数约数')
        positive=[args.sro_tol,args.field_mv_cm,args.temperature,args.half_width_nm,args.tol,args.hole_mass,args.lattice_a_nm,args.lattice_c_nm]
        if not np.isfinite(positive).all() or min(positive)<=0 or not np.isfinite(args.donors_cm3) or args.donors_cm3<0 or args.y_copies<1 or args.subdiv<1:raise ValueError('尺度/容差/温度/外场/副本和细分须为正；施主浓度非负')
        if not args.overwrite and any((args.out/name).exists() for name in ['results.npz','results.vtk']):raise ValueError('输出已存在，请另选out或添加--overwrite')
    except ValueError as exc:parser.error(str(exc))
    species,achieved,history=generate_structure(size,args.x_sc,args.sro,args.seed,args.sro_tol)
    counts,eta=local_statistics(species,block)
    block_shape=counts.shape;block_atoms=int(np.prod(block))
    unique,inverse=np.unique(counts,return_inverse=True)
    mapping=inverse.reshape(block_shape)
    fractions=unique/block_atoms
    base_angles=np.arange(-90,91,args.angle_step,dtype=float)
    print(f'构型: {size}, x={species.mean():.6f}, SRO={achieved:.6f}; 微区{block_shape}, 唯一成分{len(unique)}',flush=True)
    gamma_base,diagnostics=wall_energies(fractions,base_angles,args)
    angles=np.arange(0,361,args.angle_step,dtype=float)
    equivalent=np.round(np.rad2deg(np.arcsin(np.sin(np.deg2rad(angles))))/args.angle_step)*args.angle_step
    indices=np.array([np.flatnonzero(base_angles==angle)[0] for angle in equivalent])
    gamma=gamma_base[:,indices]
    unique_fields,status,vertices,vertex_counts=nucleation(fractions,angles,gamma,args)
    original={key:values[mapping] for key,values in unique_fields.items()}
    original.update(x_Sc=counts/block_atoms,Sc_count=counts,SRO_global_reference=eta,status_id=status[mapping],valid=(status[mapping]==0).astype(np.uint8))
    # 默认采用最新的“每微区长方体”索引重排，不是保持WZ内部坐标的刚性旋转。
    final_shape=(block_shape[0],block_shape[1]*args.y_copies,block_shape[2])
    ii,jj,kk=np.indices(final_shape)
    source_id=np.ravel_multi_index((ii,jj%block_shape[1],kk),block_shape)
    macro={key:value.ravel()[source_id] for key,value in original.items()}
    macro.update(subregion_id=np.arange(np.prod(final_shape)).reshape(final_shape),source_block_id=source_id,source_block_i=ii,source_block_j=jj%block_shape[1],source_block_k=kk,periodic_copy_y=jj//block_shape[1],block_i=ii,block_j=jj,block_k=kk)
    block_spacing=np.array([block[0]*args.lattice_a_nm,block[1]*np.sqrt(3)*args.lattice_a_nm/2,block[2]*args.lattice_c_nm/2])
    block_volume=float(np.prod(block_spacing)*1e-27)
    log_rate=macro['log10_R_over_R0'];valid=macro['valid'].astype(bool)
    relative=np.full(final_shape,np.nan);weight=np.full(final_shape,np.nan)
    if valid.any():
        relative[valid]=log_rate[valid]-np.max(log_rate[valid])
        normalizer=logsumexp(log_rate[valid]*np.log(10))/np.log(10)
        weight[valid]=log_rate[valid]-normalizer
    macro.update(log10_relative_rate_max=relative,relative_rate_max=np.power(10.,relative),log10_relative_event_weight=weight,relative_event_weight=np.power(10.,weight),log10_count_coefficient_m3=log_rate+np.log10(block_volume),block_volume_m3=np.full(final_shape,block_volume))
    for axis,name in enumerate('xyz'):
        index=[ii,jj,kk][axis]
        macro[name+'_min_nm']=index*block_spacing[axis];macro[name+'_max_nm']=(index+1)*block_spacing[axis]
    # 可选细分不改变微区参数，仅拆分积分事件权重。
    final={}
    for key,value in macro.items():
        for axis in range(3):value=np.repeat(value,args.subdiv,axis=axis)
        final[key]=value
    final['relative_event_weight']/=args.subdiv**3
    final['log10_relative_event_weight']-=3*np.log10(args.subdiv)
    final['log10_count_coefficient_m3']-=3*np.log10(args.subdiv)
    final['cell_volume_m3']=np.full_like(final['barrier_J'],block_volume/args.subdiv**3)
    cell_shape=final['x_Sc'].shape
    metadata=vars(args).copy();metadata['out']=str(args.out);metadata.update(size=list(size),block=list(block),actual_x_sc=float(species.mean()),actual_sro=float(achieved),original_block_shape=list(block_shape),orthogonal_block_shape=list(final_shape),cell_shape=list(cell_shape),spacing_nm=(block_spacing/args.subdiv).tolist(),box_lengths_nm=(block_spacing*np.array(final_shape)).tolist(),status_codes={'0':'valid_CNT','1':'metastable_absent','2':'nonferroelectric','3':'wall_solver_failed'},rate_calibration='common R0; relative quantities only',geometry='index-preserving orthogonal rearrangement; original internal physical positions NOT preserved',landau='thermo.pdf Table I stress-free; a2=alpha(x)/2,a4=gamma/4,a6=omega/6',wall_model='zero-field planar wall, g=5.71e-12, epsilon_b=5, Eg=3.3eV, electron_mass=0.27m_e',reference_half_height_nm=1.,invalid_unique=int(np.count_nonzero(status)),invalid_original_blocks=int(np.count_nonzero(original['status_id'])),failed_unique=int(np.count_nonzero(status==3)))
    args.out.mkdir(parents=True,exist_ok=True)
    # NPZ保留构型和参数追溯，但不保存大量中间剖面或图片；无对象数组，不需要pickle。
    np.savez_compressed(args.out/'results.npz',**final,species_sc=species,original_sc_counts=counts,original_sro=eta,unique_x_sc=fractions,unique_sc_counts=unique,unique_status=status,unique_gamma_J_m2=gamma,angles_deg=angles,base_angles_deg=base_angles,wulff_vertices_normalized=vertices,wulff_vertex_count=vertex_counts,sro_search_history=history,metadata_json=np.array(json.dumps(metadata)),**diagnostics)
    save_vtk(args.out/'results.vtk',cell_shape,block_spacing/args.subdiv,final,metadata)
    print(f'完成: {args.out}/results.npz 和 results.vtk；有效原微区{int(original["valid"].sum())}/{counts.size}，失败成分{metadata["failed_unique"]}',flush=True)
    if metadata['failed_unique']:raise SystemExit('存在畴壁求解失败，已保存带状态标记的最终文件；请查看NPZ wall_messages')


if __name__=='__main__':main()
