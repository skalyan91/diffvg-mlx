import json
import copy
import xml.etree.ElementTree as etree
from xml.dom import minidom
import warnings
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import re
import sys
import pydiffvg
from pydiffvg.color import parse_color_string
import math
from collections import namedtuple
import cssutils

def _unsupported_asgd(*args, **kwargs):
    raise ValueError("The 'ASGD' optimizer has no MLX equivalent and is not supported; use 'Adam' or 'SGD'.")

def _adam(learning_rate):
    # the original (PyTorch) Adam applied bias correction; match it
    return optim.Adam(learning_rate=learning_rate, bias_correction=True)

def _sgd(learning_rate):
    return optim.SGD(learning_rate=learning_rate)

class SvgOptimizationSettings:

    default_params = {
        "optimize_color": True,
        "color_lr": 2e-3,
        "optimize_alpha": False,
        "alpha_lr": 2e-3,
        "optimizer": "Adam",
        "transforms": {
            "optimize_transforms":True,
            "transform_mode":"rigid",
            "translation_mult":1e-3,
            "transform_lr":2e-3
        },
        "circles": {
            "optimize_center": True,
            "optimize_radius": True,
            "shape_lr": 2e-1
        },
        "paths": {
            "optimize_points": True,
            "shape_lr": 2e-1
        },
        "gradients": {
            "optimize_stops": True,
            "stop_lr": 2e-3,
            "optimize_color": True,
            "color_lr": 2e-3,
            "optimize_alpha": False,
            "alpha_lr": 2e-3,
            "optimize_location": True,
            "location_lr": 2e-1
        }
    }

    # factories: name -> callable(learning_rate) -> mlx.optimizers.Optimizer
    # learning_rate may be a float or an mlx.optimizers.schedulers schedule
    optims = {
        "Adam": _adam,
        "SGD": _sgd,
        "ASGD": _unsupported_asgd,
    }

    @staticmethod
    def make_optimizer(name, lr):
        if name not in SvgOptimizationSettings.optims:
            raise ValueError("Unknown optimizer '{}'; supported: 'Adam', 'SGD'".format(name))
        return SvgOptimizationSettings.optims[name](lr)

    #region methods
    def __init__(self, f=None):
        self.store = {}
        if f is None:
            self.store["default"] = copy.deepcopy(SvgOptimizationSettings.default_params)
        else:
            self.store = json.load(f)

    # create default alias for root
    def default_name(self, dname):
        self.dname = dname
        if dname not in self.store:
            self.store[dname] = self.store["default"]

    def retrieve(self, node_id):
        if node_id not in self.store:
            return (self.store["default"], False)
        else:
            return (self.store[node_id], True)

    def reset_to_defaults(self, node_id):
        if node_id in self.store:
            del self.store[node_id]

        return self.store["default"]

    def undefault(self, node_id):
        if node_id not in self.store:
            self.store[node_id] = copy.deepcopy(self.store["default"])

        return self.store[node_id]

    def override_optimizer(self, optimizer):
        if optimizer is not None:
            for v in self.store.values():
                v["optimizer"] = optimizer

    def global_override(self, path, value):
        for store in self.store.values():
            d = store
            for key in path[:-1]:
                d = d[key]

            d[path[-1]] = value

    def save(self, file):
        self.store["default"] = self.store[self.dname]
        json.dump(self.store, file, indent="\t")
    #endregion

def _f32(x):
    return mx.array(x, dtype=mx.float32)

def _np(x):
    return np.array(x, dtype=np.float64)

def _matmul(a, b):
    # MLX's GPU matmul loses ~1e-3 of precision even on 3x3 float32 matrices;
    # the CPU stream is exact (and rendering is CPU-only anyway). Differentiable.
    return mx.matmul(a, b, stream=mx.cpu)

class OptimizableSvg:
    """
        MLX port. MLX has no requires_grad / .grad / .backward(), so optimisation works
        on an explicit parameter tree:

            params = svg.get_params()               # nested dict of mx.arrays
            loss, grads = svg.value_and_grad(loss_fn)()   # loss_fn(svg, *args) -> scalar
            svg.step(grads)                         # per-parameter optimizers + clamps

        or simply `loss = svg.optimize_step(loss_fn)`. Inside loss_fn call svg.render()
        or svg.build_scene() to get differentiable images/scenes.
    """

    class TransformTools:
        @staticmethod
        def parse_matrix(vals):
            assert(len(vals)==6)
            return np.array([[vals[0],vals[2],vals[4]],[vals[1], vals[3], vals[5]],[0,0,1]])

        @staticmethod
        def parse_translate(vals):
            assert(len(vals)>=1 and len(vals)<=2)
            mat=np.eye(3)
            mat[0,2]=vals[0]
            if len(vals)>1:
                mat[1,2]=vals[1]
            return mat

        @staticmethod
        def parse_rotate(vals):
            assert (len(vals) == 1 or len(vals) == 3)
            mat = np.eye(3)
            rads=math.radians(vals[0])
            sint=math.sin(rads)
            cost=math.cos(rads)
            mat[0:2, 0:2] = np.array([[cost,-sint],[sint,cost]])
            if len(vals) > 1:
                tr1=OptimizableSvg.TransformTools.parse_translate(vals[1:3])
                tr2=OptimizableSvg.TransformTools.parse_translate([-vals[1],-vals[2]])
                mat=tr1 @ mat @ tr2
            return mat

        @staticmethod
        def parse_scale(vals):
            assert (len(vals) >= 1 and len(vals) <= 2)
            d=np.array([vals[0], vals[1] if len(vals)>1 else vals[0],1])
            return np.diag(d)

        @staticmethod
        def parse_skewx(vals):
            assert(len(vals)==1)
            m=np.eye(3)
            m[0,1]=vals[0]
            return m

        @staticmethod
        def parse_skewy(vals):
            assert (len(vals) == 1)
            m = np.eye(3)
            m[1, 0] = vals[0]
            return m

        @staticmethod
        def transformPoints(pointsTensor, transform):
            assert(transform is not None)
            one=mx.ones((pointsTensor.shape[0],1),dtype=pointsTensor.dtype)
            homo_points = mx.concatenate([pointsTensor, one], axis=1)
            mult = _matmul(transform, homo_points.T).T
            tfpoints=mult[:, 0:2]
            assert(pointsTensor.shape == tfpoints.shape)
            return tfpoints

        @staticmethod
        def promote_numpy(M):
            ret = np.eye(3)
            ret[0:2, 0:2] = M
            return ret

        @staticmethod
        def recompose_numpy(Theta,ScaleXY,ShearX,TXY):
            cost=math.cos(Theta)
            sint=math.sin(Theta)
            Rot=np.array([[cost, -sint],[sint, cost]])
            Scale=np.diag(ScaleXY)
            Shear=np.eye(2)
            Shear[0,1]=ShearX

            Translate=np.eye(3)
            Translate[0:2,2]=TXY

            M=OptimizableSvg.TransformTools.promote_numpy(Rot @ Scale @ Shear) @ Translate
            return M

        @staticmethod
        def promote(m):
            # differentiable 2x2 -> 3x3 homogeneous embedding
            corner=mx.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]],dtype=m.dtype)
            return mx.pad(m,[(0,1),(0,1)])+corner

        @staticmethod
        def make_rot(Theta):
            Theta=mx.array(Theta,dtype=mx.float32).reshape(())
            sint=mx.sin(Theta)
            cost=mx.cos(Theta)
            Rot=mx.stack((mx.stack((cost,-sint)),mx.stack((sint,cost))))
            return Rot

        @staticmethod
        def make_scale(ScaleXY):
            ScaleXY=mx.array(ScaleXY,dtype=mx.float32)
            if ScaleXY.size==1:
                #uniform scale
                s=ScaleXY.reshape(())
                return mx.diag(mx.stack([s,s]))
            else:
                return mx.diag(ScaleXY.reshape(-1))

        @staticmethod
        def make_shear(ShearX):
            ShearX=mx.array(ShearX,dtype=mx.float32).reshape(())
            one=mx.array(1.,dtype=mx.float32)
            zero=mx.array(0.,dtype=mx.float32)
            return mx.stack((mx.stack((one,ShearX)),mx.stack((zero,one))))

        @staticmethod
        def make_translate(TXY):
            TXY=mx.array(TXY,dtype=mx.float32).reshape(2,1)
            return mx.eye(3,dtype=mx.float32)+mx.pad(TXY,[(0,1),(2,0)])

        @staticmethod
        def recompose(Theta,ScaleXY,ShearX,TXY):
            Rot=OptimizableSvg.TransformTools.make_rot(Theta)
            Scale=OptimizableSvg.TransformTools.make_scale(ScaleXY)
            Shear=OptimizableSvg.TransformTools.make_shear(ShearX)
            Translate=OptimizableSvg.TransformTools.make_translate(TXY)

            return _matmul(OptimizableSvg.TransformTools.promote(_matmul(_matmul(Rot, Scale), Shear)), Translate)

        TransformDecomposition=namedtuple("TransformDecomposition","theta scale shear translate")
        TransformProperties=namedtuple("TransformProperties", "has_rotation has_scale has_mirror scale_uniform has_shear has_translation")

        @staticmethod
        def make_named(decomp):
            if not isinstance(decomp,OptimizableSvg.TransformTools.TransformDecomposition):
                decomp=OptimizableSvg.TransformTools.TransformDecomposition(theta=decomp[0],scale=decomp[1],shear=decomp[2],translate=decomp[3])
            return decomp

        @staticmethod
        def analyze_transform(decomp):
            decomp=OptimizableSvg.TransformTools.make_named(decomp)
            theta=float(_np(decomp.theta).reshape(-1)[0])
            scale=_np(decomp.scale).reshape(-1)
            shear=float(_np(decomp.shear).reshape(-1)[0])
            translate=_np(decomp.translate).reshape(-1)
            epsilon=1e-3
            has_rotation=abs(theta)>epsilon
            has_scale=np.abs(np.abs(scale)-1).max()>epsilon
            scale_len=scale.size>1
            has_mirror=bool(scale_len and scale[0]*scale[1] < 0)
            scale_uniform=bool(not scale_len or abs(abs(scale[0])-abs(scale[1]))<epsilon)
            has_shear=abs(shear)>epsilon
            has_translate=max(abs(translate[0]),abs(translate[1]))>epsilon

            return OptimizableSvg.TransformTools.TransformProperties(has_rotation=has_rotation,has_scale=has_scale,has_mirror=has_mirror,scale_uniform=scale_uniform,has_shear=has_shear,has_translation=has_translate)

        @staticmethod
        def check_and_decomp(M):
            decomp=OptimizableSvg.TransformTools.decompose(M) if M is not None else OptimizableSvg.TransformTools.TransformDecomposition(theta=0,scale=(1,1),shear=0,translate=(0,0))
            props=OptimizableSvg.TransformTools.analyze_transform(decomp)
            return (decomp, props)

        @staticmethod
        def tf_to_string(M):
            M=_np(M)
            tfstring = "matrix({} {} {} {} {} {})".format(M[0, 0], M[1, 0], M[0, 1], M[1, 1], M[0, 2], M[1, 2])
            return tfstring

        @staticmethod
        def decomp_to_string(decomp):
            decomp = OptimizableSvg.TransformTools.make_named(decomp)
            theta=float(_np(decomp.theta).reshape(-1)[0])
            scale=_np(decomp.scale).reshape(-1)
            shear=float(_np(decomp.shear).reshape(-1)[0])
            translate=_np(decomp.translate).reshape(-1)
            ret=""
            props=OptimizableSvg.TransformTools.analyze_transform(decomp)
            if props.has_rotation:
                ret+="rotate({}) ".format(math.degrees(theta))
            if props.has_scale:
                if scale.size==1:
                    ret += "scale({}) ".format(scale[0])
                else:
                    ret+="scale({} {}) ".format(scale[0], scale[1])
            if props.has_shear:
                ret+="skewX({}) ".format(shear)
            if props.has_translation:
                ret+="translate({} {}) ".format(translate[0],translate[1])

            return ret

        @staticmethod
        def decompose(M):
            m = M[0:2, 0:2]
            t0=M[0:2, 2]
            #get translation so that we can post-multiply with it
            TXY=np.linalg.solve(m,t0)

            T=np.eye(3)
            T[0:2,2]=TXY

            q, r = np.linalg.qr(m)

            ref = np.array([[1, 0], [0, np.sign(np.linalg.det(q))]])

            Rot = np.dot(q, ref)

            ref2 = np.array([[1, 0], [0, np.sign(np.linalg.det(r))]])

            r2 = np.dot(ref2, r)

            Ref = np.dot(ref, ref2)

            sc = np.diag(r2)
            Scale = np.diagflat(sc)

            Shear = np.eye(2)
            Shear[0, 1] = r2[0, 1] / sc[0]
            #the actual shear coefficient
            ShearX=r2[0, 1] / sc[0]

            if np.sum(sc) < 0:
                # both scales are negative, flip this and add a 180 rotation
                Rot = np.dot(Rot, -np.eye(2))
                Scale = -Scale

            Theta = math.atan2(Rot[1, 0], Rot[0, 0])
            ScaleXY = np.array([Scale[0,0],Scale[1,1]*Ref[1,1]])

            return OptimizableSvg.TransformTools.TransformDecomposition(theta=Theta, scale=ScaleXY, shear=ShearX, translate=TXY)

    #region suboptimizers

    class ParamGroup:
        """
            A named set of mx.array parameters, each with its own MLX optimizer
            (so every parameter can carry its own learning rate).
            on_update(params) is called whenever values change, so the owning node
            can write the arrays back into its own fields.
            post(params) -> params is applied after each optimizer update (clamps etc.).
        """
        def __init__(self, params, optim_name, lrs, on_update=None, post=None):
            self.params={k: v for k, v in params.items()}
            if not isinstance(lrs, dict):
                lrs={k: lrs for k in self.params}
            self.optims={k: SvgOptimizationSettings.make_optimizer(optim_name, lrs[k]) for k in self.params}
            self.on_update=on_update
            self.post=post

        def get_params(self):
            return dict(self.params)

        def set_params(self, params):
            self.params={k: params[k] for k in self.params}
            if self.on_update is not None:
                self.on_update(self.params)

        def zero_grad(self):
            pass

        def step(self, grads):
            new={}
            for k, p in self.params.items():
                g=grads.get(k) if grads is not None else None
                if g is None:
                    new[k]=p
                    continue
                new[k]=self.optims[k].apply_gradients({"p": g}, {"p": p})["p"]
            if self.post is not None:
                new=self.post(new)
            mx.eval(new, [o.state for o in self.optims.values()])
            self.set_params(new)

    #optimizes color, but really any tensor that needs to stay between 0 and 1 per-entry
    class ColorOptimizer(ParamGroup):
        def __init__(self,tensor,optim_type,lr,on_update=None):
            super().__init__({"value": tensor}, optim_type, lr,
                             on_update=(lambda p: on_update(p["value"])) if on_update is not None else None,
                             post=lambda p: {"value": mx.clip(p["value"], 1e-4, 1.)})

        @property
        def tensor(self):
            return self.params["value"]

    #optimizes gradient stop positions
    class StopOptimizer(ParamGroup):
        def __init__(self,stops,optim_type,lr,on_update=None):
            super().__init__({"value": stops}, optim_type, lr,
                             on_update=(lambda p: on_update(p["value"])) if on_update is not None else None,
                             post=OptimizableSvg.StopOptimizer._fix_stops)

        @staticmethod
        def _fix_stops(p):
            s=mx.sort(mx.clip(p["value"], 0., 1.))
            if s.shape[0]>=2:
                s=mx.concatenate([mx.zeros((1,),dtype=s.dtype), s[1:-1], mx.ones((1,),dtype=s.dtype)])
            return {"value": s}

        @property
        def stops(self):
            return self.params["value"]

    class CompositeOptimizer:
        """ A named collection of ParamGroups exposed as one nested parameter dict. """
        def __init__(self):
            self.groups={}

        def get_params(self):
            return {k: g.get_params() for k, g in self.groups.items()}

        def set_params(self, params):
            for k, g in self.groups.items():
                g.set_params(params[k])

        def zero_grad(self):
            pass

        def step(self, grads):
            for k, g in self.groups.items():
                if grads is not None and k in grads:
                    g.step(grads[k])

    #optimizes gradient: stop, positions, colors+opacities, locations
    class GradientOptimizer(CompositeOptimizer):
        def __init__(self, begin, end, offsets, stops, optim_params):
            super().__init__()
            self.begin=_f32(begin) if begin is not None else None
            self.end=_f32(end) if end is not None else None
            self.offsets=_f32(offsets) if offsets is not None else None
            self.stop_colors=_f32(stops)[:,0:3] if stops is not None else None
            self.stop_alphas=_f32(stops)[:,3] if stops is not None else None
            oname=optim_params["optimizer"]

            if optim_params["gradients"]["optimize_stops"] and self.offsets is not None:
                self.groups["offsets"]=OptimizableSvg.StopOptimizer(self.offsets,oname,optim_params["gradients"]["stop_lr"],
                                                                     on_update=lambda v: setattr(self,"offsets",v))
            if optim_params["gradients"]["optimize_color"] and self.stop_colors is not None:
                self.groups["stop_colors"]=OptimizableSvg.ColorOptimizer(self.stop_colors,oname,optim_params["gradients"]["color_lr"],
                                                                         on_update=lambda v: setattr(self,"stop_colors",v))
            if optim_params["gradients"]["optimize_alpha"] and self.stop_alphas is not None:
                self.groups["stop_alphas"]=OptimizableSvg.ColorOptimizer(self.stop_alphas,oname,optim_params["gradients"]["alpha_lr"],
                                                                         on_update=lambda v: setattr(self,"stop_alphas",v))
            if optim_params["gradients"]["optimize_location"] and self.begin is not None and self.end is not None:
                def upd(p):
                    self.begin=p["begin"]
                    self.end=p["end"]
                self.groups["location"]=OptimizableSvg.ParamGroup({"begin": self.begin, "end": self.end},oname,
                                                                  optim_params["gradients"]["location_lr"],on_update=upd)

        def get_vals(self):
            return self.begin, self.end, self.offsets, mx.concatenate((self.stop_colors,self.stop_alphas[:,None]),1) if self.stop_colors is not None and self.stop_alphas is not None else None

    class TransformOptimizer:
        def __init__(self,transform,optim_params):
            self.transform=transform
            self.optimizes=optim_params["transforms"]["optimize_transforms"] and transform is not None
            self.params=copy.deepcopy(optim_params)
            self.transform_mode=optim_params["transforms"]["transform_mode"]
            self.group=None

            if self.optimizes:
                self.residual=None
                self.scale_sign=None
                self.shear=None
                lr=optim_params["transforms"]["transform_lr"]
                tmult=optim_params["transforms"]["translation_mult"]
                decomp,props=OptimizableSvg.TransformTools.check_and_decomp(np.array(transform,dtype=np.float64))
                if self.transform_mode=="move":
                    #only translation and rotation should be set
                    if props.has_scale or props.has_shear or props.has_mirror:
                        print("Warning: set to optimize move only, but input transform has residual scale or shear")
                        self.residual=mx.stop_gradient(self.transform)
                        self.Theta=_f32(0.)
                        self.translation=_f32([0, 0])
                    else:
                        self.residual=None
                        self.Theta=_f32(decomp.theta)
                        self.translation=_f32(decomp.translate)
                    names={"Theta": lr, "translation": lr*tmult}
                elif self.transform_mode=="rigid":
                    #only translation, rotation, and uniform scale should be set
                    if props.has_shear or props.has_mirror or not props.scale_uniform:
                        print("Warning: set to optimize rigid transform only, but input transform has residual shear, mirror or non-uniform scale")
                        self.residual = mx.stop_gradient(self.transform)
                        self.Theta = _f32(0.)
                        self.translation = _f32([0, 0])
                        self.scale=_f32(1.)
                    else:
                        self.residual = None
                        self.Theta = _f32(decomp.theta)
                        self.translation = _f32(decomp.translate)
                        self.scale = _f32(decomp.scale[0])
                    names={"Theta": lr, "scale": lr, "translation": lr*tmult}
                elif self.transform_mode=="similarity":
                    if props.has_shear or not props.scale_uniform:
                        print("Warning: set to optimize rigid transform only, but input transform has residual shear or non-uniform scale")
                        self.residual = mx.stop_gradient(self.transform)
                        self.Theta = _f32(0.)
                        self.translation = _f32([0, 0])
                        self.scale=_f32(1.)
                        self.scale_sign=_f32(1.)
                    else:
                        self.residual = None
                        self.Theta = _f32(decomp.theta)
                        self.translation = _f32(decomp.translate)
                        self.scale = _f32(decomp.scale[0])
                        self.scale_sign = _f32(np.sign(decomp.scale[0]*decomp.scale[1]))
                    names={"Theta": lr, "scale": lr, "translation": lr*tmult}
                elif self.transform_mode=="affine":
                    self.Theta = _f32(decomp.theta)
                    self.translation = _f32(decomp.translate)
                    self.scale = _f32(decomp.scale)
                    self.shear = _f32(decomp.shear)
                    names={"Theta": lr, "scale": lr, "shear": lr, "translation": lr*tmult}
                else:
                    raise ValueError("Unrecognized transform mode '{}'".format(self.transform_mode))

                def upd(p):
                    for k, v in p.items():
                        setattr(self, k, v)
                self.group=OptimizableSvg.ParamGroup({k: getattr(self,k) for k in names},optim_params["optimizer"],names,on_update=upd)

        def get_params(self):
            return self.group.get_params() if self.group is not None else {}

        def set_params(self, params):
            if self.group is not None:
                self.group.set_params(params)

        def _similarity_scale(self):
            s=self.scale.reshape(1)
            return mx.concatenate((s,s*self.scale_sign.reshape(1)))

        def get_transform(self):
            if not self.optimizes:
                return self.transform
            else:
                zero=mx.array(0.,dtype=mx.float32)
                if self.transform_mode == "move":
                    composed=OptimizableSvg.TransformTools.recompose(self.Theta,mx.array([1.],dtype=mx.float32),zero,self.translation)
                    return _matmul(self.residual, composed) if self.residual is not None else composed
                elif self.transform_mode == "rigid":
                    composed = OptimizableSvg.TransformTools.recompose(self.Theta, self.scale, zero, self.translation)
                    return _matmul(self.residual, composed) if self.residual is not None else composed
                elif self.transform_mode == "similarity":
                    composed=OptimizableSvg.TransformTools.recompose(self.Theta, self._similarity_scale(),zero,self.translation)
                    return _matmul(self.residual, composed) if self.residual is not None else composed
                elif self.transform_mode == "affine":
                    composed = OptimizableSvg.TransformTools.recompose(self.Theta, self.scale, self.shear, self.translation)
                    return composed
                else:
                    raise ValueError("Unrecognized transform mode '{}'".format(self.transform_mode))

        def tfToString(self):
            if self.transform is None:
                return None
            elif not self.optimizes:
                return OptimizableSvg.TransformTools.tf_to_string(self.transform)
            else:
                if self.transform_mode == "move":
                    str=OptimizableSvg.TransformTools.decomp_to_string((self.Theta,np.array([1.]),0.,self.translation))
                    return (OptimizableSvg.TransformTools.tf_to_string(self.residual) if self.residual is not None else "")+" "+str
                elif self.transform_mode == "rigid":
                    str = OptimizableSvg.TransformTools.decomp_to_string((self.Theta, self.scale, 0., self.translation))
                    return (OptimizableSvg.TransformTools.tf_to_string(self.residual) if self.residual is not None else "")+" "+str
                elif self.transform_mode == "similarity":
                    str=OptimizableSvg.TransformTools.decomp_to_string((self.Theta, self._similarity_scale(),0.,self.translation))
                    return (OptimizableSvg.TransformTools.tf_to_string(self.residual) if self.residual is not None else "")+" "+str
                elif self.transform_mode == "affine":
                    str = OptimizableSvg.TransformTools.decomp_to_string((self.Theta, self.scale, self.shear, self.translation))
                    return str

        def zero_grad(self):
            pass

        def step(self, grads):
            if self.group is not None and grads is not None:
                self.group.step(grads)

    #endregion

    #region Nodes
    class SvgNode:
        def __init__(self,id,transform,appearance,settings):
            self.id=id
            self.children=[]
            # name -> sub-optimizer (ParamGroup / CompositeOptimizer / TransformOptimizer)
            self.optimizers={}
            self.device = settings.device
            self.transform=_f32(transform) if transform is not None else None
            self.transform_optim=OptimizableSvg.TransformOptimizer(self.transform,settings.retrieve(self.id)[0])
            self.optimizers["transform"]=self.transform_optim
            self.proc_appearance(appearance,settings.retrieve(self.id)[0])

        def tftostring(self):
            return self.transform_optim.tfToString()

        def appearanceToString(self):
            appstring=""
            for key,value in self.appearance.items():
                if key in ["fill", "stroke"]:
                    #a paint-type value
                    if value[0] == "none":
                        appstring+="{}:none;".format(key)
                    elif value[0] == "solid":
                        appstring += "{}:{};".format(key,OptimizableSvg.rgb_to_string(value[1]))
                    elif value[0] == "url":
                        appstring += "{}:url(#{});".format(key,value[1].id)
                elif key in ["opacity", "fill-opacity", "stroke-opacity", "stroke-width"]:
                    appstring+="{}:{};".format(key,float(_np(value).reshape(-1)[0]))
                elif key == "fill-rule":
                    appstring+="{}:{};".format(key,value)
                else:
                    raise ValueError("Don't know how to write appearance parameter '{}'".format(key))
            return appstring


        def write_xml_common_attrib(self,node,tfname="transform"):
            if self.transform is not None:
                node.set(tfname,self.tftostring())
            if len(self.appearance)>0:
                node.set('style',self.appearanceToString())
            if self.id is not None:
                node.set('id',self.id)


        def proc_appearance(self,appearance,optim_params):
            self.appearance=appearance
            oname=optim_params["optimizer"]
            for key, value in appearance.items():
                if key == "fill" or key == "stroke":
                    if optim_params["optimize_color"] and value[0]=="solid":
                        def upd(v, key=key):
                            self.appearance[key]=("solid", v)
                        self.optimizers[key]=OptimizableSvg.ColorOptimizer(value[1],oname,optim_params["color_lr"],on_update=upd)
                elif key == "fill-opacity" or key == "stroke-opacity" or key == "opacity":
                    if optim_params["optimize_alpha"]:
                        def upd(v, key=key):
                            self.appearance[key]=v
                        self.optimizers[key]=OptimizableSvg.ColorOptimizer(value,oname,optim_params["alpha_lr"],on_update=upd)
                elif key == "fill-rule" or key == "stroke-width":
                    pass
                else:
                    raise RuntimeError("Unrecognized appearance key '{}'".format(key))

        def prop_transform(self,intform):
            return _matmul(intform, self.transform_optim.get_transform()) if self.transform is not None else intform

        def prop_appearance(self,inappearance):
            outappearance=copy.copy(inappearance)
            for key,value in self.appearance.items():
                if key == "fill":
                    #gets replaced
                    outappearance[key]=value
                elif key == "fill-opacity":
                    #gets multiplied
                    outappearance[key] = outappearance[key]*value
                elif key == "fill-rule":
                    #gets replaced
                    outappearance[key] = value
                elif key =="opacity":
                    # gets multiplied
                    outappearance[key] = outappearance[key]*value
                elif key == "stroke":
                    # gets replaced
                    outappearance[key] = value
                elif key == "stroke-opacity":
                    # gets multiplied
                    outappearance[key] = outappearance[key]*value
                elif key =="stroke-width":
                    # gets replaced
                    outappearance[key] = value
                else:
                    raise RuntimeError("Unrecognized appearance key '{}'".format(key))
            return outappearance

        def get_params(self):
            """ Nested dict: {"optim": {name: params}, "children": {index: params}} (empty entries omitted). """
            ret={}
            optim_params={k: o.get_params() for k, o in self.optimizers.items()}
            optim_params={k: v for k, v in optim_params.items() if len(v)>0}
            if len(optim_params)>0:
                ret["optim"]=optim_params
            # non-numeric keys so mlx.utils.tree_unflatten round-trips these as dicts
            children={"c{}".format(i): c.get_params() for i, c in enumerate(self.children)}
            children={k: v for k, v in children.items() if len(v)>0}
            if len(children)>0:
                ret["children"]=children
            return ret

        def set_params(self, params):
            for k, v in params.get("optim", {}).items():
                self.optimizers[k].set_params(v)
            for k, v in params.get("children", {}).items():
                self.children[int(k[1:])].set_params(v)

        def zero_grad(self):
            pass

        def step(self, grads):
            if grads is None:
                return
            for k, v in grads.get("optim", {}).items():
                self.optimizers[k].step(v)
            for k, v in grads.get("children", {}).items():
                self.children[int(k[1:])].step(v)

        def get_type(self):
            return "Generic node"

        def is_shape(self):
            return False

        def build_scene(self,shapes,shape_groups,transform,appearance):
            raise NotImplementedError("Abstract SvgNode cannot recurse")

    class GroupNode(SvgNode):
        def __init__(self, id, transform, appearance,settings):
            super().__init__(id, transform, appearance,settings)

        def get_type(self):
            return "Group node"

        def build_scene(self,shapes,shape_groups,transform,appearance):
            outtf=self.prop_transform(transform)
            outapp=self.prop_appearance(appearance)
            for child in self.children:
                child.build_scene(shapes,shape_groups,outtf,outapp)

        def write_xml(self, parent):
            elm=etree.SubElement(parent,"g")
            self.write_xml_common_attrib(elm)

            for child in self.children:
                child.write_xml(elm)

    class RootNode(SvgNode):
        def __init__(self, id, transform, appearance,settings):
            super().__init__(id, transform, appearance,settings)

        def write_xml(self,document):
            elm=etree.Element('svg')
            self.write_xml_common_attrib(elm)
            elm.set("version","2.0")
            elm.set("width",str(document.canvas[0]))
            elm.set("height", str(document.canvas[1]))
            elm.set("xmlns","http://www.w3.org/2000/svg")
            elm.set("xmlns:xlink","http://www.w3.org/1999/xlink")
            #write definitions before we write any children
            document.write_defs(elm)

            #write the children
            for child in self.children:
                child.write_xml(elm)

            return elm

        def get_type(self):
            return "Root node"

        def build_scene(self,shapes,shape_groups,transform,appearance):
            outtf = self.prop_transform(transform)
            for child in self.children:
                child.build_scene(shapes,shape_groups,outtf,appearance)

        @staticmethod
        def get_default_appearance(device=None):
            default_appearance = {"fill": ("solid", mx.array([0., 0., 0.])),
                                  "fill-opacity": mx.array([1.]),
                                  "fill-rule": "nonzero",
                                  "opacity": mx.array([1.]),
                                  "stroke": ("none", None),
                                  "stroke-opacity": mx.array([1.]),
                                  "stroke-width": mx.array([0.])}
            return default_appearance

        @staticmethod
        def get_default_transform():
            return mx.eye(3)



    class ShapeNode(SvgNode):
        def __init__(self, id, transform, appearance,settings):
            super().__init__(id, transform, appearance,settings)

        def get_type(self):
            return "Generic shape node"

        def is_shape(self):
            return True

        def construct_paint(self,value,combined_opacity,transform):
            if value[0]   == "none":
                return None
            elif value[0] == "solid":
                return mx.concatenate([value[1].reshape(-1),combined_opacity.reshape(-1)])
            elif value[0] == "url":
                #get the gradient object from this node
                return value[1].getGrad(combined_opacity,transform)
            else:
                raise ValueError("Unknown paint value type '{}'".format(value[0]))

        def make_shape_group(self,appearance,transform,num_shapes,num_subobjects):
            fill=self.construct_paint(appearance["fill"],appearance["opacity"]*appearance["fill-opacity"],transform)
            stroke=self.construct_paint(appearance["stroke"],appearance["opacity"]*appearance["stroke-opacity"],transform)
            sg = pydiffvg.ShapeGroup(shape_ids=mx.array(list(range(num_shapes, num_shapes + num_subobjects)),dtype=mx.int32),
                                     fill_color=fill,
                                     use_even_odd_rule=appearance["fill-rule"]=="evenodd",
                                     stroke_color=stroke,
                                     shape_to_canvas=transform,
                                     id=self.id)
            return sg

    class PathNode(ShapeNode):
        def __init__(self, id, transform, appearance,settings, paths):
            super().__init__(id, transform, appearance,settings)
            self.proc_paths(paths,settings.retrieve(self.id)[0])

        def proc_paths(self,paths,optim_params):
            self.paths=paths
            if optim_params["paths"]["optimize_points"]:
                def upd(p):
                    for i, path in enumerate(self.paths):
                        path.points=p["p{}".format(i)]
                self.optimizers["points"]=OptimizableSvg.ParamGroup({"p{}".format(i): path.points for i, path in enumerate(paths)},
                                                                    optim_params["optimizer"],optim_params["paths"]["shape_lr"],on_update=upd)

        def get_type(self):
            return "Path node"

        def build_scene(self,shapes,shape_groups,transform,appearance):
            applytf=self.prop_transform(transform)
            applyapp = self.prop_appearance(appearance)
            sg=self.make_shape_group(applyapp,applytf,len(shapes),len(self.paths))
            for path in self.paths:
                disp_path=pydiffvg.Path(path.num_control_points,path.points,path.is_closed,applyapp["stroke-width"],path.id)
                shapes.append(disp_path)
            shape_groups.append(sg)

        def path_to_string(self,path):
            points=_np(path.points)
            path_string = "M {},{} ".format(points[0][0], points[0][1])
            idx = 1
            numpoints = points.shape[0]
            for type in np.array(path.num_control_points).tolist():
                toproc = type + 1
                if type == 0:
                    # add line
                    path_string += "L "
                elif type == 1:
                    # add quadric
                    path_string += "Q "
                elif type == 2:
                    # add cubic
                    path_string += "C "
                while toproc > 0:
                    path_string += "{},{} ".format(points[idx % numpoints][0],
                                                   points[idx % numpoints][1])
                    idx += 1
                    toproc -= 1
            if path.is_closed:
                path_string += "Z "

            return path_string

        def paths_string(self):
            pstr=""
            for path in self.paths:
                pstr+=self.path_to_string(path)
            return pstr

        def write_xml(self, parent):
            elm = etree.SubElement(parent, "path")
            self.write_xml_common_attrib(elm)
            elm.set("d",self.paths_string())

            for child in self.children:
                child.write_xml(elm)

    class _ArrayShapeNode(ShapeNode):
        """ Shape whose geometry is a single optimizable array stored in attribute `attr`. """
        def _setup_array(self, attr, value, settings):
            setattr(self, attr, value)
            optim_params=settings.retrieve(self.id)[0]
            #borrowing path settings for this
            if optim_params["paths"]["optimize_points"]:
                self.optimizers[attr]=OptimizableSvg.ParamGroup({attr: value},optim_params["optimizer"],optim_params["paths"]["shape_lr"],
                                                                on_update=lambda p: setattr(self, attr, p[attr]))

    class RectNode(_ArrayShapeNode):
        def __init__(self, id, transform, appearance,settings, rect):
            super().__init__(id, transform, appearance,settings)
            self._setup_array("rect", _f32(rect), settings)

        def get_type(self):
            return "Rect node"

        def build_scene(self,shapes,shape_groups,transform,appearance):
            applytf=self.prop_transform(transform)
            applyapp = self.prop_appearance(appearance)
            sg=self.make_shape_group(applyapp,applytf,len(shapes),1)
            shapes.append(pydiffvg.Rect(self.rect[0:2],self.rect[0:2]+self.rect[2:4],applyapp["stroke-width"],self.id))
            shape_groups.append(sg)

        def write_xml(self, parent):
            elm = etree.SubElement(parent, "rect")
            self.write_xml_common_attrib(elm)
            r=_np(self.rect)
            elm.set("x",str(r[0]))
            elm.set("y", str(r[1]))
            elm.set("width", str(r[2]))
            elm.set("height", str(r[3]))

            for child in self.children:
                child.write_xml(elm)

    class CircleNode(_ArrayShapeNode):
        def __init__(self, id, transform, appearance,settings, rect):
            super().__init__(id, transform, appearance,settings)
            self._setup_array("circle", _f32(rect), settings)

        def get_type(self):
            return "Circle node"

        def build_scene(self,shapes,shape_groups,transform,appearance):
            applytf=self.prop_transform(transform)
            applyapp = self.prop_appearance(appearance)
            sg=self.make_shape_group(applyapp,applytf,len(shapes),1)
            shapes.append(pydiffvg.Circle(self.circle[2],self.circle[0:2],applyapp["stroke-width"],self.id))
            shape_groups.append(sg)

        def write_xml(self, parent):
            elm = etree.SubElement(parent, "circle")
            self.write_xml_common_attrib(elm)
            c=_np(self.circle)
            elm.set("cx",str(c[0]))
            elm.set("cy", str(c[1]))
            elm.set("r", str(c[2]))

            for child in self.children:
                child.write_xml(elm)


    class EllipseNode(_ArrayShapeNode):
        def __init__(self, id, transform, appearance,settings, ellipse):
            super().__init__(id, transform, appearance,settings)
            self._setup_array("ellipse", _f32(ellipse), settings)

        def get_type(self):
            return "Ellipse node"

        def build_scene(self,shapes,shape_groups,transform,appearance):
            applytf=self.prop_transform(transform)
            applyapp = self.prop_appearance(appearance)
            sg=self.make_shape_group(applyapp,applytf,len(shapes),1)
            shapes.append(pydiffvg.Ellipse(self.ellipse[2:4],self.ellipse[0:2],applyapp["stroke-width"],self.id))
            shape_groups.append(sg)

        def write_xml(self, parent):
            elm = etree.SubElement(parent, "ellipse")
            self.write_xml_common_attrib(elm)
            e=_np(self.ellipse)
            elm.set("cx", str(e[0]))
            elm.set("cy", str(e[1]))
            elm.set("rx", str(e[2]))
            elm.set("ry", str(e[3]))

            for child in self.children:
                child.write_xml(elm)

    class PolygonNode(_ArrayShapeNode):
        def __init__(self, id, transform, appearance,settings, points):
            super().__init__(id, transform, appearance,settings)
            self._setup_array("points", _f32(points), settings)

        def get_type(self):
            return "Polygon node"

        def build_scene(self,shapes,shape_groups,transform,appearance):
            applytf=self.prop_transform(transform)
            applyapp = self.prop_appearance(appearance)
            sg=self.make_shape_group(applyapp,applytf,len(shapes),1)
            shapes.append(pydiffvg.Polygon(self.points,True,applyapp["stroke-width"],self.id))
            shape_groups.append(sg)

        def point_string(self):
            ret=""
            pts=_np(self.points)
            for i in range(pts.shape[0]):
                pt=pts[i,:]
                ret+= str(pt[0])+","+str(pt[1])+" "
            return ret

        def write_xml(self, parent):
            elm = etree.SubElement(parent, "polygon")
            self.write_xml_common_attrib(elm)
            elm.set("points",self.point_string())

            for child in self.children:
                child.write_xml(elm)

    class GradientNode(SvgNode):
        def __init__(self, id, transform,settings,begin,end,offsets,stops,href):
            super().__init__(id, transform, {},settings)
            self.optim=OptimizableSvg.GradientOptimizer(begin, end, offsets, stops, settings.retrieve(id)[0])
            self.optimizers["gradient"]=self.optim
            self.href=href

        def is_ref(self):
            return self.href is not None

        def get_type(self):
            return "Gradient node"

        def get_stops(self):
            _, _, offsets, stops=self.optim.get_vals()
            return offsets, stops

        def get_points(self):
            begin, end, _, _ =self.optim.get_vals()
            return begin, end

        def write_xml(self, parent):
            elm = etree.SubElement(parent, "linearGradient")
            self.write_xml_common_attrib(elm,tfname="gradientTransform")

            begin, end, offsets, stops = self.optim.get_vals()

            if self.href is None:
                #we have stops
                offsets_np=_np(offsets)
                stops_np=_np(stops)
                for idx, offset in enumerate(offsets_np):
                    stop=etree.SubElement(elm,"stop")
                    stop.set("offset",str(float(offset)))
                    stop.set("stop-color",OptimizableSvg.rgb_to_string(stops_np[idx,0:3]))
                    stop.set("stop-opacity",str(float(stops_np[idx,3])))
            else:
                elm.set('xlink:href', "#{}".format(self.href.id))

            if begin is not None and end is not None:
                #no stops
                b=_np(begin)
                e=_np(end)
                elm.set('x1', str(b[0]))
                elm.set('y1', str(b[1]))
                elm.set('x2', str(e[0]))
                elm.set('y2', str(e[1]))

                # magic value to make this work
                elm.set("gradientUnits", "userSpaceOnUse")

            for child in self.children:
                child.write_xml(elm)

        def getGrad(self,combined_opacity,transform):
            if self.is_ref():
                offsets, stops=self.href.get_stops()
            else:
                offsets, stops=self.get_stops()

            stops=mx.concatenate([stops[:,0:3], stops[:,3:4]*combined_opacity.reshape(1,-1)],axis=1)

            begin,end = self.get_points()

            applytf=self.prop_transform(transform)
            begin=OptimizableSvg.TransformTools.transformPoints(begin.reshape(1,2),applytf).reshape(2)
            end = OptimizableSvg.TransformTools.transformPoints(end.reshape(1,2), applytf).reshape(2)

            return pydiffvg.LinearGradient(begin, end, offsets, stops)
    #endregion

    def __init__(self, filename, settings=None,optimize_background=False, verbose=False, device=None):
        if settings is None:
            settings=SvgOptimizationSettings()
        self.settings=settings
        self.verbose=verbose
        self.device=device if device is not None else pydiffvg.get_device()
        self.settings.device=self.device

        tree = etree.parse(filename)
        root = tree.getroot()

        #in case we need global optimization
        self.optimizers={}
        self.background=mx.array([1.,1.,1.],dtype=mx.float32)

        if optimize_background:
            p=settings.retrieve("default")[0]
            self.optimizers["background"]=OptimizableSvg.ColorOptimizer(self.background,p["optimizer"],p["color_lr"],
                                                                         on_update=lambda v: setattr(self,"background",v))

        self.defs={}

        self.depth=0

        self.dirty=True
        self.scene=None

        self.parseRoot(root)

    recognised_shapes=["path","circle","rect","ellipse","polygon"]

    #region core functionality
    def build_scene(self):
        if self.dirty:
            shape_groups=[]
            shapes=[]
            self.root.build_scene(shapes,shape_groups,OptimizableSvg.RootNode.get_default_transform(),OptimizableSvg.RootNode.get_default_appearance())
            self.scene=(self.canvas[0],self.canvas[1],shapes,shape_groups)
            self.dirty=False
        return self.scene

    def _def_nodes(self):
        return [(k, v) for k, v in self.defs.items() if issubclass(v.__class__,OptimizableSvg.SvgNode)]

    def get_params(self):
        """
            Returns the trainable parameters as a nested dict (pytree) of mx.arrays:
            {"root": ..., "defs": {id: ...}, "global": {"background": ...}}.
        """
        ret={}
        rp=self.root.get_params()
        if len(rp)>0:
            ret["root"]=rp
        defs={k: v.get_params() for k, v in self._def_nodes()}
        defs={k: v for k, v in defs.items() if len(v)>0}
        if len(defs)>0:
            ret["defs"]=defs
        glob={k: o.get_params() for k, o in self.optimizers.items()}
        if len(glob)>0:
            ret["global"]=glob
        return ret

    def set_params(self, params):
        """ Writes a parameter tree (same structure as get_params()) back into the scene. """
        self.dirty=True
        if "root" in params:
            self.root.set_params(params["root"])
        for k, v in params.get("defs", {}).items():
            self.defs[k].set_params(v)
        for k, v in params.get("global", {}).items():
            self.optimizers[k].set_params(v)

    def value_and_grad(self, loss_fn):
        """
            Returns a function (*args) -> (loss, grads) where loss = loss_fn(self, *args)
            and grads has the structure of get_params(). The scene parameters are
            restored to their current values afterwards.
        """
        def wrapped(*args):
            params=self.get_params()
            def f(p):
                self.set_params(p)
                return loss_fn(self, *args)
            try:
                loss, grads = mx.value_and_grad(f)(params)
            finally:
                self.set_params(params)
            return loss, grads
        return wrapped

    def optimize_step(self, loss_fn, *args):
        """ One optimisation iteration: computes gradients of loss_fn(self, *args) and applies step(). Returns the loss. """
        loss, grads = self.value_and_grad(loss_fn)(*args)
        self.step(grads)
        return loss

    def zero_grad(self):
        # MLX gradients are functional; nothing to reset. Kept for API compatibility.
        pass

    def render(self,scale=None,seed=0):
        #render at native resolution
        scene = self.build_scene()
        scene_args = pydiffvg.RenderFunction.serialize_scene(*scene)
        render = pydiffvg.RenderFunction.apply
        out_size=(scene[0],scene[1]) if scale is None else (int(scene[0]*scale),int(scene[1]*scale))
        img = render(int(out_size[0]),  # width
                     int(out_size[1]),  # height
                     2,  # num_samples_x
                     2,  # num_samples_y
                     seed,  # seed
                     None, # background_image
                     *scene_args)
        return img

    def step(self, grads):
        """ Applies a gradient tree (structure of get_params()) with the per-parameter optimizers, then clamps. """
        self.dirty=True
        if grads is None:
            return
        if "root" in grads:
            self.root.step(grads["root"])
        for k, v in grads.get("global", {}).items():
            self.optimizers[k].step(v)
        for k, v in grads.get("defs", {}).items():
            self.defs[k].step(v)
    #endregion

    #region reporting

    def offset_str(self,s):
        return ("\t"*self.depth)+s

    def reportSkippedAttribs(self, node, non_skipped=[]):
        skipped=set([k for k in node.attrib.keys() if not OptimizableSvg.is_namespace(k)])-set(non_skipped)
        if len(skipped)>0:
            tag=OptimizableSvg.remove_namespace(node.tag) if "id" not in node.attrib else "{}#{}".format(OptimizableSvg.remove_namespace(node.tag),node.attrib["id"])
            print(self.offset_str("Warning: Skipping the following attributes of node '{}': {}".format(tag,", ".join(["'{}'".format(atr) for atr in skipped]))))

    def reportSkippedChildren(self,node,skipped):
        skipped_names=["{}#{}".format(elm.tag,elm.attrib["id"]) if "id" in elm.attrib else elm.tag for elm in skipped]
        if len(skipped)>0:
            tag = OptimizableSvg.remove_namespace(node.tag) if "id" not in node.attrib else "{}#{}".format(OptimizableSvg.remove_namespace(node.tag),
                                                                                            node.attrib["id"])
            print(self.offset_str("Warning: Skipping the following children of node '{}': {}".format(tag,", ".join(["'{}'".format(name) for name in skipped_names]))))

    #endregion

    #region parsing
    @staticmethod
    def remove_namespace(s):
        """
            {...} ... -> ...
        """
        return re.sub('{.*}', '', s)

    @staticmethod
    def is_namespace(s):
        return re.match('{.*}', s) is not None

    @staticmethod
    def parseTransform(node):
        if "transform" not in node.attrib and "gradientTransform" not in node.attrib:
            return None

        tf_string=node.attrib["transform"] if "transform" in node.attrib else node.attrib["gradientTransform"]
        tforms=tf_string.split(")")[:-1]
        mat=np.eye(3)
        for tform in tforms:
            type = tform.split("(")[0]
            args = [float(val) for val in re.split("[, ]+",tform.split("(")[1])]
            if type == "matrix":
                mat=mat @ OptimizableSvg.TransformTools.parse_matrix(args)
            elif type == "translate":
                mat = mat @ OptimizableSvg.TransformTools.parse_translate(args)
            elif type == "rotate":
                mat = mat @ OptimizableSvg.TransformTools.parse_rotate(args)
            elif type == "scale":
                mat = mat @ OptimizableSvg.TransformTools.parse_scale(args)
            elif type == "skewX":
                mat = mat @ OptimizableSvg.TransformTools.parse_skewx(args)
            elif type == "skewY":
                mat = mat @ OptimizableSvg.TransformTools.parse_skewy(args)
            else:
                raise ValueError("Unknown transform type '{}'".format(type))
        return mat

    #dictionary that defines what constant do we need to multiply different units to get the value in pixels
    #gleaned from the CSS definition
    unit_dict = {"px":1,
                 "mm":4,
                 "cm":40,
                 "in":25.4*4,
                 "pt":25.4*4/72,
                 "pc":25.4*4/6
                 }

    @staticmethod
    def parseLength(s):
        #length is a number followed possibly by a unit definition
        #we assume that default unit is the pixel (px) equal to 0.25mm
        #last two characters might be unit
        val=None
        for i in range(len(s)):
            try:
                val=float(s[:len(s)-i])
                unit=s[len(s)-i:]
                break
            except ValueError:
                continue
        if len(unit)>0 and unit not in OptimizableSvg.unit_dict:
            raise ValueError("Unknown or unsupported unit '{}' encountered while parsing".format(unit))
        if unit != "":
            val*=OptimizableSvg.unit_dict[unit]
        return val

    @staticmethod
    def parseOpacity(s):
        is_percent=s.endswith("%")
        s=s.rstrip("%")
        val=float(s)
        if is_percent:
            val=val/100
        return np.clip(val,0.,1.)

    @staticmethod
    def parse_color(s):
        """
            Color string (#rgb, #rrggbb, rgb()/rgba() with numbers or
            percentages, named colors, currentColor) to an RGB mx.array.
            Shares pydiffvg.color.parse_color_string with parse_svg.py; the
            alpha of rgba()/#rrggbbaa is dropped (opacity is handled by the
            *-opacity attributes here).
        """
        try:
            rgba = parse_color_string(s)
        except ValueError:
            raise ValueError("Color argument `{}` not supported".format(s))
        if rgba is None:
            raise ValueError("Color argument `{}` not supported".format(s))
        return mx.array(list(rgba[:3]), dtype=mx.float32)


    @staticmethod
    def rgb_to_string(val):
        byte_rgb=np.clip((np.array(val, dtype=np.float64).reshape(-1)[0:3]*255).astype(np.int64), 0, 255)
        s="#{:02x}{:02x}{:02x}".format(*[int(c) for c in byte_rgb])
        return s

    #parses a "paint" string for use in fill and stroke definitions
    @staticmethod
    def parsePaint(paintStr,defs,device):
        paintStr=paintStr.strip()
        if paintStr=="none":
            return ("none", None)
        elif paintStr.startswith("url"):
            url=paintStr.lstrip("url(").rstrip(")").strip("\'\"").lstrip("#")
            if url not in defs:
                raise ValueError("Paint-type attribute referencing an unknown object with ID '#{}'".format(url))
            return ("url",defs[url])
        else:
            try:
                return ("solid",OptimizableSvg.parse_color(paintStr))
            except ValueError:
                raise ValueError("Unrecognized paint string: '{}'".format(paintStr))

    appearance_keys=["fill","fill-opacity","fill-rule","opacity","stroke","stroke-opacity","stroke-width"]

    @staticmethod
    def parseAppearance(node, defs, device):
        ret={}
        parse_keys = OptimizableSvg.appearance_keys
        local_dict={key:value for key,value in node.attrib.items() if key in parse_keys}
        css_dict={}
        style_dict={}
        appearance_dict={}
        if "class" in node.attrib:
            cls=node.attrib["class"]
            if "."+cls in defs:
                css_string=defs["."+cls]
                css_dict={item.split(":")[0]:item.split(":")[1] for item in css_string.split(";") if len(item)>0 and item.split(":")[0] in parse_keys}
        if "style" in node.attrib:
            style_string=node.attrib["style"]
            style_dict={item.split(":")[0]:item.split(":")[1] for item in style_string.split(";") if len(item)>0 and item.split(":")[0] in parse_keys}
        appearance_dict.update(css_dict)
        appearance_dict.update(style_dict)
        appearance_dict.update(local_dict)
        for key,value in appearance_dict.items():
            if key=="fill":
                ret[key]=OptimizableSvg.parsePaint(value,defs,device)
            elif key == "fill-opacity":
                ret[key]=mx.array(float(OptimizableSvg.parseOpacity(value)),dtype=mx.float32)
            elif key == "fill-rule":
                ret[key]=value
            elif key == "opacity":
                ret[key]=mx.array(float(OptimizableSvg.parseOpacity(value)),dtype=mx.float32)
            elif key == "stroke":
                ret[key]=OptimizableSvg.parsePaint(value,defs,device)
            elif key == "stroke-opacity":
                ret[key]=mx.array(float(OptimizableSvg.parseOpacity(value)),dtype=mx.float32)
            elif key == "stroke-width":
                ret[key]=mx.array(float(OptimizableSvg.parseLength(value)),dtype=mx.float32)
            else:
                raise ValueError("Error while parsing appearance attributes: key '{}' should not be here".format(key))

        return ret

    def parseRoot(self,root):
        if self.verbose:
            print(self.offset_str("Parsing root"))
        self.depth += 1

        # get document canvas dimensions
        self.parseViewport(root)
        canvmax=np.max(self.canvas)
        self.settings.global_override(["transforms","translation_mult"],canvmax)
        id=root.attrib["id"] if "id" in root.attrib else None

        transform=OptimizableSvg.parseTransform(root)
        appearance=OptimizableSvg.parseAppearance(root,self.defs,self.device)

        version=root.attrib["version"] if "version" in root.attrib else "<unknown version>"
        if version != "2.0":
            print(self.offset_str("Warning: Version {} is not 2.0, strange things may happen".format(version)))

        self.root=OptimizableSvg.RootNode(id,transform,appearance,self.settings)

        if self.verbose:
            self.reportSkippedAttribs(root, ["width", "height", "id", "transform","version", "style"]+OptimizableSvg.appearance_keys)

        #go through the root children and parse them appropriately
        skipped=[]
        for child in root:
            if OptimizableSvg.remove_namespace(child.tag) in OptimizableSvg.recognised_shapes:
                self.parseShape(child,self.root)
            elif OptimizableSvg.remove_namespace(child.tag) == "defs":
                self.parseDefs(child)
            elif OptimizableSvg.remove_namespace(child.tag) == "style":
                self.parseStyle(child)
            elif OptimizableSvg.remove_namespace(child.tag) == "g":
                self.parseGroup(child,self.root)
            else:
                skipped.append(child)

        if self.verbose:
            self.reportSkippedChildren(root,skipped)

        self.depth-=1

    def parseShape(self,shape,parent):
        tag=OptimizableSvg.remove_namespace(shape.tag)
        if self.verbose:
            print(self.offset_str("Parsing {}#{}".format(tag,shape.attrib["id"] if "id" in shape.attrib else "<No ID>")))

        self.depth+=1
        if tag == "path":
            self.parsePath(shape,parent)
        elif tag == "circle":
            self.parseCircle(shape,parent)
        elif tag == "rect":
            self.parseRect(shape,parent)
        elif tag == "ellipse":
            self.parseEllipse(shape,parent)
        elif tag == "polygon":
            self.parsePolygon(shape,parent)
        else:
            raise ValueError("Encountered unknown shape type '{}'".format(tag))
        self.depth -= 1

    def parsePath(self,shape,parent):
        path_string=shape.attrib['d']
        name = ''
        if 'id' in shape.attrib:
            name = shape.attrib['id']
        paths = pydiffvg.from_svg_path(path_string)
        for idx, path in enumerate(paths):
            path.stroke_width = mx.array([0.],dtype=mx.float32)
            path.num_control_points=mx.array(path.num_control_points).astype(mx.int32)
            path.points=mx.array(path.points).astype(mx.float32)
            path.source_id = name
            path.id = "{}-{}".format(name,idx) if len(paths)>1 else name
        transform = OptimizableSvg.parseTransform(shape)
        appearance = OptimizableSvg.parseAppearance(shape,self.defs,self.device)
        node=OptimizableSvg.PathNode(name,transform,appearance,self.settings,paths)
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(shape, ["id","d","transform","style"]+OptimizableSvg.appearance_keys)
            self.reportSkippedChildren(shape,list(shape))

    def parseEllipse(self, shape, parent):
        cx = float(shape.attrib["cx"]) if "cx" in shape.attrib else 0.
        cy = float(shape.attrib["cy"]) if "cy" in shape.attrib else 0.
        rx = float(shape.attrib["rx"])
        ry = float(shape.attrib["ry"])
        name = ''
        if 'id' in shape.attrib:
            name = shape.attrib['id']
        transform = OptimizableSvg.parseTransform(shape)
        appearance = OptimizableSvg.parseAppearance(shape, self.defs, self.device)
        node = OptimizableSvg.EllipseNode(name, transform, appearance, self.settings, (cx, cy, rx, ry))
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(shape, ["id", "x", "y", "r", "transform",
                                              "style"] + OptimizableSvg.appearance_keys)
            self.reportSkippedChildren(shape, list(shape))

    def parsePolygon(self, shape, parent):
        points_string = shape.attrib['points']
        name = ''
        points=[]
        for point_string in points_string.split(" "):
            if len(point_string) == 0:
                continue
            coord_strings=point_string.split(",")
            assert len(coord_strings)==2
            points.append([float(coord_strings[0]),float(coord_strings[1])])
        points=mx.array(points,dtype=mx.float32)
        if 'id' in shape.attrib:
            name = shape.attrib['id']
        transform = OptimizableSvg.parseTransform(shape)
        appearance = OptimizableSvg.parseAppearance(shape, self.defs, self.device)
        node = OptimizableSvg.PolygonNode(name, transform, appearance, self.settings, points)
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(shape, ["id", "points", "transform", "style"] + OptimizableSvg.appearance_keys)
            self.reportSkippedChildren(shape, list(shape))

    def parseCircle(self,shape,parent):
        cx = float(shape.attrib["cx"]) if "cx" in shape.attrib else 0.
        cy = float(shape.attrib["cy"]) if "cy" in shape.attrib else 0.
        r = float(shape.attrib["r"])
        name = ''
        if 'id' in shape.attrib:
            name = shape.attrib['id']
        transform = OptimizableSvg.parseTransform(shape)
        appearance = OptimizableSvg.parseAppearance(shape, self.defs, self.device)
        node = OptimizableSvg.CircleNode(name, transform, appearance, self.settings, (cx, cy, r))
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(shape, ["id", "x", "y", "r", "transform",
                                              "style"] + OptimizableSvg.appearance_keys)
            self.reportSkippedChildren(shape, list(shape))

    def parseRect(self,shape,parent):
        x =      float(shape.attrib["x"]) if "x" in shape.attrib else 0.
        y =      float(shape.attrib["y"]) if "y" in shape.attrib else 0.
        width =  float(shape.attrib["width"])
        height = float(shape.attrib["height"])
        name = ''
        if 'id' in shape.attrib:
            name = shape.attrib['id']
        transform = OptimizableSvg.parseTransform(shape)
        appearance = OptimizableSvg.parseAppearance(shape, self.defs, self.device)
        node = OptimizableSvg.RectNode(name, transform, appearance, self.settings, (x,y,width,height))
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(shape, ["id", "x", "y", "width", "height", "transform", "style"] + OptimizableSvg.appearance_keys)
            self.reportSkippedChildren(shape, list(shape))

    def parseGroup(self,group,parent):
        tag = OptimizableSvg.remove_namespace(group.tag)
        id = group.attrib["id"] if "id" in group.attrib else "<No ID>"
        if self.verbose:
            print(self.offset_str("Parsing {}#{}".format(tag, id)))

        self.depth+=1

        transform=self.parseTransform(group)

        #todo process more attributes
        appearance=OptimizableSvg.parseAppearance(group,self.defs,self.device)
        node=OptimizableSvg.GroupNode(id,transform,appearance,self.settings)
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(group,["id","transform","style"]+OptimizableSvg.appearance_keys)

        skipped_children=[]
        for child in group:
            if OptimizableSvg.remove_namespace(child.tag) in OptimizableSvg.recognised_shapes:
                self.parseShape(child,node)
            elif OptimizableSvg.remove_namespace(child.tag) == "defs":
                self.parseDefs(child)
            elif OptimizableSvg.remove_namespace(child.tag) == "style":
                self.parseStyle(child)
            elif OptimizableSvg.remove_namespace(child.tag) == "g":
                self.parseGroup(child,node)
            else:
                skipped_children.append(child)

        if self.verbose:
            self.reportSkippedChildren(group,skipped_children)

        self.depth-=1

    def parseStyle(self,style_node):
        tag = OptimizableSvg.remove_namespace(style_node.tag)
        id = style_node.attrib["id"] if "id" in style_node.attrib else "<No ID>"
        if self.verbose:
            print(self.offset_str("Parsing {}#{}".format(tag, id)))

        if style_node.attrib["type"] != "text/css":
            raise ValueError("Only text/css style recognized, got {}".format(style_node.attrib["type"]))

        self.depth += 1

        # creating only a dummy node
        node = OptimizableSvg.SvgNode(id, None, {}, self.settings)

        if self.verbose:
            self.reportSkippedAttribs(def_node, ["id"])

        if len(style_node)>0:
            raise ValueError("Style node should not have children (has {})".format(len(style_node)))

        # collect CSS classes
        sheet = cssutils.parseString(style_node.text)
        for rule in sheet:
            if hasattr(rule, 'selectorText') and hasattr(rule, 'style'):
                name = rule.selectorText
                if len(name) >= 2 and name[0] == '.':
                    self.defs[name] = rule.style.getCssText().replace("\n","")
                else:
                    raise ValueError("Unrecognized CSS selector {}".format(name))
            else:
                raise ValueError("No style or selector text in CSS rule")

        if self.verbose:
            self.reportSkippedChildren(def_node, skipped_children)

        self.depth -= 1

    def parseDefs(self,def_node):
        #only linear gradients are currently supported
        tag = OptimizableSvg.remove_namespace(def_node.tag)
        id = def_node.attrib["id"] if "id" in def_node.attrib else "<No ID>"
        if self.verbose:
            print(self.offset_str("Parsing {}#{}".format(tag, id)))

        self.depth += 1


        # creating only a dummy node
        node = OptimizableSvg.SvgNode(id, None, {},self.settings)

        if self.verbose:
            self.reportSkippedAttribs(def_node, ["id"])

        skipped_children = []
        for child in def_node:
            if OptimizableSvg.remove_namespace(child.tag) == "linearGradient":
                self.parseGradient(child,node)
            elif OptimizableSvg.remove_namespace(child.tag) in OptimizableSvg.recognised_shapes:
                raise NotImplementedError("Definition/instantiation of shapes not supported")
            elif OptimizableSvg.remove_namespace(child.tag) == "defs":
                raise NotImplementedError("Definition within definition not supported")
            elif OptimizableSvg.remove_namespace(child.tag) == "g":
                raise NotImplementedError("Groups within definition not supported")
            else:
                skipped_children.append(child)

            if len(node.children)>0:
                #take this node out and enter it into defs
                self.defs[node.children[0].id]=node.children[0]
                node.children.pop()


        if self.verbose:
            self.reportSkippedChildren(def_node, skipped_children)

        self.depth -= 1

    def parseGradientStop(self,stop):
        param_dict={key:value for key,value in stop.attrib.items() if key in ["id","offset","stop-color","stop-opacity"]}
        style_dict={}
        if "style" in stop.attrib:
            style_dict={item.split(":")[0]:item.split(":")[1] for item in stop.attrib["style"].split(";") if len(item)>0}
        param_dict.update(style_dict)

        offset=OptimizableSvg.parseOpacity(param_dict["offset"])
        color=OptimizableSvg.parse_color(param_dict["stop-color"])
        opacity=OptimizableSvg.parseOpacity(param_dict["stop-opacity"]) if "stop-opacity" in param_dict else 1.

        return offset, color, opacity

    def parseGradient(self, gradient_node, parent):
        tag = OptimizableSvg.remove_namespace(gradient_node.tag)
        id = gradient_node.attrib["id"] if "id" in gradient_node.attrib else "<No ID>"
        if self.verbose:
            print(self.offset_str("Parsing {}#{}".format(tag, id)))

        self.depth += 1
        if "stop" not in [OptimizableSvg.remove_namespace(child.tag) for child in gradient_node]\
            and "href" not in [OptimizableSvg.remove_namespace(key) for key in gradient_node.attrib.keys()]:
            raise ValueError("Gradient {} has neither stops nor a href link to them".format(id))

        transform=self.parseTransform(gradient_node)
        begin=None
        end = None
        offsets=[]
        stops=[]
        href=None

        if "x1" in gradient_node.attrib or "y1" in gradient_node.attrib:
            begin=np.array([0.,0.])
            if "x1" in gradient_node.attrib:
                begin[0] = float(gradient_node.attrib["x1"])
            if "y1" in gradient_node.attrib:
                begin[1] = float(gradient_node.attrib["y1"])
            begin = mx.array(begin,dtype=mx.float32)

        if "x2" in gradient_node.attrib or "y2" in gradient_node.attrib:
            end=np.array([0.,0.])
            if "x2" in gradient_node.attrib:
                end[0] = float(gradient_node.attrib["x2"])
            if "y2" in gradient_node.attrib:
                end[1] = float(gradient_node.attrib["y2"])
            end=mx.array(end,dtype=mx.float32)

        stop_nodes=[node for node in list(gradient_node) if OptimizableSvg.remove_namespace(node.tag)=="stop"]
        if len(stop_nodes)>0:
            stop_nodes=sorted(stop_nodes,key=lambda n: float(n.attrib["offset"]))

            for stop in stop_nodes:
                offset, color, opacity = self.parseGradientStop(stop)
                offsets.append(offset)
                stops.append(np.concatenate((np.array(color,dtype=np.float64),np.array([opacity]))))

        hkey=next((value for key,value in gradient_node.attrib.items() if OptimizableSvg.remove_namespace(key)=="href"),None)
        if hkey is not None:
            href=self.defs[hkey.lstrip("#")]

        parent.children.append(OptimizableSvg.GradientNode(id,transform,self.settings,begin,end,mx.array(offsets,dtype=mx.float32) if len(offsets)>0 else None,mx.array(np.array(stops),dtype=mx.float32) if len(stops)>0 else None,href))

        self.depth -= 1

    def parseViewport(self, root):
        if "width" in root.attrib and "height" in root.attrib:
            self.canvas = np.array([int(math.ceil(float(root.attrib["width"]))), int(math.ceil(float(root.attrib["height"])))])
        elif "viewBox" in root.attrib:
            s=root.attrib["viewBox"].split(" ")
            w=s[2]
            h=s[3]
            self.canvas = np.array(
                [int(math.ceil(float(w))), int(math.ceil(float(h)))])
        else:
            raise ValueError("Size information is missing from document definition")
    #endregion

    #region writing
    def write_xml(self):
        tree=self.root.write_xml(self)
        
        return minidom.parseString(etree.tostring(tree, 'utf-8')).toprettyxml(indent="  ")

    def write_defs(self,root):
        if len(self.defs)==0:
            return

        defnode = etree.SubElement(root, 'defs')
        stylenode = etree.SubElement(root,'style')
        stylenode.set('type','text/css')
        stylenode.text=""

        defcpy=copy.copy(self.defs)
        while len(defcpy)>0:
            torem=[]
            for key,value in defcpy.items():
                if issubclass(value.__class__,OptimizableSvg.SvgNode):
                    if value.href is None or value.href not in defcpy:
                        value.write_xml(defnode)
                        torem.append(key)
                    else:
                        continue
                else:
                    #this is a string, and hence a CSS attribute
                    stylenode.text+=key+" {"+value+"}\n"
                    torem.append(key)

            for key in torem:
                del defcpy[key]
    #endregion


