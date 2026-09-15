import math
import mlx.core as mx
import numpy as np

def _norm(v):
    return mx.sqrt(mx.sum(v * v))

class GeometryLoss:
    def __init__(self, pathObj, xyalign=True, parallel=True, smooth_node=True):
        self.pathObj=pathObj
        self.pathId=pathObj.id
        self.get_segments(pathObj)
        self.xyalign=xyalign
        self.parallel=parallel
        self.smooth_node=smooth_node

        if xyalign:
            self.make_hor_ver_constraints(pathObj)

        if parallel:
            self.make_parallel_constraints(pathObj)

        if smooth_node:
            self.make_smoothness_constraints(pathObj)

    def make_smoothness_constraints(self,pathObj):
        self.smooth_nodes=[]
        for idx, node in enumerate(self.iterate_nodes()):
            sm, t0, t1=self.node_smoothness(node,pathObj)
            if abs(sm.item())<1e-2:
                self.smooth_nodes.append((node,((_norm(t0)/self.segment_approx_length(node[0],pathObj)).item(),(_norm(t1)/self.segment_approx_length(node[1],pathObj)).item())))
                #print("Node {} is smooth (smoothness {})".format(idx,sm))
            else:
                #print("Node {} is not smooth (smoothness {})".format(idx, sm))
                pass

    def node_smoothness(self,node,pathObj):
        t0=self.tangent_out(node[0],pathObj)
        t1=self.tangent_in(node[1],pathObj)
        t1rot=mx.stack((-t1[1],t1[0]))
        smoothness=mx.sum(t0*t1rot)/(_norm(t0)*_norm(t1))

        return smoothness, t0, t1

    def segment_approx_length(self,segment,pathObj):
        # Sum of the lengths of the control polygon of the segment
        # (line: 2 points, quadric: 3 points, cubic: 4 points).
        idxs=self.segList[segment[0]][segment[1]]
        length=mx.array(0.)
        for a, b in zip(idxs[:-1], idxs[1:]):
            length=length+_norm(pathObj.points[b,:]-pathObj.points[a,:])
        return length

    def tangent_in(self, segment,pathObj):
        idxs=self.segList[segment[0]][segment[1]]
        if segment[0]==0:
            #line
            return (pathObj.points[idxs[1],:]-pathObj.points[idxs[0],:])/2
        elif segment[0] in (1, 2):
            #quadric or cubic
            return pathObj.points[idxs[1],:] - pathObj.points[idxs[0],:]

        assert(False)

    def tangent_out(self, segment, pathObj):
        idxs = self.segList[segment[0]][segment[1]]
        if segment[0] == 0:
            # line
            return (pathObj.points[idxs[0],:] - pathObj.points[idxs[1],:]) / 2
        elif segment[0] == 1:
            # quadric
            return pathObj.points[idxs[1],:] - pathObj.points[idxs[2],:]
        elif segment[0] == 2:
            # cubic
            return pathObj.points[idxs[2],:] - pathObj.points[idxs[3],:]

        assert (False)

    def get_segments(self, pathObj):
        self.segments=[]
        self.lines = []
        self.quadrics=[]
        self.cubics=[]
        self.segList =(self.lines,self.quadrics,self.cubics)
        idx=0
        total_points=pathObj.points.shape[0]
        for ncp in np.array(pathObj.num_control_points).tolist():
            if ncp==0:
                self.segments.append((0,len(self.lines)))
                self.lines.append((idx, (idx + 1) % total_points))
                idx+=1
            elif ncp==1:
                self.segments.append((1, len(self.quadrics)))
                self.quadrics.append((idx, (idx + 1), (idx+2) % total_points))
                idx+=ncp+1
            elif ncp==2:
                self.segments.append((2, len(self.cubics)))
                self.cubics.append((idx, (idx + 1), (idx+2), (idx + 3) % total_points))
                idx += ncp + 1

    def iterate_nodes(self):
        for prev, next in zip([self.segments[-1]]+self.segments[:-1],self.segments):
            yield (prev, next)

    def make_hor_ver_constraints(self, pathObj):
        self.horizontals=[]
        self.verticals=[]
        for idx, line in enumerate(self.lines):
            dif=np.array(pathObj.points[line[1],:]-pathObj.points[line[0],:])

            if abs(dif[0])<1e-6:
                #is horizontal
                self.horizontals.append(idx)

            if abs(dif[1])<1e-6:
                #is vertical
                self.verticals.append(idx)

    def make_parallel_constraints(self,pathObj):
        slopes=[]
        for lidx, line in enumerate(self.lines):
            dif=np.array(pathObj.points[line[1],:]-pathObj.points[line[0],:])

            slope=math.atan2(dif[1],dif[0])
            if slope<0:
                slope+=math.pi

            minidx=-1
            for idx, s in enumerate(slopes):
                if abs(s[0]-slope)<1e-3:
                    minidx=idx
                    break

            if minidx>=0:
                slopes[minidx][1].append(lidx)
            else:
                slopes.append((slope,[lidx]))

        self.parallel_groups=[sgroup[1] for sgroup in slopes if len(sgroup[1])>1 and (not self.xyalign or (sgroup[0]>1e-3 and abs(sgroup[0]-(math.pi/2))>1e-3))]

    def make_line_diff(self,pathObj,lidx):
        line = self.lines[lidx]
        return pathObj.points[line[1], :] - pathObj.points[line[0], :]

    # MLX arrays are immutable, so each loss term returns the updated loss
    # instead of adding to it in place.
    def calc_hor_ver_loss(self,loss,pathObj):
        for lidx in self.horizontals:
            dif = self.make_line_diff(pathObj,lidx)
            loss = loss + dif[0] ** 2

        for lidx in self.verticals:
            dif = self.make_line_diff(pathObj,lidx)
            loss = loss + dif[1] ** 2
        return loss

    def calc_parallel_loss(self,loss,pathObj):
        for group in self.parallel_groups:
            diffs=[self.make_line_diff(pathObj,lidx) for lidx in group]
            difmat=mx.stack(diffs,1)
            lengths=mx.sqrt(mx.sum(difmat ** 2, axis=0))
            difmat=difmat/lengths
            rotmat=difmat[:,list(range(1,difmat.shape[1]))+[0]]
            # The z component is the only non-zero component of the cross
            # product of two vectors in the xy plane.
            cross=difmat[0]*rotmat[1]-difmat[1]*rotmat[0]
            ploss=mx.sum(cross ** 2)*mx.sum(lengths)*10
            loss = loss + ploss
        return loss

    def calc_smoothness_loss(self,loss,pathObj):
        for node, tlengths in self.smooth_nodes:
            sl,t0,t1=self.node_smoothness(node,pathObj)
            #add smoothness loss
            loss = loss + sl ** 2 * mx.sqrt(_norm(t0)) * mx.sqrt(_norm(t1))
            tl=((_norm(t0)/self.segment_approx_length(node[0],pathObj))-tlengths[0]) ** 2+((_norm(t1)/self.segment_approx_length(node[1],pathObj))-tlengths[1]) ** 2
            loss = loss + tl*10
        return loss

    def compute(self, pathObj):
        if pathObj.id != self.pathId:
            raise ValueError("Path ID {} does not match construction-time ID {}".format(pathObj.id,self.pathId))

        loss=mx.array(0.)
        if self.xyalign:
            loss=self.calc_hor_ver_loss(loss,pathObj)

        if self.parallel:
            loss=self.calc_parallel_loss(loss, pathObj)

        if self.smooth_node:
            loss=self.calc_smoothness_loss(loss,pathObj)

        #print(loss.item())

        return loss
