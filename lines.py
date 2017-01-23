import numpy as np
from numpy import random
import scipy as sp
from scipy import ndimage
from scipy import spatial
from scipy import interpolate
import sklearn
from sklearn import neighbors

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections

from numba import jit

import time

@jit
def sample(mycdf,N):
    inds=np.array([0]*N)
    vals=np.random.rand(N)
    for k in range(N):
        #inds[k]=np.where(mycdf>=vals[k])[0][0]

        ## numba jitting doesn't work with this
        #inds[k]=(mycdf>=vals[k]).nonzero()[0][0]

        for i in range(len(mycdf)):
            inds[k]=i
            if mycdf[i]>=vals[k]: 
                break
        #inds=[np.where(imgcdf>=a)[0][0] for a in vals]
    return inds

def sample_from_image(x,N=1000):
    # treat image as a probability density.
    # 1. compute CDF as cumsum of the pixels (normalize)

    xorig=np.ravel(x)
    siz=x.shape
    yy,xx=np.mgrid[0:siz[0],0:siz[1]]

    x=np.ravel(x)
    x=np.double(x)
    x=x/np.sum(x)
    imgcdf=np.cumsum(x)

    # 3. choose first pixel whose value is larger
    # TODO: write a function and use @jit from that library
    #inds=[np.where(imgcdf>=a)[0][0] for a in vals]

    inds=sample(imgcdf,N)

    # 4. convert indices to 3-D points (x,y,intensity)
    ys=np.ravel(yy)
    xs=np.ravel(xx)
    pts=np.array([(xs[i],ys[i]) for i in inds])

    return pts,np.array([xorig[inds]]).T

def get_edges(pts,K):
    #TODO: note that scipy.spatial does this also
    tree = sklearn.neighbors.KDTree(pts)
    # K+1 because the first is trivial
    neighbors=tree.query(pts,k=K+1,return_distance=False)[:,1:]

    edges=[]
    for k in range(len(pts)):
        nbs=pts[neighbors[k]]
        edges+=[np.vstack((pts[k][:2],a[:2])) for a in nbs]

    edges=np.array(edges)
    
    return edges

def interp_image(X,pts):
    f=sp.interpolate.interp2d(range(X.shape[1]),range(X.shape[0]),X)
    return f(pts[:,1],pts[:,0])

#@jit(nopython=True)
def get_centroid(pts):
    # from https://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon
    cx=0.
    cy=0.
    A=0.
    if(pts.shape[0]==1): return pts

    pts=np.vstack((pts,pts[0]))
    for k in range(pts.shape[0]-1):
        aincr=(pts[k,0]*pts[k+1,1]-pts[k+1,0]*pts[k,1])
        cx += (pts[k,0]+pts[k+1,0])*aincr
        cy += (pts[k,1]+pts[k+1,1])*aincr
        A += aincr

    A=0.5*A
    centroid=1/(6*A)*np.array([cx,cy])
    return centroid


def stipple_points(x):
    c=x
    converged=False
    maxiters=50

    niter=0
    # TODO: clamp voronoi vertices to image bounding box
    while not converged and niter < maxiters:
        niter+=1
        vd=sp.spatial.Voronoi(c)
        verts=vd.vertices
        regions=vd.regions

        # compute centroids
        c=[]
        for k in range(len(regions)):
            if len(regions[k])>0:
                r=np.array(regions[k])
                #r=r[r>=0] #-1 codes for no point

                # here we just scrap the edge cells
                if np.any(r<0): continue
                #c+=[np.mean(verts[r], axis=0)]
                v=1.0*verts[r]
                c+=[get_centroid(v)]

        #stipples=np.array(map(lambda x:np.mean(x,axis=1,keepdims=True), c))

    stipples=np.array(c)
    return stipples

def get_angle(edges):
    # calculate 2-quadrant angle
    vectors=np.squeeze(np.diff(edges,axis=1))
    angle=np.angle(vectors[:,0]+1j*vectors[:,1])

    return angle*180/np.pi

def plot_edges(edges,angle=None):
    plt.figure()
    brightness=0.012
    c=[(1,1,1,brightness)]*len(edges)
    bgc=(0.1,0.1,0.1)

    angletol=10 #degrees
    if angle is not None:
        # filter edges to given angles
        edgeangles=np.array([get_angle(edges)]).T
        if np.isscalar(angle):
            angle=[angle]
        angle=np.array([angle])

        edges=edges[np.any(np.abs(edgeangles-angle)<angletol, axis=1)]

    lc=matplotlib.collections.LineCollection(edges,color=c)
    plt.gca().axis('image')
    plt.gca().set_axis_bgcolor(bgc)
    plt.gca().add_collection(lc)
    plt.gca().autoscale()
    plt.gca().invert_yaxis()

def run(fname='lion.jpg',N=1000,K=5,intscale=500,stipple=False):
    # load image and make grayscale
    X=sp.ndimage.imread(fname)
    #X=0.33*X[:,:,0]+0.33*X[:,:,1]+0.33*X[:,:,2] #TODO: proper rgb2gray
    if len(X.shape)>2:
        X=X[:,:,1]


    # contrast stretching
    X=np.maximum(0,X-0.05*np.max(X))
    X=np.minimum(np.max(X),X+0.01*np.max(X))
    X=np.double(X)
    X=X-np.min(X)
    X=X/np.max(X)

    # if there are more bright pixels than dark ones,
    # invert the colors
    if np.median(X)>0.5: X=1.-X
    # TODO: make intscale relative to image scale. and possibly
    # interpolate the image to a fixed size!
    X=X*intscale

    # TODO: maybe apply e.g. bilateral filter for smoothing

    # generate random points based on intensity
    # TODO: return x-y coordinates and intensities in separate
    # vectors, and combine them only when needed.
    t=time.time()
    pts,ints=sample_from_image(X,N) 
    t=time.time()-t
    print "Sampled in %d seconds\n"%(t)

    newpts=np.hstack((pts,ints))

    # weighted voronoi stippling (Secord 2002, NPAR)
    #if stipple: pts=stipple_points(pts)
    if stipple: 
        t=time.time()
        newpts=stipple_points(newpts)
        t=time.time()-t
        print "Stippled in %d seconds\n"%(t)

    # interpolate from image
    #ints=interp_image(X,pts)

    # draw lines between points 
    # use a 3d mapping of the points (x,y,intensity).
    # connect each point to its K nearest neighbors _in this space_ 
    #newpts=np.hstack((pts,ints))
    edges=get_edges(newpts,K)

    return (edges,pts)
