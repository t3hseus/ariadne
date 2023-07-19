#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
import math
import numpy as np
from tqdm import tqdm


def ExtrapToR(pt, charge, theta, phi, z0, Rc):
    pi = 3.14156
    deg = 180/pi
    B = 0.8 # magnetic field [T}
    
    pz = pt / math. tan(theta  ) *charge

    phit = phi - pi/2
    R = pt/0.29/B # mm
    k0 = R/math.tan(theta)
    x0 = R*math.cos(phit)
    y0 = R*math.sin(phit)

    if R < Rc/2 : # no intersection
       return (0, 0 ,0)

    R = charge*R; # both polarities
    alpha = 2*math.asin(Rc/2/R)
  
    if (alpha > pi):
        return (0,0,0); # algorithm doesn't work for spinning tracks

    extphi = phi - alpha/2
    if extphi > 2*pi:
        extphi = extphi - 2*pi

    if extphi < 0:
        extphi = extphi + 2*pi

    x = Rc*math.cos(extphi)
    y = Rc*math.sin(extphi)

    radial = np.array( [ x-x0*charge , y-y0*charge ] )

    rotation_matrix = np.array([[0, -1], [1, 0]])
    tangent = np.dot(rotation_matrix,radial)

    tangent /= np.sqrt( np.sum( tangent**2 ) )#pt
    tangent *= -pt*charge
    px,py = tangent[0],tangent[1]

    z = z0 + k0*alpha
    return (x,y,z,px,py,pz)



if __name__ == '__main__':
    nevents = int(sys.argv[1])
    save_path = sys.argv[2]
    #track_coords_all = []
    eff = 0.98  # detector efficiency
    
    radii = np.linspace(270, 580, 35) # mm

    with open(f'{save_path}//{nevents}_{eff}.tsv', 'w') as f:
      for evt in tqdm(range(0, nevents)):
          pi = 3.14156
          
          vtxx = random.gauss(0, 10)
          vtxy = random.gauss(0, 10)
          vtxz = random.uniform(-300, 300) # mm
          ntrk = int(random.uniform(1,10))
          for trk in range(0, ntrk):
              
              pt = random.uniform(100,1000) # MeV/c
              phi = random.uniform(0, 2*pi)
              theta = math.acos(random.uniform(-1,1))
              
              charge = 0

              while charge == 0:
                  charge = random.randint(-1,1)
             
              station = 1
              for R in radii:

                  x,y,z,px,py,pz = ExtrapToR(pt, charge, theta, phi, vtxz, R)

                  if (x,y,z) == (0,0,0):
                      continue
                  if z>=2386 or z <=-2386 :
                      continue
                  z = z + random.gauss(0,0.1)
                  phit = math.atan2(x,y)
                  delta = random.gauss(0,0.1)
                  x = x + delta*math.sin(phit)
                  y = y - delta*math.cos(phit)

                  if random.uniform(0,1) > eff:
                      continue

                  f.write("%d\t%f\t%f\t%f\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\n" % (evt,x,y,z,station,trk,px,py,pz,vtxx,vtxy,vtxz) )

                  station = station + 1

    # add noise hits
          nhit = int(random.uniform(10, 500)) # up to 100 noise hits
          for ihit in range(0, nhit):
              sta = int(random.uniform(0,35))
              R = radii[sta]
              phi = random.uniform(0, 2*pi)
              z = random.uniform(-2386, 2386)
              x = R*math.cos(phi)
              y = R*math.sin(phi)
              f.write("%d\t%f\t%f\t%f\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\n" % (evt,x,y,z,sta,-1,0,0,0,0,0,0) )


    f.close()

