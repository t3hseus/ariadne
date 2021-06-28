import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
import gin
import pandas as pd
import os
import numpy as np


@gin.configurable
class TrackDataSet(Dataset):


    def __init__(self, data_root, x_max, x_min, y_max, y_min, x_res=256, y_res=256):
        self.root = data_root
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
        self.x_res = x_res
        self.y_res = y_res
        self.x_pixel_size = self.get_x_pixel_size()
        self.y_pixel_size = self.get_y_pixel_size()
        self.data = [x for x in os.listdir(data_root) if x.endswith(".csv")]
        self.names = [' ', 'event',	'z',	'station', 	'track', 	'px',	'py',	'pz',	'vz',	'y',	'x',	'vx',	'vy']


    def get_x_pixel_size(self):
        return (self.x_max-self.x_min)/self.x_res


    def get_y_pixel_size(self):
        return (self.y_max-self.y_min)/self.y_res


    def xcoord2pix(self, x):
        cord = ((x-self.x_min)/self.x_pixel_size).astype(int)
        return cord


    def ycoord2pix(self, y):
        cord = ((y-self.y_min)/self.y_pixel_size).astype(int)
        return cord


    def filter_data(self, data_to_filter, m=2.):
        d = np.abs(data_to_filter - np.median(data_to_filter))
        mdev = np.median(d)
        s = d / (mdev if mdev else 1.)
        return data_to_filter[s < m]


    def get_vertex(self, event):
        vx = event['vx'].values
        vy = event['vy'].values
        vz = event['vz'].values
        # vx = self.filter_data(vx[vx.nonzero()])
        # vy = self.filter_data(vy[vy.nonzero()])
        # vz = self.filter_data(vz[vz.nonzero()])
        return np.array([vx.mean(), vy.mean(), vz.mean()], dtype=np.float32)  

    
    def event_img(self, event):

        X = np.zeros((7, self.y_res, self.x_res), dtype=np.float32)
        phi_mtrx = 0
        z_mtrx = 1
        r_mtrx = 2
        image_mtrx = 4
        x, y = np.mgrid[0:(self.y_res - 1) * self.y_pixel_size :256j, 0: (self.x_res - 1) * self.y_pixel_size : 256j]

        # station_i = None
        X[phi_mtrx, :, :] = x + y
        X[z_mtrx, :, :] = x + y
       
        for i in range(3):
          # station_i_prev = station_i
          station_i = event.loc[event.station==i]        
          ypix=self.ycoord2pix(station_i.y)
          xpix=self.xcoord2pix(station_i.x)
          # phi = station_i.x 
          X[phi_mtrx, ypix, xpix] = station_i.x.values
          res = np.where(X[phi_mtrx, :, :] == 0)
          listOfCoordinates= list(zip(res[0], res[1]))
          for j in listOfCoordinates:
            X[phi_mtrx, j[0], j[1]] = self.x_pixel_size * j[0] + self.x_pixel_size * j[1] 

          # z = station_i.y
          X[z_mtrx, ypix, xpix] = station_i.y.values
          
          res = np.where(X[z_mtrx, :, :] == 0)
          listOfCoordinates= list(zip(res[0], res[1]))

          for j in listOfCoordinates:
            X[z_mtrx, j[0], j[1]] = self.y_pixel_size * j[0] + self.y_pixel_size * j[1] 

          # r_i = station_i.z
          X[r_mtrx + i, ypix, xpix] = station_i.z.values
          
          if (i > 0):
            X[r_mtrx + (i - 1), ypix, xpix] = station_i.z.values
            prev_matrix = np.zeros((256, 256),  dtype=np.float32)
            station_prev = event.loc[event.station==(i-1)]
            ypix_prev = self.ycoord2pix(station_prev.y)
            xpix_prev = self.xcoord2pix(station_prev.x)
            prev_matrix[ypix_prev, xpix_prev] = station_prev.z.values
            X[r_mtrx + (i - 1), :, :] = X[r_mtrx + (i - 1), :, :] - prev_matrix[:, :]

          if (i > 0):
            # station_i_prev = event.loc[event.station==0] 
            X[r_mtrx + (i - 1), ypix, xpix] = station_i.z.values
            # X[r_mtrx + i, ypix, xpix] = station_i_prev.z.values
            # X[r_mtrx + (i - 1), ypix, xpix] = X[r_mtrx + (i - 1), ypix, xpix] - X[r_mtrx + i, ypix, xpix]
          
          X[image_mtrx + i, ypix, xpix] = 1
          
        return X

    def __getitem__(self, item):
        event_path = self.root + '/' + self.data[item]
        self.event = pd.read_csv(event_path, engine='python')   
        return {"inputs": self.event_img(self.event)},  self.get_vertex(self.event)


    def __len__(self):
      return len(self.data)


# @gin.configurable
# class VertexLootDataset(Dataset):
#     """ 
#     """

#     def __init__(self, data_root, x_max, x_min, y_max, y_min, x_res=256, y_res=256):
#         self.root = data_root
#         self.x_max = x_max
#         self.x_min = x_min
#         self.y_max = y_max
#         self.y_min = y_min
#         self.x_res = x_res
#         self.y_res = y_res
#         self.x_pixel_size = self.get_x_pixel_size()
#         self.y_pixel_size = self.get_y_pixel_size()
#         self.data = [x for x in os.listdir(data_root) if x.endswith(".csv")]
#         self.names = [' ', 'event',	'z',	'station', 	'track', 	'px',	'py',	'pz',	'vz',	'y',	'x',	'vx',	'vy']


#     def get_x_pixel_size(self):
#         return (self.x_max-self.x_min)/self.x_res


#     def get_y_pixel_size(self):
#         return (self.y_max-self.y_min)/self.y_res


#     def xcoord2pix(self, x):
#         cord = ((x-self.x_min)/self.x_pixel_size).astype(int)
#         return cord


#     def ycoord2pix(self, y):     
#         cord = ((y-self.y_min)/self.y_pixel_size).astype(int)
#         return cord


#     def filter_data(self, data_to_filter, m=2.):
#         d = np.abs(data_to_filter - np.median(data_to_filter))
#         mdev = np.median(d)
#         s = d / (mdev if mdev else 1.)   
#         return data_to_filter[s < m]


#     def get_vertex(self, event):
#         # vx = event['vx'].values
#         # vy = event['vy'].values
#         # vz = event['vz'].values
        
#         # vx = self.filter_data(vx[vx.nonzero()])
#         # vy = self.filter_data(vy[vy.nonzero()])
#         # vz = self.filter_data(vz[vz.nonzero()])
        
#         vx = self.filter_data(event['vx'].values)
#         vy = self.filter_data(event['vy'].values)
#         vz = self.filter_data(event['vz'].values)
#         return np.array([vx.mean(), vy.mean(), vz.mean()], dtype=np.float32)  


#     def get_event_df(self):
#       return self.event


#     def event_img(self, event):
#         X = np.zeros((8, self.y_res, self.x_res), dtype=np.float32)
#         phi_mtrx = 0
#         z_mtrx = 1
#         r_mtrx = 2
#         image_mtrx = 5
#         for i in range(3):   
#             station_i = event.loc[event.station==i] 
#             ypix=self.ycoord2pix(station_i.y)
#             xpix=self.xcoord2pix(station_i.x) 
#             # phi = station_i.x   
#             X[phi_mtrx, ypix, xpix] = station_i.x.values
#             # z = station_i.y
#             X[z_mtrx, ypix, xpix] = station_i.y.values
#             # r_i = station_i.z
#             X[r_mtrx + i, ypix, xpix] = station_i.z.values          
#             X[image_mtrx + i, ypix, xpix] = 1    
#         return X


#     def __getitem__(self, item):
#         event_path = self.root + '/' + self.data[item]
#         self.event = pd.read_csv(event_path, engine='python')        
#         return {"inputs": self.event_img(self.event)},  self.get_vertex(self.event)


#     def __len__(self):
#         return len(self.data)