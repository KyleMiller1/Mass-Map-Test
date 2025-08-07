import tarfile as tar
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import zoom
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from scipy.ndimage import maximum_filter, label, generate_binary_structure
from photutils.detection import find_peaks
import healsparse as hsp
import healpy as hp
import streamlit as st

st.title('Interactive Adaptive Mass Map')

with fits.open('Om0.230_Ode0.770_w-1.000_wa0.000_si0.800/512b260/kappa/WLconv_z2.00_0803r.fits') as hdul:
    data = hdul[0].data  # Assuming image is in primary HDU
    header = hdul[0].header

lowres_data = zoom(data, 1/8)

class MultiResMap:
    def __init__(self, base_map, base_res):
        self.base_map = base_map          # 2D numpy array
        self.base_res = base_res          # arcmin or deg per pixel
        self.highres_tiles = []           # list of HighResTile

    def add_tile(self, tile):
        self.highres_tiles.append(tile)

    def plot_map(self, colorscheme):
        # Create figure
        fig = go.Figure()
        
        # Low-res background
        fig.add_trace(go.Heatmap(
            z=self.base_map,
            colorscale='Viridis',
            showscale=False,
            zmin=0, zmax=1
        ))
        i=0
        colors = ['reds_r','greens_r', 'blues_r']
        for highres_tile in self.highres_tiles:
        
            # Define axes for plotting
            x_high = np.linspace(0, self.base_map.shape[0]-1, int(3.5/highres_tile.res))
            y_high = np.linspace(0, self.base_map.shape[1]-1, int(3.5/highres_tile.res))

            if colorscheme == "Distinguish Different Resolutions":
                color = colors[i]
            elif colorscheme == "Uniform Colormap":
                color = 'Viridis'
            elif colorscheme == "Distinguish 'Clusters' vs Background":
                color = 'Plasma'
            
            # High-res overlay
            fig.add_trace(go.Heatmap(
                z=highres_tile.data,
                x=x_high,
                y=y_high,
                colorscale= color,
                showscale=False,
                zmin=0, zmax=1,
                opacity=1.0
            ))
            i+=1
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(dragmode='zoom', width=600, height=600)
        return fig
    


class HighResTile:
    def __init__(self, data, center, res, size):
        self.data = data                  # 2D numpy array
        self.center = center              # (y, x) center in base map coords or sky coords
        self.res = res                    # resolution (arcmin or deg per pixel)
        self.size = size                  # radius


Map = MultiResMap(lowres_data, 3.5/lowres_data.shape[0])

peaks = find_peaks(lowres_data, threshold = 0.18)
R = 32
x_peaks = np.array(peaks['x_peak'])
y_peaks = np.array(peaks['y_peak'])
high_combined_map = np.full(data.shape, np.nan)
int1_size = int(data.shape[0]/2)
int2_size = int(data.shape[0]/4)
int1_combined_map = np.full((int1_size,int1_size), np.nan)
int2_combined_map = np.full((int2_size,int2_size), np.nan)

for i in range(len(peaks)):
    high_x_peaks = x_peaks * data.shape[0]/lowres_data.shape[0]
    high_y_peaks = y_peaks * data.shape[0]/lowres_data.shape[0]

    int1_x_peaks = x_peaks * int1_size/lowres_data.shape[0]
    int1_y_peaks = y_peaks * int1_size/lowres_data.shape[0]

    int2_x_peaks = x_peaks * int2_size/lowres_data.shape[0]
    int2_y_peaks = y_peaks * int2_size/lowres_data.shape[0]
    
    Y, X = np.ogrid[:data.shape[0], :data.shape[1]]
    Y1, X1 = np.ogrid[:int1_size, :int1_size]
    Y2, X2 = np.ogrid[:int2_size, :int2_size]

    
    dist_from_peak = np.sqrt((X - high_x_peaks[i])**2 + (Y - high_y_peaks[i])**2)
    mask = dist_from_peak < R 
    disk = np.where(mask, data, np.nan)

    dist_from_peak1 = np.sqrt((X1 - int1_x_peaks[i])**2 + (Y1 - int1_y_peaks[i])**2)
    mask1 = dist_from_peak1 < R / (data.shape[0]/int1_size)
    disk1 = np.where(mask1, zoom(data, int1_size/data.shape[0]), np.nan)

    dist_from_peak2 = np.sqrt((X2 - int2_x_peaks[i])**2 + (Y2 - int2_y_peaks[i])**2)
    mask2 = dist_from_peak2 < R / (data.shape[0]/int2_size)
    disk2 = np.where(mask2, zoom(data, int2_size/data.shape[0]), np.nan)
    
    high_disk = np.where(disk> 0.12, disk, np.nan)
    int1_disk = np.where(disk1> 0.07, disk1, np.nan)
    int2_disk = np.where(disk2> 0.04, disk2, np.nan)
    
    high_update_mask = ~np.isnan(high_disk)
    high_combined_map[high_update_mask] = high_disk[high_update_mask]
    int1_update_mask = ~np.isnan(int1_disk)
    int1_combined_map[int1_update_mask] = int1_disk[int1_update_mask]
    int2_update_mask = ~np.isnan(int2_disk)
    int2_combined_map[int2_update_mask] = int2_disk[int2_update_mask]

Map.add_tile(HighResTile(int2_combined_map, [x_peaks[i],y_peaks[i]], 3.5/int2_size, R))
Map.add_tile(HighResTile(int1_combined_map, [x_peaks[i],y_peaks[i]], 3.5/int1_size, R))
Map.add_tile(HighResTile(high_combined_map, [x_peaks[i],y_peaks[i]], 3.5/data.shape[0], R))
    

colorscheme = st.selectbox("Visualization Options:", ["Distinguish Between Different Resolutions", "Uniform Colormap", "Distinguish Between 'Clusters' and Background"])
#high_colormap = st.selectbox("Cluster Colormap", ["Viridis", "Cividis", "Plasma", "Magma"])

fig = Map.plot_map(colorscheme)

st.plotly_chart(fig, use_container_width=False)




