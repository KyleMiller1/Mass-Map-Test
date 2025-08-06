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

st.title('Interactive Adaptive Mass Maps')

path = r"/mnt/c/Users/kylem/Downloads/convergence_CMB_mnv0.00000_om0.30000_As2.1000.tar"
file = tar.open(path, 'r')
map1 = file.extract('Maps11000/WLconv_z1100.00_3158r.fits')
with fits.open('Maps11000/WLconv_z1100.00_3158r.fits') as hdul:
    data = hdul[0].data  # Assuming image is in primary HDU
    header = hdul[0].header

lowres_data = zoom(data, 0.5)

class MultiResMap:
    def __init__(self, base_map, base_res):
        self.base_map = base_map          # 2D numpy array
        self.base_res = base_res          # arcmin or deg per pixel
        self.highres_tiles = []           # list of HighResTile

    def add_tile(self, tile):
        self.highres_tiles.append(tile)

    def plot_map(self, lowcolor, highcolor):
        # Create figure
        fig = go.Figure()
        
        # Low-res background
        fig.add_trace(go.Heatmap(
            z=self.base_map,
            colorscale=lowcolor,
            showscale=False,
            zmin=0, zmax=1,
            hoverinfo='skip'
        ))
    
        for highres_tile in self.highres_tiles:
        
            # Define axes for plotting
            x_high = np.linspace(0, self.base_map.shape[0]-1, int(3.5/highres_tile.res))
            y_high = np.linspace(0, self.base_map.shape[1]-1, int(3.5/highres_tile.res))
        
            # x_crop = np.arange(x0_crops[i], x0_crops[i] + crop_size)
            # y_crop = np.arange(y0_crops[i], y0_crops[i] + crop_size)
        
            # High-res overlay
            fig.add_trace(go.Heatmap(
                z=highres_tile.data,
                x=x_high,
                y=y_high,
                colorscale=highcolor,
                showscale=False,
                zmin=0, zmax=1,
                opacity=1.0,
                hoverinfo='skip'
            ))
        
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

peaks = find_peaks(lowres_data, threshold = 0.4)
R = 16
x_peaks = np.array(peaks['x_peak'])*2
y_peaks = np.array(peaks['y_peak'])*2 #factor for high res coords

for i in range(len(peaks)):
    Y, X = np.ogrid[:data.shape[0], :data.shape[1]]
    dist_from_peak = np.sqrt((X - x_peaks[i])**2 + (Y - y_peaks[i])**2)
    mask = dist_from_peak < 16
    Map.add_tile(HighResTile(np.where(mask, data, np.nan), [x_peaks[i],y_peaks[i]], 3.5/data.shape[0], R))

low_colormap = st.selectbox("Background Colormap", ["Viridis", "Cividis", "Plasma", "Magma"])
high_colormap = st.selectbox("Cluster Colormap", ["Viridis", "Cividis", "Plasma", "Magma"])

fig = Map.plot_map(low_colormap, high_colormap)

st.plotly_chart(fig, use_container_width=False)




