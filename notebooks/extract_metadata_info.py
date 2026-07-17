from datetime import timedelta
 
import astropy.units as u
from astropy.time import Time

from sunpy.net import Fido, attrs as a
from sunpy.coordinates import frames

from astropy.coordinates import SkyCoord
from sunpy.coordinates import SphericalScreen

import matplotlib.pyplot as plt
import sunpy.map 
import numpy as np

import re

def extract_solar_regions(smap):

    coords = sunpy.map.all_coordinates_from_map(smap)
    Tx = coords.Tx
    Ty = coords.Ty

    on_disk = sunpy.map.coordinate_is_on_solar_disk(coords)

    data = smap.data.astype(float)

    masks = {
        'N':  (Ty.value >= 0) & on_disk,
        'S':  (Ty.value <  0) & on_disk,
        'NE': (Tx.value <= 0) & (Ty.value >= 0) & on_disk,
        'SE': (Tx.value <= 0) & (Ty.value <  0) & on_disk,
        'NW': (Tx.value >  0) & (Ty.value >= 0) & on_disk,
        'SW': (Tx.value >  0) & (Ty.value <  0) & on_disk,
        'W': (Tx.value < 0) & on_disk,
        'E': (Tx.value >= 0) & on_disk,
    }

    results = {}
    
    for name, mask in masks.items():
        region_data = data[mask]
        region_Tx = Tx[mask]
        region_Ty = Ty[mask]
    
        # Geometric center: plain average position of pixels in the region
        center_Tx = np.nanmean(region_Tx)
        center_Ty = np.nanmean(region_Ty)
        geometric_center = SkyCoord(center_Tx, center_Ty, frame=smap.coordinate_frame)

        # Bounding extent of the region
        tx_min, tx_max = np.nanmin(region_Tx), np.nanmax(region_Tx)
        ty_min, ty_max = np.nanmin(region_Ty), np.nanmax(region_Ty)

        bottom_left = SkyCoord(tx_min, ty_min, frame=smap.coordinate_frame)
        top_right = SkyCoord(tx_max, ty_max, frame=smap.coordinate_frame)
            
        # Intensity-weighted centroid (shift weights so they're non-negative)
        weights = region_data - np.nanmin(region_data)
        w_sum = np.nansum(weights)
        if w_sum > 0:
            weighted_Tx = np.nansum(region_Tx * weights) / w_sum
            weighted_Ty = np.nansum(region_Ty * weights) / w_sum
        else:
            weighted_Tx, weighted_Ty = center_Tx, center_Ty
        weighted_center = SkyCoord(weighted_Tx, weighted_Ty, frame=smap.coordinate_frame)

        variance = np.nanvar(region_data)
        std = np.nanstd(region_data)
    
        results[name] = {
            'geometric_center': geometric_center,
            'weighted_center': weighted_center,
            'variance': variance,
            'bottom_left': bottom_left,
            'top_right': top_right,
            'tx_min': tx_min, 'tx_max': tx_max,
            'ty_min': ty_min, 'ty_max': ty_max,
            'std': std,
            'n_pixels': int(mask.sum()),
        }
    
        #print(f"{name:>2}: geo_center=({center_Tx.value:7.1f}\", {center_Ty.value:7.1f}\")  "
        #    f"weighted_center=({weighted_Tx.value if hasattr(weighted_Tx,'value') else weighted_Tx:7.1f}\", "
        #    f"{weighted_Ty.value if hasattr(weighted_Ty,'value') else weighted_Ty:7.1f}\")  "
        #    f"variance={variance:10.2f}  std={std:8.2f}  n_px={mask.sum()}")
    
    return results

def extract_events(smap, t_start, t_end, event_type):

    result = Fido.search(
        a.Time(t_start, t_end),
        a.hek.EventType(event_type),
        #a.hek.OBS.Observatory == "GOES",
        #a.hek.OBS.Instrument == "AIA",
        #a.hek.OBS.ChannelID == "171"
    )

    with SphericalScreen(smap.observer_coordinate, only_off_disk=True):
        event_coords = [
            event["event_coord"].transform_to(smap.coordinate_frame) 
            for event in result["hek"]
        ]

    return event_coords

