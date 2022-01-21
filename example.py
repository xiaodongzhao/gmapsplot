#! /usr/bin/env python
import matplotlib.pyplot as plt
from gmapsplot import GMapsPlot
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, "gmapsplot"))

center = (52.382882, -1.565237) # this will be the x,y origin
zoom = 21
scale = 1
gmaps = GMapsPlot(api_key="GOOGLE_MAP_API_KEY", center=center, zoom=18, scale=1, size_meter=(200, 200))
gmaps.download()
fig, ax = gmaps.plot()
ax.scatter([-56, -36, 10, 30, 100], [-42, -30, 10, 30, 100], s=20)
plt.show()
