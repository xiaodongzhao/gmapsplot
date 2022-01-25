#! /usr/bin/env python
from __future__ import print_function, division
import os
import sys
try:
    from StringIO import StringIO as MemoryIO
except ModuleNotFoundError:
    from io import BytesIO as MemoryIO
import cv2
import googlemaps
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger('gmapsplot')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class GMapsPlot(object):
    def __init__(self, api_key, center, zoom=21, scale=1, size_pixel=None, size_meter=(200, 100), image_dir=None):
        self.check_map_params(center, zoom, scale)

        # some constants from Google Maps
        self.TILE_SIZE = 256
        # https://groups.google.com/forum/#!msg/google-maps-js-api-v3/hDRO4oHVSeM/osOYQYXg2oUJ
        self.EARTH_RADIUS = 6378137
        self.MAX_SIZE = 640
        self.MIN_SIZE = 180
        self.LOGO_SIZE = 25
        self.DEFAULT_ZOOM = 21
        self.DEFAULT_SCALE = 1

        # folder to save map images
        if image_dir is None:
            cur_dir = os.getcwd()
            self.image_dir = os.path.join(cur_dir, "image_dir")
            if not os.path.exists(self.image_dir):
                os.mkdir(self.image_dir)
        elif not os.path.exists(self.image_dir):
            raise ValueError("folder %s could not be found" % self.image_dir)
        else:
            pass

        self.api_key = api_key
        self.center = list(center)
        self.zoom = self.DEFAULT_ZOOM if zoom is None else zoom
        self.scale = self.DEFAULT_SCALE if scale is None else scale
        if size_pixel is not None:
            self.size = size_pixel
        elif size_meter is not None:
            res = self.get_map_resolution(self.center[0], self.zoom, self.scale)
            self.size = [size_meter[0]/res, size_meter[1]/res]
        else:
            raise ValueError("Please specify image size either in pixel or in meters")
        self.image_data = None
        self.gmaps_client = googlemaps.Client(key=self.api_key)

    def check_map_params(self, center, zoom, scale):
        assert(len(center) == 2)
        assert(-90 <= center[0] <= 90 and -180 <= center[1] <= 180)
        assert(scale == 1 or scale == 2)
        assert(isinstance(zoom, int) and zoom > 0)

    def get_map_resolution(self, latitude, zoom=None, scale=None):
        """Google map resolution (m/pixel)
        """
        if zoom is None:
            zoom = self.DEFAULT_ZOOM
        if scale is None:
            scale = self.DEFAULT_SCALE
        # meter per pixel at zoom 0, which is whole earth. at zoom 0, there are 256 pixel, as per
        # https://developers.google.com/maps/documentation/javascript/coordinates
        meter_per_pixel_zoom_0 = self.EARTH_RADIUS * 2 * np.pi / 256.0
        meter_per_pixel_zoom = meter_per_pixel_zoom_0 / np.power(2, zoom)  # meter per pixel at zoom level, which is decreased by 2^zoom
        radius_ratio = np.cos(latitude * np.pi / 180)  # radius ratio at latitude,
        return meter_per_pixel_zoom * radius_ratio / scale

    def lat_lng_to_google_world_coordinate(self, lat, lng):
        """
        convert latitude, longitude to [Google Web Mercator](https://en.wikipedia.org/wiki/Web_Mercator_projection)
        based on https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
        """
        siny = np.sin(np.deg2rad(lat))
        # Truncating to 0.9999 effectively limits latitude to 89.189. This is
        # about a third of a tile past the edge of the world tile.
        siny = min(max(siny, -0.9999), 0.9999)
        x = self.TILE_SIZE * (0.5 + lng / 360.0)
        y = self.TILE_SIZE * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
        return x, y

    def lat_lng_from_google_world_coordinate(self, x, y):
        lng = 360 * (x / self.TILE_SIZE - 0.5)
        lat = np.rad2deg(2 * (np.arctan(np.exp(np.pi - 2 * np.pi * y / self.TILE_SIZE)) - np.pi / 4.0))
        lat = max(min(lat, 89.189), -89.189)
        return lat, lng

    def lat_lng_to_google_pixel_coordinate(self, lat, lng, zoom, scale):
        zoom_ratio = 2**zoom
        x, y = self.lat_lng_to_google_world_coordinate(lat, lng)
        return np.floor(x * scale * zoom_ratio), np.floor(y * scale * zoom_ratio)

    def lat_lng_from_google_pixel_coordinate(self, x, y, zoom, scale):
        zoom_ratio = 2**zoom
        lat, lng = self.lat_lng_from_google_world_coordinate(x / scale / zoom_ratio, y / scale / zoom_ratio)
        return lat, lng

    def get_map_image_name(self, center, zoom, scale, size):
        map_file = os.path.join(self.image_dir, "lat%.6f-lng%.6f_zoom%d_size%dx%d_scale%d.png" % (center[0], center[1], zoom, size[0], size[1], scale))
        return map_file

    def save_map_image(self, image, map_file):
        cv2.imwrite(map_file, image)

    def load_map_image(self, map_file):
        if os.path.exists(map_file):
            img = cv2.imread(map_file, cv2.IMREAD_COLOR)
            return img
        else:
            return None

    def download_single_map_image(self, center, zoom, scale, size):
        self.check_map_params(center, zoom, scale)
        if size[0] > self.MAX_SIZE or size[1] > self.MAX_SIZE:
            size = (min(size[0], self.MAX_SIZE), min(size[1], self.MAX_SIZE))
            logger.warning("Max map size allowed is %dx%d" % (self.MAX_SIZE, self.MAX_SIZE))
        if size[0] < self.MIN_SIZE or size[1] < self.MIN_SIZE:
            size = (max(size[0], self.MIN_SIZE), max(size[1], self.MIN_SIZE))
            logger.warning("Min map size allowed is %dx%d" % (self.MIN_SIZE, self.MIN_SIZE))

        map_file = self.get_map_image_name(center, zoom, scale, size)
        image = self.load_map_image(map_file)
        if image is None:
            logger.info("Downloading map image centered at (%f,%f)" % (center[0], center[1]))
            response = self.gmaps_client.static_map(size=(int(size[0]), int(size[1])), zoom=int(zoom), center=center, maptype="satellite", format="png", scale=int(scale))
            f = MemoryIO()
            for chunk in response:
                if chunk:
                    f.write(chunk)
            arr = np.frombuffer(f.getvalue(), np.uint8)
            f.close()
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            self.save_map_image(image, map_file)
        else:
            logger.info("%s already downloaded" % map_file)
        return image

    def download_map_image(self, center, zoom,  scale, size, keep_logo=True):
        logger.info("map image center=%s, size=%s, scale=%s, zoom=%s" % (center, size, scale, zoom))

        # load downloaded image first
        map_file = self.get_map_image_name(center, zoom, scale, size)
        image = self.load_map_image(map_file)
        if image is not None:
            logger.info("%s already downloaded" % map_file)
            return image

        # download multi images to form a large image
        size_x, size_y = size
        center_x, center_y = self.lat_lng_to_google_pixel_coordinate(center[0], center[1], zoom, scale)
        tl_x, tl_y = center_x - size_x * scale / 2, center_y - size_y * scale / 2  # top left
        tiles = []
        y = tl_y  # start from top
        while True:  # by row
            row_tiles = []
            remain_y = tl_y + size_y - y  # remaining pixels to download
            keep_logo_row = keep_logo
            if remain_y <= 0:
                break
            elif remain_y < self.MIN_SIZE:  # too small
                sub_size_y = self.MIN_SIZE
            elif remain_y <= self.MAX_SIZE:
                sub_size_y = remain_y
            else:  # too big, split again
                sub_size_y = self.MAX_SIZE
            if self.MIN_SIZE < remain_y <= (self.MAX_SIZE+self.MIN_SIZE):
                keep_logo_row = True  # keep logo on the last full image
            y = y + sub_size_y / 2
            x = tl_x  # start from left boarder
            while True:  # by column
                remain_x = tl_x + size_x - x  # remaining pixels to download
                if remain_x <= 0:
                    break
                elif remain_x < self.MIN_SIZE:  # too small
                    sub_size_x = self.MIN_SIZE
                elif remain_x <= self.MAX_SIZE:
                    sub_size_x = remain_x
                else:  # too big, split again
                    sub_size_x = self.MAX_SIZE
                x = x + sub_size_x / 2
                sub_center = self.lat_lng_from_google_pixel_coordinate(x, y, zoom, scale)
                image = self.download_single_map_image(sub_center, zoom,  scale, (sub_size_x, sub_size_y))
                # remove extra downloaded size
                max_shape_x = remain_x if remain_x < self.MIN_SIZE else image.shape[1]
                max_shape_y = remain_y if remain_y < self.MIN_SIZE else image.shape[0]
                image = image[:int(max_shape_y), :int(max_shape_x), :]
                # remove logo
                if not keep_logo_row and image.shape[0] > self.LOGO_SIZE:
                    dy = image.shape[0]/2 - self.LOGO_SIZE
                    image = image[:-self.LOGO_SIZE, :, :]
                else:
                    dy = image.shape[0]/2
                row_tiles.append(image)
                # move x to the right border
                x += image.shape[1] / 2
            y += dy  # bottom border
            tiles.append(np.hstack(row_tiles))
        image = np.vstack(tiles).astype(np.uint8)
        self.save_map_image(image, map_file)
        return image

    def download(self,):
        self.image_data = self.download_map_image(self.center, self.zoom,  self.scale, self.size, keep_logo=True)

    def plot(self,):
        fig, ax = plt.subplots()
        if self.image_data is not None:
            img = np.flip(self.image_data, axis=0)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resolution = self.get_map_resolution(self.center[0], zoom=self.zoom)
            extent = np.array([-0.5, img_rgb.shape[1]-0.5, -0.5, img_rgb.shape[0]-0.5])
            extent = extent*resolution
            origin = (extent[0]+extent[1])/2, (extent[2]+extent[3])/2
            extent[0:2] = extent[0:2]-origin[0]
            extent[2:4] = extent[2:4]-origin[1]
            ax.imshow(img_rgb, origin="lower", extent=extent)
        return fig, ax
