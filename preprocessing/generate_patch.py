# -*- coding: utf-8 -*
#!/usr/bin/env python
#
# deepzoom_tile - Convert whole-slide images to Deep Zoom format
#
# Copyright (c) 2010-2015 Carnegie Mellon University
#
# This library is free software; you can redistribute it and/or modify it
# under the terms of version 2.1 of the GNU Lesser General Public License
# as published by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""An example program to generate a Deep Zoom directory tree from a slide."""
from __future__ import print_function
import json
from multiprocessing import Process, JoinableQueue
from optparse import OptionParser
import os
import re
import shutil
import sys
from unicodedata import normalize
from PIL import Image
from PIL import ImageFile
import numpy as np
import glob
import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator

VIEWER_SLIDE_NAME = 'slide'
tumor_threshold = 0.3
missslide = []

def OTSU_enhance(img_gray, th_begin=0, th_end=256, th_step=1):
    assert img_gray.ndim == 2, "must input a gary_img"
 
    max_g = 0
    suitable_th = 0
    for threshold in range(th_begin, th_end, th_step):
        bin_img = img_gray > threshold
        bin_img_inv = img_gray <= threshold
        fore_pix = np.sum(bin_img)
        back_pix = np.sum(bin_img_inv)
        if 0 == fore_pix:
            break
        if 0 == back_pix:
            continue
 
        w0 = float(fore_pix) / img_gray.size
        u0 = float(np.sum(img_gray * bin_img)) / fore_pix
        w1 = float(back_pix) / img_gray.size
        u1 = float(np.sum(img_gray * bin_img_inv)) / back_pix
        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        if g > max_g:
            max_g = g
            suitable_th = threshold
    suitable_th = suitable_th * 1.05
    if suitable_th < 210:
        suitable_th = 210
    return suitable_th


class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds,
                quality):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath 
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._slide = None

    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            associated, level, address, outfile = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            #try:
            tile = dz.get_tile(level, address) 
            tile.save(outfile, quality=self._quality)
            self._queue.task_done()
                

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated]) 
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz, outpath, basenameJPG, format, queue, slide, Imgextension, Magnification, tile_size, associated=None):
        self._dz = dz
        self._outpath = outpath
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._slide = slide
        self._ImgExtension = Imgextension
        self._Mag = Magnification
        self._basenameJPG = basenameJPG
        self._tile_size = tile_size

    def run(self):
        self._write_tiles()

    def _write_tiles(self):
        Magnification = self._Mag
        
        
        Factors = self._slide.level_downsamples
        print('svs.level_downsamples:', Factors)
        print('svs.level_count:', self._slide.level_count)
        
        try:
            Objective = float(self._slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        except:
            print(self._basenameJPG + '- No Obj information found.')
            print(self._ImgExtension)
            if ('jpg' in self._ImgExtension) | ('dcm' in self._ImgExtension) | ('tif' in self._ImgExtension):
                Objective = 1.
                Magnification = Objective
                print('input is jpg, will be tiled as such with %f' % Objective)
            else:
                return

        print('Objective:', Objective)
        Available = tuple(Objective / x for x in Factors)
        print('Available:', Available)
        Mismatch = tuple(x - Magnification for x in Available)
        AbsMismatch = tuple(abs(x) for x in Mismatch)
        if len(AbsMismatch) < 1:
            print(self._basenameJPG, '- Objective field empty!')
            return

        for level in range(self._dz.level_count-1, -1, -1):
            ThisMag = Available[0]/pow(2, self._dz.level_count-(level+1))

            if self._Mag > 0:
                if ThisMag != self._Mag:
                    continue
            
            print('level:', level)
            print('ThisMag', ThisMag)
            
            tiledir = os.path.join("%s" % self._outpath, self._basenameJPG)
            if not os.path.exists(tiledir):
                os.makedirs(tiledir)
            cols, rows = self._dz.level_tiles[level]
            print('cols, rows:', cols, rows)

            downsamples=self._slide.level_downsamples
            print(downsamples)
            [w,h]=self._slide.level_dimensions[0]
            print(w,h)
            if len(downsamples) == 2:
                size1=int(w*(downsamples[0]/downsamples[1]))
                size2=int(h*(downsamples[0]/downsamples[1]))
                upsample = downsamples[1] / downsamples[0]
                region=np.array(self._slide.read_region((0,0),1,(size1,size2)))
                print(region.shape)
            else:
                size1=int(w*(downsamples[0]/downsamples[2]))
                size2=int(h*(downsamples[0]/downsamples[2]))
                upsample = downsamples[2] / downsamples[0]
                region=np.array(self._slide.read_region((0,0),2,(size1,size2)))
                print(region.shape)
            print(np.sum(region))
            thumbnail = Image.fromarray(region)
            gray = thumbnail.convert('L')
            gray1 = np.array(gray)
            otsu = OTSU_enhance(gray1)
            print(otsu)
            gray = gray.point(lambda x: 1 if x < otsu and x > 100 else 0, 'F')
            
            label_array = np.array(gray)
            print(np.sum(label_array))
            print('working2')
            
            (max_row, max_col) = label_array.shape
            print('labelimage size: ', max_row, max_col)

            label_multiple = round(self._tile_size / upsample * (Objective / ThisMag))
            print('labelimage multiple: ', label_multiple)

            print(max_col / label_multiple)
            print(max_row / label_multiple)


            for row in range(rows):
                for col in range(cols):
                    if (row+1)*label_multiple > max_row or (col+1)*label_multiple > max_col:
                        continue
                    cur_label_array_rows = label_array[int(row * label_multiple) : int(min(max_row, (row + 1) * label_multiple)), :]
                    cur_label_array = cur_label_array_rows[: , int(col * label_multiple) : int(min(max_col, (col + 1) * label_multiple))]
                    avglabel = np.average(cur_label_array)
                    #print(avglabel)
                    patch_label = 1 if avglabel > tumor_threshold else 0
                    if patch_label == 0:
                        continue 

                    tilename = os.path.join(tiledir, '%d_%d.%s' % (
                                    col, row, self._format))
                    if not os.path.exists(tilename):
                        self._queue.put((self._associated, level, (col, row),
                                    tilename))
                    self._tile_done()

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count 
        print("Tiling %s: wrote %d/%d tiles" % (self._associated or 'slide', count, total), end='\r', file=sys.stderr)
        if count == total:
            print(file=sys.stderr)

class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, slidepath, outpath, basenameJPG, format, tile_size, overlap,
                limit_bounds, quality, workers, with_viewer, ImgExtension, Mag):
        if with_viewer:
            # Check extra dependency before doing a bunch of work
            import jinja2
        self._slide = open_slide(slidepath)
        self._outpath = outpath
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._with_viewer = with_viewer
        self._dzi_data = {}
        self._ImgExtension = ImgExtension
        self._Mag = Mag
        self._basenameJPG = basenameJPG

        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                        limit_bounds, quality).start()
        print('working')
        

    def run(self):
        self._run_image()
        self._shutdown()

    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        if associated is None:
            image = self._slide
            if self._with_viewer:
                outpath = os.path.join(self._outpath, VIEWER_SLIDE_NAME)
            else:
                outpath = self._outpath
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            outpath = os.path.join(self._outpath, self._slugify(associated))
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)
        print('dz.level_tiles', dz.level_tiles)
        print('dz.level_dimensions', dz.level_dimensions)

        tiler = DeepZoomImageTiler(dz, outpath, basenameJPG, self._format, self._queue, self._slide, self._ImgExtension, self._Mag, self._tile_size, associated=None)
        tiler.run()

    def _url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self._slugify(associated)
        return '%s.dzi' % base

    def _write_html(self):
        import jinja2
        env = jinja2.Environment(loader=jinja2.PackageLoader(__name__),
                    autoescape=True)
        template = env.get_template('slide-multipane.html')
        associated_urls = dict((n, self._url_for(n))
                    for n in self._slide.associated_images)
        try:
            mpp_x = self._slide.properties[openslide.PROPERTY_NAME_MPP_X]
            mpp_y = self._slide.properties[openslide.PROPERTY_NAME_MPP_Y]
            mpp = (float(mpp_x) + float(mpp_y)) / 2
        except (KeyError, ValueError):
            mpp = 0
        data = template.render(slide_url=self._url_for(None),
                    slide_mpp=mpp,
                    associated=associated_urls,
                    properties=self._slide.properties,
                    dzi_data=json.dumps(self._dzi_data))
        with open(os.path.join(self._basename, 'index.html'), 'w') as fh:
            fh.write(data)

    def _write_static(self):
        basesrc = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'static')
        basedst = os.path.join(self._basename, 'static')
        self._copydir(basesrc, basedst)
        self._copydir(os.path.join(basesrc, 'images'),
                    os.path.join(basedst, 'images'))

    def _copydir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                shutil.copy(srcpath, os.path.join(dest, name))

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()


if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] <slide>')
    parser.add_option('-B', '--ignore-bounds', dest='limit_bounds',
                default=True, action='store_false',
                help='display entire scan area')
    parser.add_option('-e', '--overlap', metavar='PIXELS', dest='overlap',
                type='int', default=1,
                help='overlap of adjacent tiles [1]')
    parser.add_option('-f', '--format', metavar='{jpeg|png}', dest='format',
                default='jpeg',
                help='image format for tiles [jpeg]')
    parser.add_option('-j', '--jobs', metavar='COUNT', dest='workers',
                type='int', default=4,
                help='number of worker processes to start [4]')
    parser.add_option('-o', '--output', metavar='NAME', dest='outpath',
                help='name of output folder')
    parser.add_option('-Q', '--quality', metavar='QUALITY', dest='quality',
                type='int', default=90,
                help='JPEG compression quality [90]')
    parser.add_option('-r', '--viewer', dest='with_viewer',
                action='store_true',
                help='generate directory tree with HTML viewer')
    parser.add_option('-s', '--size', metavar='PIXELS', dest='tile_size',
                type='int', default=254,
                help='tile size [254]')
    parser.add_option('-l', '--labelimage', metavar='NAME', 
        dest='labelimage_path', help='base name of label image file')
    parser.add_option('-M', '--Mag', metavar='PIXELS', 
        dest='Mag', type='float', default=-1, help='Magnification at which tiling should be done (-1 of all)')



    (opts, args) = parser.parse_args()

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    try:
        slidepath = args[0]
    except IndexError:
        parser.error('Missing slide argument')
    for classes in ['glioma', 'lymphoma']:
            
        slides = glob.glob(os.path.join(slidepath, classes, '*.svs'))

        ImgExtension = 'svs'
        for num, slide in enumerate(slides):
            basenameJPG = os.path.basename(slide).split('.')[0]
            casename = basenameJPG.split('_')[1][0:6]
            outpath = os.path.join(opts.outpath, classes, casename)
            try:
                DeepZoomStaticTiler(slide, outpath, basenameJPG, opts.format,
                            opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality,
                            opts.workers, opts.with_viewer, ImgExtension, opts.Mag).run()
            except Exception as e:
                if basenameJPG not in missslide:
                    missslide.append(basenameJPG)
            print(missslide)

