import bokeh
from bokeh.plotting import figure as bokeh_figure
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models.widgets import NumberFormatter
import bokeh.io
import numpy as np
from PIL import Image


def to_rgba(im):
    im = Image.fromarray(im)
    im = np.asarray(im.convert('RGBA'))
    im = np.flipud(im)
    return im


def plot_image(im, axes=True, tools='', im_size=None, figure=None):
    if np.asarray(im).ndim == 2:
        gray = True
    else:
        im = to_rgba(im)
        gray = False

    if im_size is None:
        h, w = im.shape[:2]
    else:
        h, w = im_size
    source = bokeh.models.sources.ColumnDataSource(dict(rgba=[im]))
    f = image_figure('rgba', source, im_size=(h, w), axes=axes, tools=tools, gray=gray, figure=figure)
    return f, source


def make_image_figure(im_size=(240, 320), axes=True):
    w, h = im_size
    f = bokeh_figure(x_range=(0, w), y_range=(0, h),
                     plot_width=w, plot_height=h, tools='',
                     tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
    f.toolbar.logo = None
    if not axes:
        f.xaxis[0].visible = False
        f.yaxis[0].visible = False
    return f


def image_figure(key, source, im_size=(240, 320), axes=True, tools='',
                 gray=False, figure=None):
    h, w = im_size
    if figure is None:
        f = bokeh_figure(x_range=(0, w), y_range=(0, h),
                         plot_width=w, plot_height=h, tools=tools,
                         tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
    else:
        f = figure

    f.toolbar.logo = None
    if not axes:
        f.xaxis[0].visible = False
        f.yaxis[0].visible = False
    # f.image_rgba(key, x=0, y=0, dw=w, dh=h, source=source)
    if gray:
        f.image(key, x=0, y=0, dw=w, dh=h, source=source)
    else:
        f.image_rgba(key, x=0, y=0, dw=w, dh=h, source=source)
    return f


def convert_df(df):
    columns = []
    for column in df.columns:
        if df.dtypes[column].kind == 'f':
            formatter =  NumberFormatter(format='0.000')
        else:
            formatter = None
        table_col = TableColumn(field=column, title=column, formatter=formatter)
        columns.append(table_col)
    data_table = DataTable(columns=columns, source=ColumnDataSource(df), height=200)
    return data_table
