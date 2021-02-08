import plotly.express as px
from PIL import Image
from io import BytesIO
import plotly.graph_objects as go
import numpy as np

def line_plot(title, xaxis_title, yaxis_title, lines):
    fig = go.Figure()
    for line in lines:
        fig.add_trace(go.Scatter(**line))
    
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return Image.open(BytesIO(fig.to_image()))

def table_plot(header, cells, height=500, width=300, align='left'):
    fig = go.Figure(data=[go.Table(
        header=dict(values=header,
        line_color='darkslategray',
        fill_color='lightskyblue',
        align=align),
        cells=dict(values=cells,
        line_color='darkslategray',
        fill_color='lightcyan',
        align=align)
    )])

    fig.update_layout(width=width, height=height)
    return Image.open(BytesIO(fig.to_image()))
