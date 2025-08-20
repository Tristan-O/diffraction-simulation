import numpy as np
from pint import UnitRegistry

ureg = UnitRegistry()
def convert_unit(value:float|np.ndarray, from_unit:str, to_unit:str):
    return (value * ureg(from_unit)).to(to_unit).magnitude  # Compute conversion factor. Magnitude here preserves the sign of the value.
def get_unit_conversion(from_unit:str, to_unit:str):
    return convert_unit(value=1, from_unit=from_unit, to_unit=to_unit)
def pretty_unit(unit:str):
    return f'{ureg(unit).u:~P}'


class KeyAwareDefaultDict(dict):
    def __init__(self, factory, **kwargs):
        self.factory = factory
        self.update(**kwargs)
    def __missing__(self,key:str):
        self[key] = self.factory(key)
        return self[key]


from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from PIL import Image
import win32clipboard, io

def fig2clipboard(fig:Figure, dpi:int=200):
    '''Convert a pyplot Figure into a png in the clipboard. 
    Does not work for animations.'''
    img_buf = io.BytesIO()
    output = io.BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0.0, dpi=dpi )

    # saving GIF
    Image.open(img_buf).convert('RGB').save(output, 'BMP')
    data = output.getvalue()[14:]
    output.close()
    img_buf.close()

    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
    win32clipboard.CloseClipboard()

def figures2gif(figures, path, duration=500, dpi=200, close=True):
    """
    Convert a list of matplotlib figures to an animated GIF.
    
    Parameters:
    - figures: List of matplotlib.figure.Figure objects
    - path: Output filename for the GIF
    - duration: Duration of each frame in milliseconds
    - dpi
    - close
    """
    if not path.endswith('.gif'):
        path = path + '.gif'

    img_buffers = []

    bboxes = [fig.get_tightbbox(fig.canvas.get_renderer()) for fig in figures]
    max_bbox = Bbox.union(bboxes)

    for fig in figures:
        #preparing to save GIF
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches=max_bbox, pad_inches=0.0, dpi=dpi )
        img_buffers.append( img_buf )
        if close:
            plt.close(fig) # this might not work?

    # saving GIF
    frames = [Image.open(buffer).convert('RGB') for buffer in img_buffers]
    # If having issues with images not being the same size, this can be uncommented to resize images
    min_x, min_y = np.inf,np.inf
    for f in frames:
        px_x, px_y = f.size
        min_x = int(np.min((px_x, min_x)))
        min_y = int(np.min((px_y, min_y)))
    frames = [f.resize((min_x, min_y)) for f in frames]
    frames[0].save( path, format='GIF', append_images=frames[1:],
                    save_all=True, duration=duration, loop=0) #duration is ms per frame?
    for img_buf in img_buffers:
        img_buf.close()