import seaborn as sns
import os

BASE_DIR = '/scratch/c.c21013066/images/paper'

def set_dir():
    IMAGE_DIR = os.path.join(BASE_DIR, 'ProdromalUKBB')
    return IMAGE_DIR

def plot_context():
    sns.set_context("talk", rc={"font.size":18,"axes.titlesize":18,"axes.labelsize":16,"font_scale":0.9})

class Color(object):
    def __init__(self, palette=[], healthy='blue',prodromal='orange',diseased='red'):
        self.palette = palette
        self.healthy = healthy
        self.prodromal = prodromal
        self.diseased = diseased
        
def set_colors(palette=['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00'], healthy='#377eb8',prodromal='#ff7f00',diseased='#4daf4a'):
    return Color(palette, healthy,prodromal,diseased)

def add_median_labels(ax, values,fmt='.1f',remove=0):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for i,median in enumerate(lines[4:len(lines)-remove:lines_per_box]):
        x, y = (data.mean() for data in median.get_data())
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        value = values.iloc[i]
        text = ax.text(x, y+0.01, value, ha='center', va='center',
                       fontweight='bold', color='k',fontsize=12,rotation=90,bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
