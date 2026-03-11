import brainglobe_heatmap as bgh

data_dict = {
    "VISpm": 0.0,
    "VISp": 0.14285714285714285,
    "VISl": 0.2857142857142857,
    "VISli": 0.42857142857142855,
    "VISal": 0.5714285714285714,
    "VISrl": 0.7142857142857142,
    "SSp-bfd": 0.8571428571428571,
    "VISam": 1.0,
}

scene = bgh.Heatmap(
    data_dict,
    position=[7500, 8000, 8500, 9000, 9500, 9500, 9500, 9500],
    orientation="frontal",
    thickness=10,
    format="2D",
    cmap="Reds",
    vmin=0,
    vmax=1,
    label_regions=False,
).plot()
