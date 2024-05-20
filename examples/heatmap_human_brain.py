import brainglobe_heatmap as bgh

values = dict(  # scalar values for each region
    SFG=1,
    PrCG=2,
    Ca=4,
    Pu=10,
)

scene = bgh.heatmap(
    values,
    position=(100000, 100000, 100000),
    thickness=1000,
    atlas_name="allen_human_500um",
    format="2D",
).show()
