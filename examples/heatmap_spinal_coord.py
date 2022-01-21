import bgheatmaps as bgh


values = {
    "2Ssp": 3.5,
    "3Sp": 0.5,
    "IB": -2,
    "5Sp": -1,
    "7Sp": -4,
    "LDCom": 2.5,
    "10Sp": 4,
    "D": -3,
    "ICl": -4,
    "6Sp": 4,
    "gr": 8,
    "vf": -1,
    "rs": -4,
    "LSp": 7,
    "dcs": -5,
}


f = bgh.heatmap(
    values,
    position=1000,
    orientation="sagittal",  # 'frontal' or 'sagittal', or 'horizontal' or a tuple (x,y,z)
    thickness=1000,
    atlas_name="allen_cord_20um",
    format="2D",
).show()
