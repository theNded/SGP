# Dataloader

The overall target of a dataloader of SGP is to provide the loader of tuples:
```python
def __getitem__(self, idx):
    # some processing
    return data_src, data_dst, info_src, info_dst, info_pair
```
where each tuple contains
- `data_src`, `data_dst`: image for 2D perception, point cloud for 3D perception.
- `info`: additional properties, e.g. (unary) intrinsics for one image, (mutual) overlaps between two point clouds. They do not directly provide the supervision, but may serve as very weak supervision signals in geometry perception tasks.


As SGP works on pairs of data with overlaps in a scene, we assume a large dataset is consisting of various smaller scenes where overlaps exist:
```
root/
|_ scene_0/
   |_ data_0
   |_ data_1
   |_ ...
   |_ data_n
   |_ pairs.txt
   |_ metadata.txt
|_ scene_1/
|_ ...
|_ scene_m/
```
Here, the root folder contains `m` scenes. Each scene includes `n` data files. 

Assuming we have some prior knowledge of the rough overlaps between data, a scene can also provide a file storing pair associations in pair.txt:
```
data_0 data_2
data_0 data_8
data_1 data_3
...
```
Otherwise a random selection will be applied. It is strongly recommended to specify a `pair.txt` to ensure valid self supervision.

Optionally, `metadata.txt` could be provided for more info. For instance, image-wise intrinsic matrix could be provided per image, where the perception task uses the geometry model to estimate extrinsic matrix between frames:
```
data_0   fx_0 fy_0 cx_0 cy_0
data_1   fx_1 fy_1 cx_1 cy_1
...
```

So the intermediate interface will be based on scenes:
```python
def parse_scene(self, scene):
    # some processing
    return {'folder':   scene,  # str
            'fnames':   fnames, # len == n, list of str
            'pairs':    pairs,  # len == m, list of (i, j) tuple
            # Optionally metadata
            'unary_metadata' : unary_metadata,  # len == n, list of object
            'binary_metadata': mutual_metadata  # len == m, list of object
            }
```
A list of such `scene`s construct the data field, where `collect_scenes` call `parse_scene`:
```python
def __init__(self, root, scenes):
    self.root = root
    self.scenes = self.collect_scenes(root, scenes)
```
Now data length is given by the sum of `len(scene['pairs'])`, and the get item function is separated to get the scene id then the pair id, with a map array (details ommitted).
```python
def __getitem__(self, idx):
    # Use the LUT
    scene_idx = self.scene_idx_map[idx]
    pair_idx = self.pair_idx_map[idx]

    # Access actual data
    scene = self.scenes[scene_idx]
    folder = scene['folder']

    i, j = scene['pairs'][pair_idx]
    fname_src = scene['fnames'][i]
    fname_dst = scene['fnames'][j]

    print(i, j, fname_src, fname_dst)

    data_src = self.load_data(folder, fname_src)
    data_dst = self.load_data(folder, fname_dst)

    # Optional. Could be None
    metadata_src = scene['unary_metadata'][i]
    metadata_dst = scene['unary_metadata'][j]
    metadata_pair = scene['binary_metadata'][pair_idx]

    return data_src, data_dst, metadata_src, metadata_dst, metadata_pair
```

In reality, there could be minor changes in the dataset structure. For instance, there could be subfolders in a scene, and the corresponding `pairs.txt` and `metadata.txt` are renamed and outside the data folder.
```
root/
|_ scene_0/
   |_ day/
      |_ images/
         |_ data_0.jpg
         |_ data_1.jpg
         |_ ...
      |_ pairs.txt
      |_ cameras.txt
   |_ night/
      |_ images/
         |_ data_0.jpg
         |_ data_1.jpg
         |_ ...
      |_ pairs.txt
      |_ cameras.txt      
```
In this case, we only need to override `parse_scene` to re-interpret the low level structure, and override `collect_scenes` to collate various subscenes from a scene.