# Run on custom data

Before preparing your own data, you should checkout the [dataset module](../lib/datasets/scannet.py) carefully.

Overall, you need to place your data with the following structure and create a corresponding config file.
```
manhattan_sdf
├───data
|   ├───$scene_name
|   |   ├───intrinsic.txt
|   |   ├───images
|   |   |   ├───0.png
|   |   |   ├───1.png
|   |   |   └───...
|   |   ├───pose
|   |   |   ├───0.txt
|   |   |   ├───1.txt
|   |   |   └───...
|   |   ├───depth_colmap
|   |   |   ├───0.npy
|   |   |   ├───1.npy
|   |   |   └───...
|   |   └───semantic_deeplab
|   |       ├───0.png
|   |       ├───1.png
|   |       └───...
|   └───...
├───configs
|   ├───$scene_name.yaml
|   └───...
└───...
```
## Images
You should place RGB images in `data/$scene_name/images` folder, note that the filenames can be arbitrary but you need to make sure that they are consistent with files in other folders under `data/$scene_name`.

## Intrinsic parameters
Save the `4x4` intrinsic matrix in `data/$scene_name/intrinsic.txt`.

## Camera poses
You can solve camera poses with COLMAP or other tools you like. Then you need to normalize the camera poses and modify some configs to ensure that:
- The scene is inside [bounding radius](../configs/scannet/0050.yaml#L14), note that bounding radius should be lower than π so that positional encoding can work well.
- The center of the scene is near the origin and the [geometric initialization radius](../configs/scannet/0050.yaml#L17) is appropriate (surface of initialized sphere and target geometry should not be too far).
- Make sure that the sampling range ([near](../configs/scannet/0050.yaml#L41), [far](../configs/scannet/0050.yaml#L42)) can cover the whole scene.

To achieve these, the simplest way is to normalize the camera poses to be inside a unit sphere, which is similar to the implementation of [VolSDF](https://github.com/lioryariv/volsdf/blob/main/DATA_CONVENTION.md) and [neurecon](https://github.com/ventusff/neurecon/blob/972e810ec252cfd16f630b1de6d2802d1b8de59a/dataio/DTU.py#L67). Then you can set the geometric initialization radius as 1.0 and bounding radius as 2.0 (note that indoor scenes are scanned from inside, which is different from object dataset such as [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36), so you need to set the geometric initialization radius and bounding radius larger than the range of camera poses).

This can work well if the images are captured by walking around the scene (namely, the camera trajectory is relatively close to the boundary of the scene), since the scale of the scene can be regarded as slightly larger than the range of camera poses. If it cannot be guaranteed, you need to rescale the camera poses to be inside a smaller sphere or adjust the hyperparameters heuristically. A more general way is to first run sparse reconstruction and define a region of interest manually, please refer to [NeuS](https://github.com/Totoro97/NeuS/tree/main/preprocess_custom_data).

Save the normalized poses as `4x4` matrices in `txt` format under `data/$scene_name/pose` folder. Remember to save the scale and offset used to normalize [here](../configs/scannet/0050.yaml#L50) so that you can transform to original coordinate if you want to extract mesh and compare with ground truth mesh.

## COLMAP depth maps
You need to run sparse and dense reconstruction of COLMAP first. Please refer to this [instruction](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses) if you want to use known camera poses.

After dense reconstruction, you can obtain depth prediction for each view. However, the depth predictions can be noisy, so we recommend you to run fusion to filter out most noises. Since original COLMAP does not have fusion mask for each view, you need to compile and run [this customized version](https://github.com/B1ueber2y/colmap/tree/c84269d693246d8294307cc32f851813f18b6a2d), which is a submodule of [NerfingMVS](https://github.com/weiyithu/NerfingMVS).

We provide a Python script [here](./run_colmap/) for reference, which includes camera poses normalization and COLMAP depth maps generation.

## Semantic predictions
You need to run 2D semantic segmentations to generate semantic predictions from images. We provide our trained model and inference code [here](./semantic_segmentation/).
