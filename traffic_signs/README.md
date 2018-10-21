# Traffic Sign Detection

### Requirements
Run `pip install -r requirements.txt` 

### Usage

```bash
python3 traffic_sign_detection.py <dirName> <outPath> <pixelMethod> [--windowMethod=<wm>]
```
where:
- `images_dir`: path to the validation dataset ('data/train_val/val')
- `output_dir`: path where masks are saved
- `pixel_method`: color segmentation method we are using (method1: 'ihsl_1', method2: 'ihsl_2', method3: 'rgb',
 method4: 'hsv_euclidean', method5: 'hsv_ranges')
- `window_method`: window selection method we are using (method1: 'ccl_features', method2: 'ccl_template', method3: 'sw_features',
 method4: 'sw_template')

### Development

Code for pixel candidates and window candidates extraction can be found in:
- `candidate_generation_pixel.py`
- `candidate_generation_window.py`

The functions used during the development can be found in:
- `data_analisis.py`: contains helper functions to analyse the data: size, form factor and filling ratio ranges
- `dataset_split.py`: script used to split data into balanced train and validation datasets
- `color_utils.py`: contains helper functions to compute dominant colors and change of color spaces
- `morphology_utils.py`: contains morphological functions for hole filling and noise filtering + granulometry help function for stuydy of the data
- `window_evaluation.py`: contains helper functions to compute integral image and the sum of an object in a region of an integral image
- `integral_image_utils.py`: contains helper functions to evaluate whether an object in a given window is a potential candidate or not

