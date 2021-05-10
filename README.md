# Image_stitching_Python

## Usage

```bash
$ python ImageStitching.py --path=./Image/file3.txt --win_size=60
    <--path = string> 
    <--win_size = int> 
    <--max_iters = int> # ransac iterators
    <--epsilon = float> # ransac epsilon 
    <--draw = bool> # draw corner
    <--fromMid = bool> # Stitching from mid  
 # example
$ python ImageStitching.py --path=./Image/file3.txt --win_size=60

The record image will output in source path. 
There are four file. 
1. leftpano.jpg: left side of image stitching
2. rightpano.jpg: right side of image stitching
3. matches.jpg: feature matching of two side pano
4. pano.jpg: final result image
```
