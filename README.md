## Introduction

This repo refactors and extends the repo of [4Dsegment2.0](https://github.com/UK-Digital-Heart-Project/4Dsegment2.0)

CMRSegment is a Cardiac Magnetic Resonance Imaging processing pipeline, consisted of 6 modules, which are in sequence a preprocessor, 
a segmentor, a refiner, an extractor, a coregister, and a motion tracker. 


## Installation

To install CMRSegment, do 
```
pip setup.py install
```

It depends on MIRTK. After you have installed MIRTK, you need to add mirtk to pythonpath

```
export PYTHONPATH=$PYTHONPATH:CMAKE_INSTALL_PREFIX/Lib/Python
```

To add mirtk to your virtual environment

Uses `add2virtualenv` from `virtualenvwrapper`: https://virtualenvwrapper.readthedocs.io/en/latest/command_ref.html#add2virtualenv

```
pip install virutalenvwrapper (virtualenvwrapper-win for windows)
add2virtualenv CMAKE_INSTALL_PREFIX/Lib/Python
```

Then you can directly import mirtk in your python script.

A prebuild docker image is available to use: 

```
docker pull lisurui6/cmr-sgement:latest
docker run -ti lisurui6/cmr-segment:latest
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/mirtk/python
```

```
docker build -t lisurui6/cmr-segment:latest .
docker push docker.io/lisurui6/cmr-segment:latest
```

```
eval $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa
```

```
export PYTHONPATH=$PYTHONPATH:/mirtk/lib/python
rf -rf PH_atlas
apt install python3.7
pip2 uninstall tensorflow
python3.7 get-pip.py
pip3.7 install -r requirements.txt
pip3.7 install tensorflow
pip3.7 install torch torchvision
```


## Modules
### 1. Preprocessor
This module splits the CMR sequence, and then resamples and enlarges each phase. 
It assumes that input subjects are in the following file structure:
```
/path/
    subject1/
        LVSA.nii.gz
    subject2/
        LVSA.nii.gz
    ...
```
It then generates output in the following structures:

```
/path/
    subject1/
        contrasted_LVSA.nii.gz          ->  Contrasted CMR sequence image
        lvsa_ED.nii.gz                  ->  ED phase image
        lvsa_ES.nii.gz                  ->  ES phase image
        lvsa_SR_ED.nii.gz               ->  Enlarged ED image
        lvsa_SR_ES.nii.gz               ->  Enlarged ES image
        lvsa_SR_ED_resampled.nii.gz     ->  Resampled ED image
        lvsa_SR_ESs_resampled.nii.gz    ->  Resampled ES image
        gray_phases/
            lvsa_00.nii.gz              -> Phase 0 image
            lvsa_01.nii.gz              -> Phase 1 image
            ...
        resampled/
            lvsa_0.nii.gz               ->  Resampled phase 0 image
            lvsa_1.nii.gz               ->  Resampled phase 1 image
            ...
        enlarged/
            lvsa_0.nii.gz               ->  Enlarged phase 0 image
            lvsa_1.nii.gz               ->  Enlarged phase 1 image
            ...
    subject2/
        ...
```

To run only the preprocessor, use one of the following commands:
```
cmrtk-pipeline -o /output-dir/ --data-dir /input-dir/

python -m CMRSegment.preprocessor -o /output-dir/ --data-dir /input-dir

cmrtk-preprocessor -o /output-dir/ --data-dir /input-dir/
```

### 2. Segmentor
This module uses a trained 3D convolutional neural network (UNet) to segment ED/ES enlarged images.
The code to train the network is in [`experiments/fcn_3d/`](experiments/fcn_3d)

It takes input 
```
/path/
    subject1/
        lvsa_SR_ED.nii.gz               ->  Enlarged ED image
        lvsa_SR_ES.nii.gz               ->  Enlarged ES image
        enlarged/
            lvsa_0.nii.gz               ->  Enlarged phase 0 image
            lvsa_1.nii.gz               ->  Enlarged phase 1 image
            ...
    subject2/
        ...
```
and generates output
```
/path/
    subject1/
        seg_lvsa_SR_ED.nii.gz           ->  ED segmentation
        seg_lvsa_SR_ES.nii.gz           ->  ES segmentation
        segs/
            lvsa_0.nii.gz               ->  Phase 0 segmentation
            lvsa_1.nii.gz               ->  Phase 1 segmentation
            ...
        4D_rview/
            4Dimg.nii.gz                ->  4D image
            4Dseg.nii.gz                ->  4D segmentation
    subject2/
        ...
```
To run only the segmentor, use one of the following commands:
```
cmrtk-pipeline -o /output-dir/ --data-dir /input-dir/ --segment --model-path /path-to-model --torch --segment-cine

python -m CMRSegment.segmentor -o /output-dir/ --data-dir /input-dir --segment --model-path /path-to-model --torch --segment-cine

cmrtk-torch-segmentor -o /output-dir/ --data-dir /input-dir/ --segment --model-path /path-to-model --torch --segment-cine
```
For RBH project, a pretrained model is located at `/cardiac/DL_segmentation/IHD_project/DL_models/inference_model.pt`


### 3. Refiner
Refiner uses registration to refine segmentations generated by the segmentor. For each segmentation, it first finds top
similar atlases, then registers the segmentation with each atlas, and finally fuses the labels of all transformed atlases 
and outputs as the refined segmentation.

It takes input from
```
/path/
    subject1/
        seg_lvsa_SR_ED.nii.gz           ->  ED segmentation
        seg_lvsa_SR_ES.nii.gz           ->  ES segmentation
    subject2/
        ...
```
and generates output
```
/path/
    subject1/
        refine/
            tmp/                                        ->  Temporary files
            seg_lvsa_SR_ED.nii_refined.nii.gz           ->  Refined ED segmentation
            seg_lvsa_SR_ES.nii_refined.nii.gz           ->  Refined ES segmentation
    subject2/
        ...
```

To run only the refiner, use one of the following commands:
```
cmrtk-pipeline -o /output-dir/ --data-dir /input-dir/ --refine --csv-path /path-to-csv --n-top 7 --n-atlas 500

python -m CMRSegment.refiner -o /output-dir/ --data-dir /input-dir --refine --csv-path /path-to-csv --n-top 7 --n-atlas 500

cmrtk-refiner -o /output-dir/ --data-dir /input-dir/ --refine --csv-path /path-to-csv --n-top 7 --n-atlas 500
```

`--csv-path` indicates a csv file that contains a list of atlas meshes to use. For RBH project, the csv file is at
`/cardiact/RBH_3D_atlases/3D.csv`

### 4. Extractor
Extractor extracts 3D triangular mesh from 3D segmentations using marching cubes algorithms. 
It also extracts landmarks. 

It takes input from
```
/path/
    subject1/
        seg_lvsa_SR_ED.nii.gz           ->  ED segmentation
        seg_lvsa_SR_ES.nii.gz           ->  ES segmentation
    subject2/
        ...
```
or from
```
/path/
    subject1/
        refine/
            seg_lvsa_SR_ED.nii_refined.nii.gz       ->  Refined ED segmentation
            seg_lvsa_SR_ES.nii_refined.nii.gz       ->  Refined ES segmentation
    subject2/
        ...
```
and generates output
```
/path/
    subject1/
        landmark_ED.vtk                 ->  ED landmarks
        landmark_ES.vtk                 ->  ES landmarks
        mesh/
            LVendo_ED.vtk               ->  LV Endo ED mesh
            LVendo_ES.vtk               ->  LV Endo ES mesh
            LVepi_ED.vtk                ->  LV Epi ED mesh
            LVepi_ES.vtk                ->  LV Epi ES mesh
            LVmyo_ED.vtk                ->  LV Myo ED mesh
            LVmyo_ES.vtk                ->  LV Myo ES mesh
            RV_ED.vtk                   ->  RV ED mesh
            RV_ES.vtk                   ->  RV ES mesh
            RVepi_ED.vtk                ->  RV Epi ED mesh
            RVepi_ES.vtk                ->  RV Epi ES mesh
    subject2/
        ...
```
To run only the extractor, use one of the following commands:
```
cmrtk-pipeline -o /output-dir/ --data-dir /input-dir/ --extract

python -m CMRSegment.extractor -o /output-dir/ --data-dir /input-dir --extract

cmrtk-extractor -o /output-dir/ --data-dir /input-dir/ --extract
```

### 5. Coregister
Coregister registers each subject mesh with an atlas and transform each subject to the same atlas space. 

It takes input from
```
/path/
    subject1/
        landmark_ED.vtk                 ->  ED landmarks
        landmark_ES.vtk                 ->  ES landmarks
        mesh/
            LVendo_ED.vtk               ->  LV Endo ED mesh
            LVendo_ES.vtk               ->  LV Endo ES mesh
            LVepi_ED.vtk                ->  LV Epi ED mesh
            LVepi_ES.vtk                ->  LV Epi ES mesh
            LVmyo_ED.vtk                ->  LV Myo ED mesh
            LVmyo_ES.vtk                ->  LV Myo ES mesh
            RV_ED.vtk                   ->  RV ED mesh
            RV_ES.vtk                   ->  RV ES mesh
            RVepi_ED.vtk                ->  RV Epi ED mesh
            RVepi_ES.vtk                ->  RV Epi ES mesh
    subject2/
        ...
```
and generates output
```
/path/
    subject1/
        registration/
            temp/                           ->  Temporary files
            debug/                          ->  Intermediate meshes
            landmarks.dof.gz
            rigid/                          ->  Meshes after rigid transformation
                LVendo_ED.vtk               ->  LV Endo ED mesh
                LVendo_ES.vtk               ->  LV Endo ES mesh
                LVepi_ED.vtk                ->  LV Epi ED mesh
                LVepi_ES.vtk                ->  LV Epi ES mesh
                LVmyo_ED.vtk                ->  LV Myo ED mesh
                LVmyo_ES.vtk                ->  LV Myo ES mesh
                RV_ED.vtk                   ->  RV ED mesh
                RV_ES.vtk                   ->  RV ES mesh
                RVepi_ED.vtk                ->  RV Epi ED mesh
                RVepi_ES.vtk                ->  RV Epi ES mesh
            nonrigid/                       -> Meshes after nonrigid transformation
                LVendo_ED.vtk               ->  LV Endo ED mesh
                LVendo_ES.vtk               ->  LV Endo ES mesh
                LVepi_ED.vtk                ->  LV Epi ED mesh
                LVepi_ES.vtk  s              ->  LV Epi ES mesh
                LVmyo_ED.vtk                ->  LV Myo ED mesh
                LVmyo_ES.vtk                ->  LV Myo ES mesh
                RV_ED.vtk                   ->  RV ED mesh
                RV_ES.vtk                   ->  RV ES mesh

    subject2/
        ...
```
To run only the coregister, use one of the following commands:
```
cmrtk-pipeline -o /output-dir/ --data-dir /input-dir/ --coregister --template-dir /template-dir --param-dir /param-dir

python -m CMRSegment.coregister -o /output-dir/ --data-dir /input-dir --coregister --template-dir /template-dir --param-dir /param-dir

cmrtk-coregister -o /output-dir/ --data-dir /input-dir/ --coregister --template-dir /template-dir --param-dir /param-dir
```

Default template dir is [`input/params`](input/params)

Default param dir is [`lib/CMRSegment/resource`](lib/CMRSegment/resource)


## The pipeline

Pipeline links all the above modules together.
```
cmrtk-pipeline -o /output-dir/ --data-dir /input-dir/
(if you want to run segmentor)
--segment
    --model-path /path
    --segment-cine
    --torch
(if you want to run refiner)
--refine
    --csv-path /path
    --n-top 7
    --n-atlas 500
(if you want to run extractor)
--extract
    --iso-value 120
    --blur 2
(if you want to run coregister)
--coregister
    --template-dir /path
    --param-dir /path
```

To see details of these options, look at `argument_parser()` function in the file [`pipeline/config.py`](lib/CMRSegment/pipeline/config.py)
