# MONAI Label - Radiology Sample Application in 2D
This app includes a segmentation_2d model to do interactive auto-segmentation over radiology (2D) images. Based on the original [3D version](https://github.com/Project-MONAI/MONAILabel/tree/main/sample-apps/radiology) by the MONAI initiative.

## How To Use the App
* Run a Docker container pulling a monailabel image from dockerhub and mounting the local path to this project `docker run --shm-size 30GB -it -w /workspace --gpus all -v your_local_path_to_MONAILabel_radiology2d/:/workspace/ -p 127.0.0.1:9998:8000 projectmonai/monailabel:0.7.0 bash` 
* Note that port forwarding was used when running the Docker container to forward the monailabel server to port 9998 on the localhost. 
* Inside the container, start MONAI Label Server with the segmentation_2d model `monailabel start_server --app /workspace/apps/radiology --studies /workspace/images/your_patient_folder_name --conf models segmentation_2d`
* Open 3D Slicer on your local machine, put adress of forwarded port (e.g., http://localhost:9998/) into the MONAI Label server field and start annotating/training/inference !

## Additional Information
* The 2D images must be of shape (h,w,1)
* The 2D images must be named with letters only (frame_eight, not frame_008), otherwise [3D Slicer will interpret them as slices of a single 3D volume](https://github.com/Project-MONAI/MONAILabel/discussions/1243)
* You can change the model architecture, roi input size, target labels, etc. under /apps/radiology/lib/configs/segmentation_2d.py
* You can change the data augmentations under /apps/radiology/lib/trainers/segmentation.py
* If you want to use epistemic uncertainty, inside your docker, you need to change some code in  `mv /usr/local//lib/python3.8/dist-packages/monailabel/tests/scpring/epistemic_v2.py` and in `./apps/radiology/configs/segmentation_2d.py`
