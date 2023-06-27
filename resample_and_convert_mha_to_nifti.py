#%%
#!/usr/bin/env python3

import os
import numpy as np
import SimpleITK as sitk  
import nibabel as nb
from num2words import num2words

#%%
def read_mha(path):
    """
    ​Given a path the function reads the image and returns header as well as the intensity values into a 3d array.
    :param path: String value that represents the path to the .mha file
    :return: image executable and 3D array filled with intensity values
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    image = reader.Execute()
    
    # get array with values from sitk image
    image_array = sitk.GetArrayFromImage(image)
    
    return image, image_array
 
    
def write_mha(path, image_array, image=None):
    """
    Given a 3D array and header the function saves a .mha file at the requested path.
    :param image: Executable of image (contains header information, etc.). Highly recommended to provide it
    :param image_array: 3D array filled with intensity values corresponding to a .mha file
    :param path: String value that represents the path to the created .mha file
    :return: Image file corresponding to the input is saved to path
    """
    # get sitk image from array of values
    new_image = sitk.GetImageFromArray(image_array)

    # write header info into new image
    if image is not None:
        new_image.SetOrigin(image.GetOrigin())
        new_image.SetSpacing(image.GetSpacing())
        
    # save image
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(new_image)
    
    
def resample_mha(image, new_spacing, interpolator=sitk.sitkLinear, default_value=-1000):
    """
    Given an SITK image  it resamples the image to a given spacing
    :param image: SITK image executable
    :param new_spacing: spacing to which image is resampled, e.g. [1,1,1] for isotropic resampling
    :param interpolator: which interpolator to use, e.g. sitk.sitkLinear, sitk.sitkNearestNeighbor, sitk.sitkBSpline
    :param default_value: pixel value when a transformed pixel is outside of the image. The default default pixel value is 0.
    :return: resampled image executable
    """
    # load sitk image and create instance of ResampleImageFilter class 
    # image = sitk.ReadImage(path)
    resample = sitk.ResampleImageFilter()
    
    # set parameters for resample instance
    resample.SetInterpolator(interpolator)
    resample.SetDefaultPixelValue(default_value)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    
    # compute new size and set parameters for resample instance
    orig_size = np.array(image.GetSize(), dtype=np.int32)
    orig_spacing = np.array(image.GetSpacing(), dtype=np.float32)
    new_spacing = np.array(new_spacing, dtype=np.float32)
    new_size = orig_size*(orig_spacing/new_spacing)
    new_size = np.ceil(new_size).astype(np.int16) #  image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size) 

   # perform the resampling with the params set above
    new_image = resample.Execute(image)
    
    return new_image


#%%
if __name__ == "__main__": 
    original_directory = 'worskspace/images/original/V002'
    target_directory = 'worskspace/images/preprocessed/V002'
    
    file_nr = 1
    for file in os.listdir(original_directory):
        # check if file is a mha file
        if file.endswith(".mha"):
            # print current file
            print(f"Current file: {file}")

            # get full pth to file
            path_to_file = os.path.join(original_directory, file)
            # keep only file name, without .mha at the end to be used for saving
            # file_only = os.path.splitext(file)[0]
            # alternatively, define name using words as 3D Slicer interprets digits as a sequence for 3D volume
            file_only = 'frame_' + num2words(file_nr)
            
            # load mha file as array
            image, image_array = read_mha(path_to_file)
            
            # resample image (do not resample last dim as it will be dropped) to specified spacing
            image_resampled = resample_mha(image, new_spacing=[1.0,1.0,image.GetSpacing()[2]], interpolator=sitk.sitkLinear, default_value=0)
            
            # get array from image and move channel dim to end (needed for 3D Slicer and MONAI label)
            image_array_resampled = sitk.GetArrayFromImage(image_resampled)[0,:,:, None]
            # print(f'Shape of resampled image: {image_array_resampled.shape}')   # (270, 270, 1)
            
            # convert numpy to nii.gz and save to disk
            output_file = os.path.join(target_directory, f"{file_only}.nii.gz")
            # print(output_file)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            affine=np.eye(4) # define the affine matrix to correctly orient image in vv, no effect in python
            affine[0,0]=0
            affine[0,1]=-1
            affine[1,1]=0
            affine[1,0]=-1
            affine[2,2]=1
            affine[3,3]=1
            image_array_resampled_nifti = nb.Nifti1Image(image_array_resampled, affine=affine)
            nb.save(image_array_resampled_nifti, output_file)
            
            file_nr += 1
