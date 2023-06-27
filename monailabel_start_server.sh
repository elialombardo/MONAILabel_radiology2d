# Specify patient data dir
study_dir='/workspace/images/V002/'

# Get the name of the patient subdir
pat_dir="$(basename "$study_dir")"
# Create folder where to later manually move trained model to
mkdir /workspace/apps/radiology/model/"${pat_dir}/"

# Start monai label server for a cine MRI specified by study_dir
monailabel start_server --app /workspace/apps/radiology --studies ${study_dir} --conf models segmentation_2d



