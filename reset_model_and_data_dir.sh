rm -rf /workspace/apps/radiology/model/segmentation_2d
rm /workspace/apps/radiology/model/segmentation_2d.pt
studies_dir='/workspace/images/preprocessed/V002/'
#rm -rf ${studies_dir}/labels
rm ${studies_dir}/.lock
rm ${studies_dir}/datastore_v2.json
