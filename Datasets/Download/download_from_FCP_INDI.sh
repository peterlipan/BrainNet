# ACPI
aws s3 cp s3://fcp-indi/data/Projects/ACPI/OutputsTimeSeriesTars/ /datastorage/li/fMRI/ACPI/ --recursive --no-sign-request

# HBN
python3 download_amazons3.py --s3-base s3://fcp-indi/data/Projects/HBN/CPAC_preprocessed_Derivatives/ \
  --output-dir /datastorage/li/fMRI/HBN

# BGSP
python3 download_amazons3.py --s3-base s3://fcp-indi/data/Projects/BGSP/cpac_out/output/pipeline_analysis/ \
  --output-dir /datastorage/li/fMRI/BGSP

# RocklandSample
python3 download_amazons3.py --s3-base s3://fcp-indi/data/Projects/RocklandSample/Outputs/cpac/ \
  --output-dir /datastorage/li/fMRI/RocklandSample