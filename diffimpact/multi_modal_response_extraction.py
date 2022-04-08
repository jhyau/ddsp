import os,sys
import argparse

checkpoints = os.listdir('./asmr/regnet-labels/')
output_path = '/juno/u/jyau/regnet/data/features/ASMR/orig_asmr_by_material_clips/modal_responses/'

# Load material ID file from the given Diffimpact checkpoint


for ckpt in checkpoints:
    if ckpt.find('3hr') != -1:
        path = os.path.join('./asmr/regnet-labels/', ckpt)
        print(path)
        ck_dir = os.listdir(path)

        # Get the gin file
        for files in ck_dir:
            if files.find('.gin') != -1:
                gin = files
                break

        gin_path = os.path.join(path, gin)
        print(gin_path)
        # Call the function to generate the modal response parts
        command = f"python modal_response_extraction.py {path} {gin_path} {output_path}"
        print(command)
        os.system(command)
