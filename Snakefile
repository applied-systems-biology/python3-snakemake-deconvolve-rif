import os

import algorithms
import json
import numpy as np

print("Input read from " + config["input"])
print("Output written to " + config["output"])

if not config["input"].endswith("/"):
    config["input"] += "/"
if not config["output"].endswith("/"):
    config["output"] += "/"

if not os.path.exists(config["output"]):
    os.makedirs(config["output"])

# Collect sample information
samples = {}
samples_config_file = config["input"] + "/samples.json"
samples_config = None

with open(samples_config_file, "r") as f:
    samples_config = json.load(f)

for sample_name in os.listdir(config["input"]):
    sample_path = config["input"] + "/" + sample_name
    output_path = config["output"] + "/" + sample_name
    if not os.path.isdir(sample_path):
        continue

    experiments = sorted([x for x in os.listdir(sample_path) if os.path.isdir(sample_path + "/" + x)])

    samples[sample_name] = samples_config[sample_name]

    print("Sample " + sample_name + " of size " + str(len(experiments)))

rule all:
    input: expand(config["output"] + "{sample}/deconvolved.tif", sample=list(samples.keys()))

rule deconvolve:
    input: config["input"] + "{sample}/in/data.tif",
           config["input"] + "{sample}/psf/psf.tif"
    output: config["output"] + "{sample}/deconvolved.tif"
    run:
        sample=samples[wildcards["sample"]]
        data_voxel = sample["voxel-size-sample-xyz"]
        data_voxel = np.array((data_voxel, data_voxel, data_voxel))
        psf_voxel = sample["voxel-size-psf"]
        psf_voxel = np.array((psf_voxel["y"], psf_voxel["xz"], psf_voxel["xz"]))

        algorithms.deconvolve(input_data_file=input[0], input_psf_file=input[1], output_file=output[0], data_voxel_size=data_voxel, psf_voxel_size=psf_voxel)

rule input_files:
    output:
        config["input"] + "{sample}/in/data.tif",
        config["input"] + "{sample}/psf/psf.tif"