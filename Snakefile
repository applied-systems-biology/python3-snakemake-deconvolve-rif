import os

import algorithms

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

for sample_name in os.listdir(config["input"]):
    sample_path = config["input"] + "/" + sample_name
    output_path = config["output"] + "/" + sample_name
    if not os.path.isdir(sample_path):
        continue

    samples[sample_name] = {}

    print("Sample " + sample_name)

rule all:
    input: expand(config["output"] + "{sample}/deconvolved.tif", sample=list(samples.keys()))

rule deconvolve:
    input: config["output"] + "{sample}/convolved.tif",
           config["input"] + "{sample}/psf/psf.tif"
    output: config["output"] + "{sample}/deconvolved.tif"
    run:
        algorithms.deconvolve(input_data_file=input[0], input_psf_file=input[1], output_file=output[0])


rule convolve:
    input: config["input"] + "{sample}/in/data.tif",
           config["input"] + "{sample}/psf/psf.tif"
    output: config["output"] + "{sample}/convolved.tif"
    run:
        algorithms.convolve(input_data_file=input[0], input_psf_file=input[1], output_file=output[0])


rule input_files:
    output:
        config["input"] + "{sample}/in/data.tif",
        config["input"] + "{sample}/psf/psf.tif"