# Run MobileNet V2 on Raspberry Pi with OpenVINO™ Toolkit using Intel® Neural Compute Stick 2

-   Raspberry Pi 3 Model B+
-   Raspbian OS ([Legacy version, 2020-08-24](https://downloads.raspberrypi.org/raspios_armhf/images/raspios_armhf-2020-08-24/))

## Prerequisites

1.  Modify `apt` source, update and upgrade packages.
2.  Install Python3, `pip3` and uninstall Python2, etc.
3.  Install `numpy` and `opencv-python`. Required versions are:
    -   `numpy==`
    -   `opencv-python==`
4.  Read the [appendix](#appendix) of this README for assets.

## Install the OpenVINO™ Toolkit for Raspbian OS Package

This section follows the instructions from the `2021.4/openvino_docs_install_guides_installing_openvino_raspbian.html` document in the [OpenVINO™ Toolkit documentation archives](https://docs.openvino.ai/archives/index.html).

### Download and Install OpenVINO™

From the [OpenVINO™ Toolkit 2021.4 Downloads page](https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.4/), download the version for Raspbian OS: `l_openvino_toolkit_runtime_raspbian_p_2021.4.582.tgz` to the `~/Downloads` folder.

Then run:

```bash
sudo mkdir -p /opt/intel/openvino_2021
sudo tar -xf  ~/Downloads/l_openvino_toolkit_runtime_raspbian_p_2021.4.582.tgz --strip 1 -C /opt/intel/openvino_2021
```

Now the OpenVINO toolkit components are installed. Additional configuration steps are still required. Continue to the next sections to install External Software Dependencies, configure the environment and set up USB rules.

### Install External Software Dependencies

```bash
sudo apt install cmake
```

### Set the Environment Variables

You must update several environment variables before you can compile and run OpenVINO toolkit applications. Run the following script to temporarily set the environment variables:

```bash
source /opt/intel/openvino_2021/bin/setupvars.sh
# Or, to set up the OpenVINO environment variables automatically at each terminal startup, run:
# echo "source /opt/intel/openvino_2021/bin/setupvars.sh" >> ~/.bashrc
# source ~/.bashrc
```

`[setupvars.sh] OpenVINO environment initialized` should be printed.

### Add USB Rules for an Intel® Neural Compute Stick 2 device

This task applies only if you have an Intel® Neural Compute Stick 2 device.

1.  Add the current Linux user to the users group:

    ```bash
    sudo usermod -a -G users "$(whoami)"
    ```
    Log out and log in for it to take effect.
2.  If you didn't modify `.bashrc` to permanently set the environment variables, run `setupvars.sh` again after logging in:
    
    ```bash
    source /opt/intel/openvino_2021/bin/setupvars.sh
    ```
3.  To perform inference on the Intel® Neural Compute Stick 2, install the USB rules running the `install_NCS_udev_rules.sh` script:
    
    ```bash
    sh /opt/intel/openvino_2021/install_dependencies/install_NCS_udev_rules.sh
    ```
    Plug in your Intel® Neural Compute Stick 2.

You are now ready to verify the Inference Engine installation. Run the following script to check the available devices:

```bash
python3 - << 'EOF'
from openvino.inference_engine import IECore
ie = IECore()
print(ie.available_devices)
EOF
```

`['MYRIAD']` should be printed.

## Prepare the MobileNet V2 Model

### Download OpenVINO™ with Model Optimizer

Since the guide `2021.4/openvino_docs_install_guides_installing_openvino_raspbian.html` writes:

> The package **does not include the Model Optimizer**. To convert models to Intermediate Representation (IR), you need to install it separately to your host machine.

So we need another host machine. For instance, **on another Ubuntu machine**, download the OpenVINO™ Toolkit 2021.4 package Ubuntu version `l_openvino_toolkit_dev_ubuntu20_p_2021.4.582.tgz` from the [OpenVINO™ Toolkit 2021.4 Downloads page](https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.4/).

Then rerun all the instructions above to install OpenVINO™ on that Ubuntu machine, but this time using the Ubuntu package. Verify that there's an `mo.py` in the directory `/opt/intel/openvino_2021/deployment_tools/model_optimizer`.

### Download and Convert the MobileNet V2 Model

First, install Pytorch (cpu-only is fine) and torchvision on your Linux machine:

```bash
# Furthermore, you can follow the instructions at https://pytorch.org/get-started/locally/
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Run the following command to export the model to the ONNX format:

```bash
python3 export.py
```

Now there should be a file `mobilenet_v2.onnx` in the current directory.

Next, run the following command to convert the ONNX model to OpenVINO™ IR format:

```bash
python3 /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py \
  --input_model mobilenet_v2.onnx \
  --output_dir . \
  --input_shape [1,3,224,224]
```

-   If you try to use MobileNet V3 Small, then an error would occur:
    
    ```bash

    ```
-   If you don't add the argument `--input_shape [1,3,224,224]`, then an error would occur:

    ```bash

    ```

Now there should be two files `mobilenet_v2.xml` and `mobilenet_v2.bin` in the current directory. (Maybe there's also a `mobilenet_v2.mapping` file, but it's not needed.)

Finally, copy these two files to your Raspberry Pi.

## Run Inference

Run the following command to test the model on validation set:

```bash
python3 main.py
```

Run the following command to start real-time inference from the camera:

```bash
python3 camera.py
```

## Results



## Appendix

In the release page of this project, there are several assets for your convenience:

1.  Validation folder, in which contains images for accuracy testing:
    -   `validation.zip`
2.  Demo folder, in which contains images for real-time inference testing:
    -   `demo.zip`
3.  Label mapping file:
    -   `label_mapping.json`
4.  OpenVINO™ Toolkit packages:
    -   `l_openvino_toolkit_runtime_raspbian_p_2021.4.582.tgz`
    -   `l_openvino_toolkit_dev_ubuntu20_p_2021.4.582.tgz`
5.  OpenVINO™ IR model files:
    -   `mobilenet_v2.xml`
    -   `mobilenet_v2.bin`
6.  Wheels for required Python packages on Raspbian OS:
    -   `numpy-`
    -   `opencv_python-`
