# Dual-domain Gradient Flatness Attack (DGFA)

## Abstract
This repository contains the implementation of the **Dual-domain Gradient Flatness Attack (DGFA)**, a novel adversarial attack method designed to improve adversarial transferability across deep neural networks (DNNs). DGFA combines both spatial and frequency gradient information to generate more transferable adversarial examples. By transforming input samples to the frequency domain, sampling similar examples within a neighborhood, and reconverting them to the spatial domain, DGFA generates flatter adversarial examples that enhance transferability. This method significantly outperforms traditional gradient-based attacks in terms of attack success rates for both Convolutional Neural Network (CNN) and Vision Transformer (ViT) models.

**Note**: This work is currently under review at *The Visual Computer*.


## Attack Process of DGFA

The attack process of DGFA is illustrated in the figure below:

![Attack process of DGFA](images/flow.png)

*Figure 1: Attack process of DGFA*

The figure shows the series of steps DGFA follows to generate adversarial examples by transforming input samples into the frequency domain, applying neighborhood sampling, and transforming them back to the spatial domain to generate flatter adversarial examples.

## Requirements
To run this code, the following dependencies are required:
- Python 3.x
- PyTorch (>= 1.10.0)
- Numpy
- OpenCV
- Scipy
- Matplotlib

You can install the required libraries by running:

```bash
conda env create -f environment.yaml
```

## Usage

### Running the Attack

To execute the DGFA attack, use the following command:

```bash
python main.py --config configs/DGFA.yaml
```

This will initiate the adversarial attack process using the configuration specified in the `DGFA.yaml` file.

### Verifying the Attack

To verify the effectiveness of the attack, you can use the following command:

```bash
python verify.py --config configs/DGFA.yaml
```

This command will assess the success rate of adversarial examples generated by the DGFA method on the specified models and dataset.

### Datasets and Models

This implementation uses datasets and models from the **[SSA repository](https://github.com/yuyang-long/SSA)**, which includes pre-trained models and data suitable for evaluating the transferability of adversarial attacks.

## Configuration

The configuration file `configs/DGFA.yaml` contains important parameters that control the attack's behavior, including:

- **Neighborhood Range Factor (Φ)**: Defines the size of the neighborhood for sampling.
- **Perturbation Constraint (δ)**: Sets the maximum perturbation allowed in the attack.
- **Attack Iterations (k)**: Specifies the number of attack iterations for refinement.
- **Neighborhood Sample Number (N)**: The number of neighborhood samples to consider during the attack process.

You can modify these parameters within the configuration file to suit your experimental setup.

## Experimental Setup

In our empirical evaluations, the DGFA method was tested on several popular architectures, including **Inception-v3**, **ResNet-152**, and **Vision Transformers (ViTs)**, using the **ImageNet dataset**. The primary metric for evaluating attack performance was the **Attack Success Rate (ASR)**, which quantifies the percentage of adversarial examples that successfully mislead the target model.

## License

This code is released under the MIT License. See the LICENSE file for more details.

## Contact

For further inquiries or questions, please contact the first author at [2010286@stu.neu.edu.cn].

## Citation
please cite it as follows:
```
Once the paper is accepted, we will update the citation.
```

<!-- ```
@article{zkb2024DGFA,
  title={Improving the Transferability of Adversarial Attacks with Dual-domain Gradient Flatness Optimization},
  author={First Author, Second Author, Third Author},
  journal={The Visual Computer},
  year={2024},
  doi={doi:XXXXXXX}
}
``` -->