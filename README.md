# Amplitude amplification inspired QAOA

This repository contains the implementations of the variants of the 
Quantum Approximate Optimization Algorithm (QAOA) presented in 
"Amplitude amplification inspired QAOA: Improving
the success probability for solving 3SAT"

The files for each variants are found in their respective subdirectory:
 - qaoa_from_bitflip.py contains the implementation of the variant.
 - qaoa_from_bitflip_exp.py contains the experiment implementation.
 - experiments.py calls the experiments to obtain the success probability.
 - plots.ipynb shows the plots for the success probability of this variant.
 
Due to their large file sizes, the experiment data for each variant is compressed. To reproduce the plots, it needs to be uncompressed first (see for example [variant1.zip](https://github.com/UST-QuAntiL/aa_inspired_qaoa/blob/ab77527ad86f12683f2639acdcf59133c18c8d99/variant1/variant1.zip) for variant 1).
To rerun the experiments see ``run_and_save`` (for example in [experiments.py](https://github.com/UST-QuAntiL/aa_inspired_qaoa/blob/ab77527ad86f12683f2639acdcf59133c18c8d99/variant1/experiments.py)

### Disclaimer of Warranty

Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.
