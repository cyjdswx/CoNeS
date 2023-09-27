# CoNeS: Conditional Neural Fields with Shift Modulation for Multi-sequence MRI Translation
The official implementation of paper:"**[CoNeS: Conditional Neural Fields with Shift Modulation for Multi-sequence MRI Translation](https://arxiv.org/abs/2309.03320)**"

# Install requirements
'''bash
pip install -r requirements.txt
'''

## Usage
1. Create the config file of your own data as brats2018_3m.json and normlize the data intensity to [-1,1].
2. To train the image translation model using brats2018, run:
'''
bash train_brats.sh
'''
3. To test the model
run test_brats.sh

## Acknowledgements
This code is based on the [ASAPNet](https://github.com/tamarott/ASAPNet.git).
