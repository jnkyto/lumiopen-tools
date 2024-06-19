# Package for machine translation fine-tuning using Accelerate and DeepSpeed

## custom_trainer

### NOTE: Currently causes an unknown GPU memory issue!

Custom trainer using native Accelerate and PyTorch code.

## hf_trainer

HuggingFace trainer. May randomly crash after dataset preprocessing and before model loading.

## Things to note

Both of these trainers are still very much experimental and can only run with the Europarl 
dataset at the moment.