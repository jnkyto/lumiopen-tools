# Package for machine translation fine-tuning using Accelerate and DeepSpeed

## custom_trainer

### NOTE: Currently crashes on an unknown GPU memory issue!

Custom trainer using native Accelerate and PyTorch code. I have since moved on to the
hf_trainer, **leaving this script to an unusable state**. The crash happens during dataset 
preprocessing, the actual training loop should be working.

## hf_trainer

### Fine-tuning script using the vanilla HuggingFace trainer.

Currently only runs with the **Helsinki-NLP/europarl** -dataset with hard-coded preprocessing.
Future plans include to either load the data from a file formatted in a specific way, or 
to leave the preprocessing step to be handled by the user in some other manner.

Most hyperparameters can be modified with script arguments, refer to the Python script or its
corresponding `.sbatch` launch script for examples.

### Known issues:
1) Cache miss during the dataset loading process
    - Characterized by `xyz.arrow FileNotFoundError` or something similar.
2) Hang with no error output after the dataset loading process
    - Might be fixed by adding `trainer.accelerator.wait_for_everyone()` torch barrier after the loading
      process. Thing is, the trainer hasn't been instated at that point by design.
3) Deepspeed launch crash after the dataset loading process
    - This one is the rarest of the crashes, might also be fixed by setting up the torch
      barrier described above.

## Things to note

Both of these trainers are still very much experimental and can only run with the Europarl 
dataset at the moment. The `deepspeed_zero3.yaml` config files are required to run the scripts, 
but most of the options are superseded by the `.sbatch` launch script configs, so don't bother changing the 
number of machines set up in there, for example.