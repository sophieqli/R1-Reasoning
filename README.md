# R1-Reasoning
After importing the libraries and loading the dataset, I began experimenting with the peft model training cell. First, I created the peft_config, which sets the parameters of the model. I noticed that the r = 8 corresponds to the # of trainable parameters (basically the “rank” of the peft model), because when I set r = 4, this number halved.

Then, I initially set up the SFTTrainer’s config with the following parameters: max_seq_length=100, per_device_train_batch_size=4, learning_rate=2e-4, num_train_epochs=3, finishing with a training loss of ~1.83 after ~3000 steps.

Trying to improve the training, I tweaked the learning_rate to 1e-4. I hoped that this would enable the model to take more precise steps, since if the lr is too high, the model might overshoot the optimal weights. To compensate for the smaller updates, I doubled the # of epochs to 6. Hoping to improve stability and convergence, I used the Adam optimizer and a linear LR scheduler.

In this notebook, I printed out the iteration where I used 8 epochs. As one can observe, loss plateaus a bit after 6000 steps in the 1.76-1.78 range, which may suggest a limitation of the dataset itself.

Another tweak I wanted to try was setting Gradient accumulation = 4, which essentially scales up the batch size, so that the gradient of the batch would be closer to the true value. However, this would likely require scaling up epochs by 4 (as the model updates less frequently), and that would be quite computationally slow.  

Extra exploration: I wanted to understand the dataset a bit better, so I printed some extra output: the input/output length distributions and the most common phrases in the input.
