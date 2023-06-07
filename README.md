# ttt-semantic-segmentation
Test Time Training for Urban Street Semantic Segmentation

Required downloads: Cityscapes and Foggy Cityscapes

Train model using train.py. Run tests for a given model using eval_all.py, which will calculate the accuracy over the Cityscapes test set for each fog level (clear, 600m, 300m, 150m). Uncomment the lines that load the model at the end of eval_all.py to enable TTT Standard.
