# Protein-subcellular-localisation-method
# Usage
To run :
```
python main.py
```
The configuration file is <config.py>, in which hyperparameter 'n' for the proposed sampling approach paths to training set images and saved models can be set.

Nonlinear mixup is defined in utils.py.

Proposed sampling approach is implemente in main.py as,
```
def train(train_loader,model,criterion,optimizer,epoch,valid_loss,best_results,start,train_gen):
```
