# Protein-subcellular-localisation-method
# Usage
To run :
```
python main.py
```
Nonlinear mixup is defined in utils.py.

Proposed sampling approach is implemente in main.py as,
```
def train(train_loader,model,criterion,optimizer,epoch,valid_loss,best_results,start,train_gen):
```
Hyperparameter 'n' for the proposed sampling approach, paths to the training and test set images can be set in <config.py>.
