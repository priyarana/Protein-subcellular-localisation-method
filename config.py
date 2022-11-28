class DefaultConfigs(object):
    train_data = "path to training images" 
    
    #test_data = "path to training images (TS2)"   # your test data
    weights = "/checkpoints/"
    best_models = "/checkpoints/best_models/"  
    submit = "/submit/"
    model_name = "TrainedModel"
    num_classes = 28
    img_weight = 512
    img_height = 512
    channels = 4 
    lr =  0.03 #0.003
    
    batch_size = "set to x, where x + n = 32 and 'n' is the hyperparameter in the proposed sampling approach"
    epochs = 100


config = DefaultConfigs()
