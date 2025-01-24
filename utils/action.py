from utils.data_manager import reduce_features, handle_features
from mlflow.tracking import MlflowClient

def execute(model, actions, context, experiment_id, mlmodels):
    actions = actions.split(",")
    if len(actions) == 2:
        if actions[0] in context.keys() and actions[1] in mlmodels.keys():
            mlmodels[actions[1]].train(context[actions[0]], experiment_id)
        elif actions[0] in mlmodels.keys() and actions[1] in context.keys():
            mlmodels[actions[0]].train(context[actions[1]], experiment_id)
        else:
            print("The action should be one feature and one model")
    elif len(actions) == 1:
        if actions[0] in mlmodels.keys():
            print(f"Please provide a feature to train the model {actions[0]}")
        elif actions[0] in context.keys():
            mlmodels[model].train(context[actions[0]], experiment_id)
        else:
            print(f"Action {actions[0]} not found. Please select from {mlmodels.keys()} and {context.keys()}.")

            

            
        
                

