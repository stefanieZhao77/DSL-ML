from textx import textx_isinstance, metamodel_from_file
from . import mlmodel
from utils.data_manager import handle_features
from utils.rule import rule_apply
from utils import autoML
from mlflow.tracking import MlflowClient

class Model(object):
    def __init__(self, parent, name, task, elements):
        self.name = name
        self.elements = elements
        self.task = task

    def execute(self, datasets, experiment_id):
        mlmodels = {}
        rulesets = []
        context = {}
        metrics = []
        starts = []
        start_feature = None
        mm = metamodel_from_file('rule_model.tx')
        # parse the DSL script
        for element in self.elements:
            if textx_isinstance(element, mm['MLModel']):
                if element.type == "AutoML":
                    ml_model = autoML.AutoML(self.task)
                else:
                    ml_model = mlmodel.MLModel(self, element.name, element.type, element.parameters, self.task)
                mlmodels[element.name] = ml_model
            # Initial features
            if textx_isinstance(element, mm['FeatureSelection']):
                context = handle_features(element.name, datasets, element.features.features, element.features.dataset, element.features.start, element.features.end, context, element.dataset1, element.feature1, element.dataset2, element.feature2)
                if len(context) == 0:
                    print("No features selected. Please check your feature selection rules.")
                    return
            if textx_isinstance(element, mm['RuleSet']):
                rulesets.append(element)
            if textx_isinstance(element, mm['Start']):
                start_feature = element.feature
                starts = element.mlModels.split(",")
            if textx_isinstance(element, mm['Metric']):
                metrics = element.name.values
        # start
        for start in starts:
            if start not in mlmodels.keys():
                print(f"Start {start} not defined in the model.")
                return
            else:
                if start_feature not in context.keys():
                    print(f"Start feature {start_feature} not found in the context.")
                    return
                mlmodels[start].train(context[start_feature], experiment_id)
                
        for ruleset in rulesets:
            for rule in ruleset.rules:                    
                rule_apply(rule.condition, rule.action, context, experiment_id, mlmodels, metrics)    
        self.log_model(experiment_id)     

    def log_model(self, experiment_id):
        client = MlflowClient()
        runs = client.search_runs(experiment_id)
        for run in runs:
            print("------")
            print(f"Run Name: {run.data.tags.get('mlflow.runName')}")
            print(f"Run ID: {run.info.run_id}")
            print(f"Metrics: {run.data.metrics}")
            print(f"Parameters: {run.data.params}")
            print("------")         
