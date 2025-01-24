from textx import textx_isinstance, metamodel_from_file
from mlflow.tracking import MlflowClient
from utils.action import execute

my_mm = metamodel_from_file('rule_model.tx')

def rule_apply(condition, actions, context, experiment_id, mlmodels, metrics):
    if textx_isinstance(condition, my_mm['SimpleExpression']):
        result = evaluate_simple_expression(condition, actions, context, experiment_id, mlmodels, metrics)
        if result:
            print(f"Rule triggered.")
            execute(condition.model, actions, context, experiment_id, mlmodels)
    elif textx_isinstance(condition, my_mm['BinaryExpression']):
        result = evaluate_binary_expression(condition, actions, context, experiment_id, mlmodels, metrics)
        if result:
            print(f"Rule triggered.")
            execute(condition.left.model, actions, context, experiment_id, mlmodels)

def evaluate_simple_expression(expr, actions, context, experiment_id, mlmodels, metrics):
    if expr.model not in mlmodels.keys():
        print(f"Model {expr.model} not found. Please select from {mlmodels.keys()}.")
        return False
    if expr.metric not in metrics:
        print(f"Metric {expr.metric} not found. Please select from {metrics}.")
        return False
    client = MlflowClient()
    runs = client.search_runs(experiment_id, max_results=2)
    for run in runs:
        if expr.model in run.data.tags.get("mlflow.runName"):
            value = run.data.metrics.get(expr.metric)
            return evaluate_simple_expression_condition(expr, value)
    return False

def evaluate_simple_expression_condition(expr, value):
    return eval(f"{value} {expr.operator} {expr.value}")

def evaluate_binary_expression(expr, actions, context, experiment_id, mlmodels, metrics):
    if expr.logicOperator == 'and':
        left_result = evaluate_simple_expression(expr.left, actions, context, experiment_id, mlmodels, metrics)
        right_result = evaluate_simple_expression(expr.right, actions, context, experiment_id, mlmodels, metrics)
        return left_result and right_result
    elif expr.logicOperator == 'or':
        left_result = evaluate_simple_expression(expr.left, actions, context, experiment_id, mlmodels, metrics)
        right_result = evaluate_simple_expression(expr.right, actions, context, experiment_id, mlmodels, metrics)
        return left_result or right_result
    else:
        raise Exception("Logic operator not supported")
