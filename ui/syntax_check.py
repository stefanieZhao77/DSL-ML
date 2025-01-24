from lark import Lark, UnexpectedInput, UnexpectedCharacters, UnexpectedToken

dsl_grammar = """
    start: element+
    
    element: load | model
    
    load: "load" NAME "from" ESCAPED_STRING
    
    model: "model" NAME "task" task "{" model_element* "}"
    
    task: "predict" | "classification" | "regression"
    
    model_element: ml_model | metric | feature_selection | rule_set | start_rule | show | select | feature_engineering
    
    ml_model: "mlModel" NAME "type" ml_model_type ("{" parameter* "}")?
    
    ml_model_type: "RandomForest" | "DecisionTree" | "SVM" | "AutoML"
    
    parameter: "parameter" "name" "=" ESCAPED_STRING "value" "=" value_list
    
    value_list: "[" value ("," value)* "]" | value
    
    value: ESCAPED_STRING | SIGNED_FLOAT | SIGNED_INT
    
    metric: "metric" metric_name ("," metric_name)*
    
    metric_name: "rmse" | "mae" | "mse" | "accuracy" | "precision" | "recall" | "f1" | "auc" | "logloss" | "error" | "hamming_loss" | "jaccard" | "matthews_corrcoef" | "zero_one_loss" | "average_precision" | "neg_log_loss" | "neg_mean_absolute_error" | "neg_mean_squared_error" | "neg_mean_squared_log_error" | "neg_median_absolute_error" | "balanced_accuracy" | "adjusted_mutual_info_score" | "adjusted_rand_score" | "completeness_score" | "explained_variance" | "fowlkes_mallows_score" | "homogeneity_score" | "mutual_info_score" | "neg_brier_score" | "normalized_mutual_info_score" | "r2_score" | "rand_score" | "v_measure_score" | "all"
    
    start_rule: "start" fqn "with" NAME
    
    fqn: NAME ("," NAME)*
    
    select: "select" NAME feature_list
    
    feature_list: (NAME "." NAME) ("," NAME "." NAME)*
    
    feature_selection: "select" NAME feature_list "goal" goal? ("where" feature_condition ("and" feature_condition)*)? ("using" NAME)?
    
    goal: NAME "." NAME
    
    feature_condition: NAME "." NAME operator NAME "." NAME
    
    rule_set: "ruleSet" NAME "{" rule+ "}"
    
    rule: "rule" ":" "if" expression "then" fqn
    
    expression: simple_expression | binary_expression
    
    binary_expression: "(" expression logic_operator expression ")"
    
    simple_expression: NAME "." NAME operator number
    
    number: SIGNED_FLOAT | SIGNED_INT
    
    operator: "<" | ">" | "<=" | ">=" | "==" | "!="
    
    logic_operator: "and" | "or"
    
    show: "show" visualization ("," visualization)* ("from" NAME)?
    
    visualization: "models" | "features" | "metrics"
    
    NAME: /[a-zA-Z_][a-zA-Z0-9_-]*/
    
    feature_engineering: "engineer" NAME "{" feature_operation+ "}"
    
    feature_operation: create_feature | transform_feature | encode_feature
    
    create_feature: "create" NAME "as" ESCAPED_STRING
    
    transform_feature: "transform" feature_match "using" transform_method
    
    transform_method: "log" | "sqrt" | "square" | "standardize" | "normalize"
    
    encode_feature: "encode" feature_match "using" encode_method
    
    encode_method: "onehot" | "label" | "frequency"
    
    feature_match: NAME "." NAME

    %import common.ESCAPED_STRING
    %import common.SIGNED_FLOAT
    %import common.SIGNED_INT
    %import common.WS
    %ignore WS
"""

parser = Lark(dsl_grammar, start='start', parser='lalr')

def validate_dsl_syntax(input_text):
    try:
        parser.parse(input_text)
        return True, "Input is valid according to the DSL syntax."
    except UnexpectedCharacters as e:
        context = input_text.splitlines()[e.line - 1]
        pointer = ' ' * e.column + '^'
        return False, f"Syntax error at line {e.line}, column {e.column}:\n{context}\n{pointer}\nUnexpected character: {e.char}"
    except UnexpectedToken as e:
        context = input_text.splitlines()[e.line - 1]
        pointer = ' ' * e.column + '^'
        expected = ', '.join(e.expected)
        return False, f"Syntax error at line {e.line}, column {e.column}:\n{context}\n{pointer}\nExpected: {expected}\nFound: {e.token}"
    except Exception as e:
        return False, f"An error occurred while parsing: {str(e)}"


# if __name__ == "__main__":
#     test_input = """
#     load dataset1 from "/path/to/data.csv"
#     model MyModel task classification {
#         mlModel RandomForest1 type RandomForest {
#             parameter name = "n_estimators" value = [100]
#             parameter name = "max_depth" value = [10]
#         }
#         metric accuracy, f1
#         select features1 dataset1.feature1, dataset1.feature2 where dataset1.feature1 > dataset1.feature2
#         ruleSet MyRules {
#             rule: if RandomForest1.accuracy > 0.8 then ShowResults
#         }
#         start RandomForest1 with features1
#         show models, metrics from RandomForest1
#     }
#     """
    
#     is_valid, message = validate_dsl_syntax(test_input)
#     print(message)
