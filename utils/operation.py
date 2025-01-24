import pandas as pd

def load_operation_processor(context, operation):
    """
        Load the file by the path and write the data into context
    """
    # Load the dataset from the file specified in the parameters
    dataset_name = operation.target
    file_param = next(param for param in operation.parameters if param.name == 'file')
    file_path = file_param.value
    file_extension = file_path.split('.')[-1]
    if file_extension == 'xlsx':
        context.datasets[dataset_name] = pd.read_excel(file_path)
    elif file_extension == 'csv':
        context.datasets[dataset_name] = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    context.datasets[dataset_name].columns = context.datasets[dataset_name].columns.str.replace(' ', '')
    

def split_operation_processor(context, operation):
    """
    Handling imbalanced data  
    """
    # Split the dataset according to the ratio specified in the parameters
    dataset_name = operation.target
    ratio_param = next(param for param in operation.parameters if param.name == 'ratio')
    ratio = float(ratio_param.value)

def fillna_operation_processor(context, operation):
    """
    Fill the missing values in the dataset
    """
    # Fill the missing values in the dataset according to the method or value specified in the parameters
    dataset_name = operation.dataset
    field_name = operation.field
    method_param = next((param for param in operation.parameters if param.name == 'method'), None)
    value_param = next((param for param in operation.parameters if param.name == 'value'), None)
    
    if method_param:
        method = method_param.value
        context.datasets[dataset_name][field_name].fillna(method=method, inplace=True)
    elif value_param:
        value = value_param.value
        context.datasets[dataset_name][field_name].fillna(value=value, inplace=True)
    else:
        raise ValueError("Either a method or a value must be specified for fillna operation")
        

def replace_operation_processor(context, operation):
    """
    replace the values in the dataset
    """
    # Replace the values in the dataset according to the method or value specified in the parameters
    dataset_name = operation.dataset
    field_name = operation.field
    old_param = next((param for param in operation.parameters if param.name == 'old'), None)
    new_param = next((param for param in operation.parameters if param.name == 'new'), None)
    
    if old_param and new_param:
        old = old_param.value
        new = new_param.value
        context.datasets[dataset_name][field_name].replace(old, new, inplace=True)
    else:
        raise ValueError("Either a method or a value must be specified for replace operation")

def remove_duplicates_operation_processor(context, operation):       
    """
    remove the duplicates in the dataset
    """
    dataset_name = operation.dataset
    if dataset_name and dataset_name in context.datasets:
        context.datasets[dataset_name].drop_duplicates(inplace=True)
    else:
        raise ValueError("Dataset name must be specified for remove_duplicates operation")
    
operation_processors = {
    'load': load_operation_processor,
    'split': split_operation_processor,
    'fillna': fillna_operation_processor,
    'replace': replace_operation_processor,
    'remove_duplicates': remove_duplicates_operation_processor,
}

