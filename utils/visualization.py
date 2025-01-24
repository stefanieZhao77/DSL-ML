import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

class Visualization:
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id
        self.runs = mlflow.search_runs(self.experiment_id)
        

    def plot_metrics_comparison(self, models, metrics):
        # Function to plot comparison of metrics for different ML models
        plt.figure(figsize=(10, 6))
        for model in models:
            model_runs = self.runs[self.runs["model"] == model]
            metric_values = model_runs[metrics]
            plt.plot(metric_values, label=model)
        plt.title("Metrics Comparison")
        plt.xlabel("Runs")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.show()
        
    