import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve 
from sklearn.metrics import auc


def error_analysis(test_df, prediction_label, prediction_prob):
    '''
    This function returns a dataframe with needed information for error analysis.
    
    parameters:
        prediction_label (numpy array): predicted labels output by the model
        prediction_prob (numpy array): predicted probabilities output by the model
    '''
    # Extract the index number of the comments that have been misclassified
    error_index = np.where((test_df.Recommended.values==prediction_label) == False)[0]

    # Create a subset of the test data that only contains rows that have been
    # misclassified and find the split of these errors regarding their true labels
    error_df = test_df.iloc[error_index]
    print('Distribution of misclassified texts:\n', error_df.Recommended.value_counts())
    
    # Add a column of the predicted value (the predicted probability instead of the hard label)
    model_df = test_df.copy()
    model_df['Prediction'] = prediction_prob
    
    # Compute the absolute difference between the true label and the predicted probabilitiy
    model_df['Absolute_error'] = model_df.apply(lambda row: abs(row.Prediction-row.Recommended), axis=1)
    
    return model_df

def plot_absolute_error(model_df):
    '''
    This function returns two subplots that show the distributions of misclassified and correctly
    classified texts' absolute error values
    
    parameter:
        model_df (pandas DataFrame): a dataframe created from test_df (which contains all
            comments in the test dataset) with additional columns containing the corresponding
            model's probability prediction as well as a calculated field, which is the absolute
            difference between the probability predictions and the true labels
            
    returns:
        Two histogram distribution subplots
    '''
    plt.rcParams["figure.figsize"] = (14,6)
    fig, ax = plt.subplots(1, 2)
    
    # Plot densities of wrongly classified texts' absolute errors
    ax[0].hist(model_df.loc[(model_df.Recommended==0) & (model_df.Absolute_error>=0.5)].Absolute_error.values, 
               bins=50, 
               alpha=0.5, 
               density=True,
               label='Negative reviews')
    ax[0].hist(model_df.loc[(model_df.Recommended==1) & (model_df.Absolute_error>=0.5)].Absolute_error.values, 
               bins=50, 
               alpha=0.5, 
               density=True,
               label='Positive reviews')
    ax[0].legend()
    ax[0].set_xlabel('Absolute error')
    ax[0].set_ylabel('Density')
    ax[0].set_title("Density distribution of misclassified texts' absolute error values")
    
    # Plot densities of correctly classified texts' absolute errors
    ax[1].hist(model_df.loc[(model_df.Recommended==0) & (model_df.Absolute_error<0.5)].Absolute_error.values, 
               bins=50, 
               alpha=0.5, 
               density=True,
               label='Negative reviews')
    ax[1].hist(model_df.loc[(model_df.Recommended==1) & (model_df.Absolute_error<0.5)].Absolute_error.values, 
               bins=50, 
               alpha=0.5,
               density=True,
               label='Positive reviews')
    ax[1].legend()
    ax[1].set_xlabel('Absolute error')
    ax[1].set_ylabel('Density')
    ax[1].set_title("Density distribution of correctly classified texts' absolute error values")
    
    plt.show()
    
def plot_auc(test_df, predict_prob, model_name):
    '''
    This function returns a AUC plot that shows the relationship between
    False Positive Rate and True Positive Rate
    '''
    plt.rcParams["figure.figsize"] = (8,6)

    fpr, tpr, threshold = roc_curve(test_df.Recommended, predict_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.legend(loc = 'lower right')
    
    plt.xlim([0, 1]) 
    plt.ylim([0, 1]) 
    plt.ylabel('True Positive Rate') 
    plt.xlabel('False Positive Rate') 
    
    plt.title(f'ROC Curve of {model_name}')
#     plt.savefig(f"{model_name}_AUC.png", pad_inches=0.05, dpi=1000)
    plt.show()
    
    return roc_auc