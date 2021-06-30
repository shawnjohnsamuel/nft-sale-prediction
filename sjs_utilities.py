###########################
#      I M P O R T S      #
###########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, f1_score, roc_auc_score, plot_confusion_matrix
from sklearn import metrics
from pmdarima.utils import diff_inv




###########################
#  PHASE 4 - TIME SERIES  #
###########################


# create dataframe to document arima scores
arima_metrics = pd.DataFrame(columns = ['Model', 'Explained Variance', 'r^2', 'MAE', 'RMSE', 'ARIMA Order', 'Seasonal Order', 'AIC'])


def arima_eval(name, ts, train, test, model, periods=0, log=False, library='pmdarima'):
    '''
    Plot of train, test and predictions as well as model summary and metrics of current and previous models.
    If the train and test have been transformed with differencing or logging (either one of the other), 
    this function will untransform them for plotting and metrics.
    
    Inputs:
    - name - str name for model
    - ts - original time series without transformation
    - train - time series train set (can be differenced or logged)
    - test - time series test set (can be differenced or logged)
    - model - trained arima model
    - periods - number of differene (default 0 = none)
    - log - boolean if train & test have been logged
    - library - 'pmdarima' or 'statsmodels'
    
    Outputs:
    - plot of train, test & predictons
    - model summary
    - df containing current and previous metrics
    
    Returns:
    - y_pred - predictions
    '''
    
    # grab model params
    ARIMA_order = model.order
    seasonal_order = model.seasonal_order
    
    # fit statsmodels
    if library == 'statsmodels':
        model = model.fit()
    
    # grab AIC score
    if library == 'pmdarima':
        AIC = model.aic()
    if library == 'statsmodels':
        AIC = model.aic
     
    # grab predictions
    if library == 'pmdarima':
        y_pred = model.predict(n_periods=test.shape[0])
    elif library == 'statsmodels':
        y_pred = model.get_forecast(steps=len(test)).predicted_mean
    else:
        display('Incorrect library entered. Only pmdarima or statsmodels are accepted')
    
    # print arima summary
    display(model.summary())
         
    # function to undifference
    def ts_undiff_plot (ts=ts, train=train, test=test, y_pred=y_pred, periods=periods):

        def inv_diff (df_orig_column, df_diff_column, periods):
            # https://stackoverflow.com/questions/49903037/pandas-reverse-of-diff
            
            # Generate np.array for the diff_inv function - it includes first n values(n = 
            # periods) of original data & further diff values of given periods
            value = np.array(df_orig_column[:periods].tolist()+df_diff_column[periods:].tolist())

            # Generate np.array with inverse diff
            inv_diff_vals = diff_inv(value, periods,1 )[periods:]
            return inv_diff_vals
    
        # create df of original ts, difference and undifference columns
        ts_df = ts.to_frame()
        ts_df['diff'] = ts_df.iloc[:,0].diff(periods=periods)
        ts_df.loc[test.index, 'y_pred'] = y_pred
        ts_df.loc[test.index, 'y_pred_undiff'] = inv_diff(ts_df.loc[test.index][ts_df.columns[0]], ts_df.loc[test.index]['y_pred'], periods=periods)

        # assign undifferenced test and predicitons to variables
        test_undiff = ts_df.loc[test.index, ts_df.columns[0]]
        y_pred_undiff = ts_df.loc[test.index, 'y_pred_undiff']
        
        plt.figure(figsize=(10,6))
        plt.plot(ts_df.loc[train.index][ts_df.columns[0]], label = 'Train')
        plt.plot(ts_df.loc[test.index][ts_df.columns[0]], label = 'Test')
        plt.plot(test.index, ts_df.loc[test.index]['y_pred_undiff'], color='red', label = 'Prediction')
        plt.legend()
        plt.show()
        
        return test_undiff, y_pred_undiff
    
    
    if periods > 0:
        test_undiff, y_pred_undiff = ts_undiff_plot(ts, train, test, y_pred, periods)
        
        # add the row to metrics df and list in reverse order so current is at top
        new_row = []
        new_row.append(name)
        new_row.append(f"{metrics.explained_variance_score(test_undiff, y_pred_undiff)*100:.2f}%")
        new_row.append(f"{metrics.r2_score(test_undiff, y_pred_undiff)*100:.2f}%")
        new_row.append(f"${metrics.mean_absolute_error(test_undiff, y_pred_undiff):,.2f}")
        new_row.append(f"${metrics.mean_squared_error(test_undiff, y_pred_undiff, squared=False):,.2f}")
        new_row.append(f"{ARIMA_order}")
        new_row.append(f"{seasonal_order}")
        new_row.append(f"{AIC:.2f}")
        arima_metrics.loc[len(arima_metrics.index)] = new_row
        display(arima_metrics.sort_index(ascending=False, axis=0)) 
        
        return y_pred_undiff
        
    elif log == True:
        # plot train, test, predictions
        plt.figure(figsize=(10,6))
        plt.plot(np.exp(train), label = 'Train')
        plt.plot(np.exp(test), label = 'Test')
        plt.plot(test.index, np.exp(y_pred), color='red', label = 'Prediction')
        plt.legend()
        plt.show()
        
        # add the row to metrics df and list in reverse order so current is at top
        new_row = []
        new_row.append(name)
        new_row.append(f"{metrics.explained_variance_score(np.exp(test), np.exp(y_pred))*100:.2f}%")
        new_row.append(f"{metrics.r2_score(np.exp(test), np.exp(y_pred))*100:.2f}%")
        new_row.append(f"${metrics.mean_absolute_error(np.exp(test), np.exp(y_pred)):,.2f}")
        new_row.append(f"${metrics.mean_squared_error(np.exp(test), np.exp(y_pred), squared=False):,.2f}")
        new_row.append(f"{ARIMA_order}")
        new_row.append(f"{seasonal_order}")
        new_row.append(f"{AIC:.2f}")
        arima_metrics.loc[len(arima_metrics.index)] = new_row
        display(arima_metrics.sort_index(ascending=False, axis=0))
        
        return y_pred
    
    else:  
        # plot train, test, predictions
        plt.figure(figsize=(10,6))
        plt.plot(train, label = 'Train')
        plt.plot(test, label = 'Test')
        plt.plot(test.index, y_pred, color='red', label = 'Prediction')
        plt.legend()
        plt.show()
        
        # add the row to metrics df and list in reverse order so current is at top
        new_row = []
        new_row.append(name)
        new_row.append(f"{metrics.explained_variance_score(test, y_pred)*100:.2f}%")
        new_row.append(f"{metrics.r2_score(test, y_pred)*100:.2f}%")
        new_row.append(f"${metrics.mean_absolute_error(test, y_pred):,.2f}")
        new_row.append(f"${metrics.mean_squared_error(test, y_pred, squared=False):,.2f}")
        new_row.append(f"{ARIMA_order}")
        new_row.append(f"{seasonal_order}")
        new_row.append(f"{AIC:.2f}")
        arima_metrics.loc[len(arima_metrics.index)] = new_row
        display(arima_metrics.sort_index(ascending=False, axis=0))
        
        return y_pred



########################
#  PHASE 3 - SPOTIFY  #
########################


def crossval(estimator, X, y, cv=5, scoring='precision'):
    '''
    Cross Fold Score with a default of 5 folds and score set to precision
    '''
    cv_scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
    print(f"Avg {scoring.capitalize()} Score of {cv_scores.mean():.4f} with Std Dev of {cv_scores.std():.4f}")
    print('')
    print(f"The scores were: {list(map('{:.4f}'.format,cv_scores))}")



# create dataframe to document scores
eval_metrics = pd.DataFrame(columns = ['Model', 'Precision', 'F1 Score', 'ROC-AUC'])

def evaluate(name, estimator, X_train, X_test, y_train, y_test, use_decision_function='yes'):
    '''
    Evaluation function to show a few scores for both the train and test set
    Also shows a confusion matrix for the test set
    
    use_decision_function allows you to toggle whether you use decision_function or
    predict_proba in order to get the output needed for roc_auc_score
    If use_decision_function == 'skip', then it ignores calculating the roc_auc_score
    
    *Created for Spotify Project*
    '''
    
    # grab predictions
    train_preds = estimator.predict(X_train)
    test_preds = estimator.predict(X_test)
    
    # output needed for roc_auc_score
    if use_decision_function == 'skip': # skips calculating the roc_auc_score
        train_out = False
        test_out = False
    elif use_decision_function == 'yes': # not all classifiers have decision_function
        train_out = estimator.decision_function(X_train)
        test_out = estimator.decision_function(X_test)
    elif use_decision_function == 'no':
        train_out = estimator.predict_proba(X_train)[:, 1] # proba for the 1 class
        test_out = estimator.predict_proba(X_test)[:, 1]
    else:
        raise Exception ("The value for use_decision_function should be 'skip', 'yes' or 'no'.")
      
    # plot test confusion matrix
    plot_confusion_matrix(estimator, X_test, y_test,
                          values_format=",.0f",
                          display_labels = ['Not Popular', 'Popular'])                            
    plt.title('Confusion Matrix (Test Set)')
    plt.show()
    
    # print scores
    print("          | Train  | Test   |")
    print("          |-----------------|")
    print(f"Precision | {precision_score(y_train, train_preds)*100:.2f}% | {precision_score(y_test, test_preds)*100:.2f}% |")
    print(f"F1 Score  | {f1_score(y_train, train_preds)*100:.2f}% | {f1_score(y_test, test_preds)*100:.2f}% |")
    if type(train_out) == np.ndarray:
        print(f"ROC-AUC   | {roc_auc_score(y_train, train_out)*100:.2f}% | {roc_auc_score(y_test, test_out)*100:.2f}% |")
    
    # add the row to metrics df and list in reverse order so current is at top
    new_row = []
    new_row.append(name)
    new_row.append(f"{precision_score(y_test, test_preds)*100:.2f}%")
    new_row.append(f"{f1_score(y_test, test_preds)*100:.2f}%")
    new_row.append(f"{roc_auc_score(y_test, test_preds)*100:.2f}%")

    eval_metrics.loc[len(eval_metrics.index)] = new_row
    display(eval_metrics.sort_index(ascending=False, axis=0))



def plot_feature_imp(estimator, X):
    '''
    Plot feature importance of model
    
    *Created for Spotify Project*
    '''
    
    # capture feature importances
    feats = estimator.feature_importances_
    feature_imps = dict(zip(X.columns, feats))

    # creating list of column names
    feat_names=list(X.columns)

    # Sort feature importances in descending order
    indices = np.argsort(feats)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [feat_names[i] for i in indices]

    # Create plot
    plt.figure(figsize = [8,5])

    # Create plot title
    plt.title("Feature Importance")

    # Add bars
    plt.bar(range(X.shape[1]), feats[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=50, ha = 'right')

    # Show plot
    plt.show()



####################################################
#  PHASE 2 - LINEAR REGRESSION - HOUSE PRICE PRED  #
####################################################



linreg_metrics = pd.DataFrame(columns = ['Model Name', 'R2', 'MAE', 'RMSE']) #df to keep track of metrics for linereg_evaluate

def linreg_evaluate(name, model, df, continuous, categoricals, log=True, OHE=True, scale=True, scaler=MinMaxScaler(), seed=42, print=True):
    '''
    Performs a train-test split & evaluates a model
    
    Returns: model, scaler, y_train_pred, y_test_pred, X_test, y_test, X, y
   
    --
   
    Inputs:
     - name - string, name describing model
     - model - Instantiated sklearn model
     - df - pandas dataframe, containing all independent variables & target
     - continuous - list of all continuous independent variables
     - categoricals - list of all categorical independant variables
     - log - boolean, whether continuous variables should be logged 
     - OHE - boolean, whether categorical variables should be One Hot Encoded
     - scale - boolean, whether to scale the data with a MinMax Scaler
     - scaler - set to MinMaxScaler as default
     - seed - integer, for the random_state of the train test split

    Outputs (if print=True):
     - R2, MAE and RMSE for training and test sets
     - Scatter plot of risiduals from training and test sets
     - Stats model summary of model
     - Metrics Dataframe which lists R2, Mean Absolute Error & Root Mean Square. 
       (This df will be listed in reverse index, with latest results at the top)
        
    Returns:
     - model - fit sklearn model
     - scaler - fit scaler
     - y_train_preds - predictions for the training set
     - y_test_preds - predictions for the test set
     - X_test, y_test - if needed for OLS
     - X, y - if needed for final model
    '''
    
    preprocessed = df.copy()
    
    if log == True:
        pp_cont = preprocessed[continuous]
        log_names = [f'{column}_log' for column in pp_cont.columns]
        pp_log = np.log(pp_cont)    
        pp_log.columns = log_names
        preprocessed.drop(columns = continuous, inplace = True)
        preprocessed = pd.concat([preprocessed['price'], pp_log, preprocessed[categoricals]], axis = 1)
    else:
        preprocessed = pd.concat([preprocessed['price'], preprocessed[continuous], preprocessed[categoricals]], axis = 1)
        
    if OHE == True:
        preprocessed = pd.get_dummies(preprocessed, prefix = categoricals, columns = categoricals, drop_first=True)
 
    # define X and y       
    X_cols = [c for c in preprocessed.columns.to_list() if c not in ['price']]
    X = preprocessed[X_cols]
    y = preprocessed.price
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 
                                                        random_state=seed)
    
    # scale
    if scale == True:
        scaler = scaler
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
    # fit model
    model.fit(X_train, y_train)

    # predict on training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
        
    # calculate residuals
    residuals_train = y_train_pred - y_train
    residuals_test = y_test_pred - y_test
    
    if print == True:
        # print train and test R2, MAE, RMSE
        print(f"Train R2: {r2_score(y_train, y_train_pred):.3f}")
        print(f"Test R2: {r2_score(y_test, y_test_pred):.3f}")
        print("---")
        print(f"Train MAE: {mean_absolute_error(y_train, y_train_pred):.2f}")
        print(f"Test MAE: {mean_absolute_error(y_test, y_test_pred):.2f}")
        print("---")
        print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.2f}")
        print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")
                    
        # risduals plot training and test predictions
        plt.figure(figsize=(7,5))
        plt.scatter(y_train_pred, residuals_train, alpha=.75, label = "Train")
        plt.scatter(y_test_pred, residuals_test, color='g', alpha=.75, label = "Test")
        plt.axhline(y=0, color='black')
        plt.legend()
        plt.title(f'Residuals for {name}')
        plt.ylabel('Residuals')
        plt.xlabel('Predicted Values')
        plt.show()
    
        # display feature weights using ELI5
        display(eli5.show_weights(model, feature_names=X_cols))
        
        # add name, metrics and description to new row
        new_row = []
        new_row.append(name)
        new_row.append(format(r2_score(y_test, y_test_pred),'.3f'))
        new_row.append(format(mean_absolute_error(y_test, y_test_pred),'.2f'))
        new_row.append(format(np.sqrt(mean_squared_error(y_test, y_test_pred)),'.2f'))
    
        # add the row to metrics df and list in reverse order so current is at top
        linreg_metrics.loc[len(linreg_metrics.index)] = new_row
        display(linreg_metrics.sort_index(ascending=False, axis=0))

    return model, scaler, y_train_pred, y_test_pred, X_test, y_test, X, y