import pandas as pd
import statsmodels.api as sm


def vif_cal(input_data, dependent_col):
    vif_df = pd.DataFrame(columns=['Var', 'Vif'])
    x_vars = input_data.drop([dependent_col], axis=1)
    xvar_names = x_vars.columns

    for i in range(0, xvar_names.shape[0]):
        y = x_vars[xvar_names[i]]
        x = x_vars[xvar_names.drop(xvar_names[i])]
        rsq = sm.OLS(y, x).fit().rsquared
        vif = round(1 / (1 - rsq), 2)
        vif_df.loc[i] = [xvar_names[i], vif]

    return vif_df.sort_values(by='Vif', axis=0, ascending=False, inplace=False)
