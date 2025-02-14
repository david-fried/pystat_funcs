import itertools
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats as stats
from scipy.stats import pearsonr, linregress, chi2_contingency
import seaborn as sns
import statsmodels as sm


def get_descriptives(df, table_color=None): #if table_color is specified returns colored styler object, else return pandas dataframe

    df = df.describe()

    df.index = [i.title() for i in df.index]

    df.rename(index={'Std': 'SD', 'Count': 'N'}, inplace=True)

    df = df.T

    if table_color is not None:

        df = df.style.format('{:.1f}', subset=['Mean', 'SD']).format('{:.0f}', subset=['N', 'Min', '25%', '50%', '75%', 'Max'])

        table_format = []

        table_format.append({'selector': 'thead',

          'props': [('background-color', table_color)]})

        table_format.append({'selector': 'tbody tr:nth-child(even)',

          'props': [('background-color', table_color)]})

        table_format.append({'selector': 'tbody tr:nth-child(odd)',

          'props': [('background-color', 'white')]})

        df = df.set_table_styles(table_format)

    else:

        for col in df.columns:

            if col not in ('Mean', 'SD'):

                df[col] = df[col].astype('int64')

            else:

                df[col] = df[col].round(1)

    return df
 

def cronbach_alpha(scale: pd.DataFrame, identify_optimal: bool = False, display_top=None): # dictionary (identify_optimal=False) or pd.DataFrame(identify_optimal=True)

 
    # Confirmed with R Code

    def compute_alpha(df):

        df = df.dropna()

        num_items = df.shape[1]

        sum_of_item_variances = df.var(axis=0).sum()

        variance_of_sum_of_items = df.sum(axis=1).var()

        alpha = num_items/(num_items - 1)*(1 - sum_of_item_variances/variance_of_sum_of_items)

        alpha_stats = {'alpha': alpha, 'sample': len(df), 'items': num_items}

        return alpha_stats

 

    if identify_optimal==False:

        return compute_alpha(scale)

 

    n_items = len(scale.columns)

 

    if n_items < 3:

        raise Exception ('Must have at least three items to identify optimal alpha.')

 

    alpha_all_items = compute_alpha(scale)

    alpha_all_items['items_dropped'] = 'None'

    cronbach_alphas = [alpha_all_items]

 

    for i in range(1, len(scale.columns)-2):

 

        revised_n_items = n_items - i

 

        combinations = list(itertools.combinations(tuple(scale.columns), revised_n_items))

 

        tables = []

       

        for items in combinations:

            revised_scale = scale.copy()

            revised_scale = revised_scale[list(items)]

            alpha_stats = compute_alpha(revised_scale)

            alpha_stats['items_dropped'] = (', ').join(list(set(scale.columns) - set(items)))

            cronbach_alphas.append(alpha_stats)

 

        tables.append(pd.DataFrame(cronbach_alphas))

 
#     highlighted_rows = (all_alphas['alpha'] == all_alphas['alpha'].max()).map({ True: 'background-color: rgba(241,235,156)', False: 'background-color: white'})

   

#     all_alphas = all_alphas.style.apply(lambda _: highlighted_rows)

 

    if display_top is not None:

 

        all_alphas = pd.concat(tables).sort_values(by='alpha', ascending=False)

       

        return all_alphas.head(display_top)

   

    return all_alphas

 
def alpha_if_item_removed(scale: pd.DataFrame) -> pd.DataFrame:

    alpha_all = cronbach_alpha(scale)

    alpha_all['item_dropped'] = 'None'

    cronbach_alphas = [alpha_all]

    for item in scale.columns:

        scale_with_item_dropped = scale.copy()

        scale_with_item_dropped.drop(columns=item, inplace=True)

        alpha_stats = cronbach_alpha(scale_with_item_dropped)

        alpha_stats['item_dropped'] = item

        alpha_stats['items'] = scale_with_item_dropped.shape[1]

        cronbach_alphas.append(alpha_stats)

    return pd.DataFrame(cronbach_alphas)

 

# Correlations

def display_corrs(df, fontsize=10, figsize=(10,10), labelsize=10):

    corr = df.corr()

    # plt.tick_params(labelbottom=False, labelsize=12, bottom=False, labeltop=True, top=True)

    plt.tick_params(labelbottom=False, labeltop=True, labelsize=labelsize)

    cmap = sns.diverging_palette(240, 240, as_cmap=True)

    # sns.set_context("talk")

    # sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})

    annot_kws={'fontsize':fontsize}

    # 'color':"k",'alpha':0.6,'rotation':"vertical",'verticalalignment':'center', #'backgroundcolor':'w'}

    sns.set(rc = {'figure.figsize':figsize})

    sns.heatmap(corr, annot=True, linewidths=.5, fmt='.2f', cmap=cmap, vmin = -1, vmax = 1, annot_kws=annot_kws)

    plt.show()

 

def get_corrs(df, fontsize=10, figsize=(10,10), labelsize=10):

    corr = df.corr()

    # plt.tick_params(labelbottom=False, labelsize=12, bottom=False, labeltop=True, top=True)

    plt.tick_params(labelbottom=False, labeltop=True, labelsize=labelsize)

    cmap = sns.diverging_palette(240, 240, as_cmap=True)

    # sns.set_context("talk")

    # sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})

    annot_kws={'fontsize':fontsize}

    # 'color':"k",'alpha':0.6,'rotation':"vertical",'verticalalignment':'center', #'backgroundcolor':'w'}

    sns.set(rc = {'figure.figsize':figsize})

    return sns.heatmap(corr, annot=True, linewidths=.5, fmt='.2f', cmap=cmap, vmin = -1, vmax = 1, annot_kws=annot_kws)

 

def corr_pvalues(df):

    """

    Pass in a dataframe and function returns matrix of p values for correlations

    """

    df = df._get_numeric_data()

    dfcols = pd.DataFrame(columns=df.columns)

    pvalues = dfcols.transpose().join(dfcols, how='outer')

    for r in df.columns:

        for c in df.columns:

            if c == r:

                df_corr = df[[r]].dropna()

            else:

                df_corr = df[[r,c]].dropna()

            pvalues[r][c] = pearsonr(df_corr[r], df_corr[c])[1]

    return pvalues

             

 

def corr_flagsig(df, symbol='---', hide_ns=False):

 
    """

    Pass in a dataframe and function will return a correlation matrix with significant correlations flagged with asterisk.

    """

   

    columns = list(df.columns)

   

    n = len(columns)

   

    corr_matrix = df.corr()

 

    p_matrix = corr_pvalues(df)

 

    new_matrix =[]

 

    for i in range(n):

       

        new_row = []

       

        for j in range(n):

           

            r = corr_matrix.iloc[i,j]

            p = p_matrix.iloc[i,j]

           

            r = f'{r:.2f}'

            if '0.' in r:

                r = r.replace('0.', '.')

            elif '1.00' in r:

                r = r.replace('1.00', '1')

            if r != '1':

                if p <= .05:

                    if '--' not in symbol:

                        r = r + symbol

                else:

                    if hide_ns:

                        r = ''

                    elif '--' in symbol:

                        r = symbol

            new_row.append(r)

           

        new_matrix.append(new_row)

 

 

    df = pd.DataFrame(new_matrix, columns=columns, index=columns)

   

    return df
        

def compute_corr(xy):

    """

    pass in pd Dataframe with two columns or list of two pd Series

    """

    if isinstance(xy, pd.core.frame.DataFrame):

        columns = list(xy.columns)

        col1 = columns[0]

        col2 = columns[1]

    else:

        col1 = xy[0].name

        col2 = xy[1].name

        xy = pd.concat([xy[0], xy[1]], axis=1)

    df_noNA = xy[(~xy[col1].isnull()) & (~xy[col2].isnull())]

    n = len(df_noNA)

    x = df_noNA[col1]

    y = df_noNA[col2]

    corr, sig = pearsonr(x, y)

    return corr, sig, n

 

def avg_iic(corr_matrix):

    corr_sum = ((corr_matrix.sum()-1)/2).sum()

    n_corrs = (len(corr_matrix)**2 - len(corr_matrix))/2

    return corr_sum/n_corrs

 

def corr_onlysigs(df, columsX, columnsY, correlations=True):

   

    """

    Pass in a dataframe, list of columns, and a list of columns

    Default: correlations==True: Will return correlations ('ns' stands for not significant)

    If False, will return sample sizes

    """

   

    values_dict = {}

 

    for col1 in columsX:

        values = []

        for col2 in columnsY:

            d = df[[col1, col2]]

            d.dropna(inplace=True)

 

            r, p, n = compute_corr(d)

            if correlations:

                r = f'{r:.2f}'

                if p > .05:

                    #r = r + '(ns)'

                    r='---'

                r = r.replace('0.', '.')

                values.append(r)

            else:

                values.append(n)

               

        values_dict[col1] = values

 

    df_ = pd.DataFrame(values_dict, index=columnsY)

   

    return df_

 

def common_obs(df):

    not_null_locations = pd.notnull(df).values.astype(int)

    common_obs = pd.DataFrame(not_null_locations.T.dot(not_null_locations),

                              columns=df.columns, index=df.columns)

    return common_obs

 

def corr_n_range(corr_matrix):

    txt = 'n='

    mins = common_obs(corr_matrix).min()

    maxes = common_obs(corr_matrix).max()

 

    if all(mins==maxes):

        return common_obs(corr_matrix).min()[0]

    else:

        mi = min(mins)

        m = max(maxes)

        return (mi, m)

 

# Effect sizes

def cohens_d(array1, array2):

    # Cohen's d is the appropriate effect size measure if two groups have similar standard deviations and are of the same size.

    mean1 = array1.mean()

    mean2 = array2.mean()

    sd_pooled = ((array1.var() + array2.var())/2)**0.5

    d = (mean1 - mean2)/sd_pooled

    return d

 

def glasses_delta(array1, array2):

    """

    Glass's delta, which uses only the standard deviation of the control group, is an alternative measure if each group has a different standard deviation.

    If you intend to report Glass's delta, then you need to enter your control group values as Group 1

    """

    mean1 = array1.mean()

    mean2 = array2.mean()

    sd_pooled = array1.std()

    delta = (mean1 - mean2)/sd_pooled

    return delta

 

def hedges_g(array1, array2):

    """

    Hedges' g, which provides a measure of effect size weighted according to the relative size of each sample,

    is an alternative where there are different sample sizes.

    (This is important! If you've got different sample sizes then you should use Hedges' g.)

    """

    mean1 = array1.mean()

    mean2 = array2.mean()

    n1 = len(array1)

    n2 = len(array2)

    sd_pooled = ((array1.var()*n1 + array2.var()*n2)/(n1 + n2))**0.5

    g = (mean1 - mean2)/sd_pooled

    return g

 

# t test

def levene(array1, array2):

    return stats.levene(array1, array2, center='mean')

 

def t_stats(df1, df2, df1_name, df2_name, stylized=True, round_two=True):

   

    """

    df1 and df2 should have same columns

    """

   

    if list(df1.columns) != list(df2.columns):

        raise('Columns do not match.')

 

    matrix = []

 

    for col in df1.columns:

        array1 = df1[col].dropna()

        array2 = df2[col].dropna()

        df1_n = len(array1)

        df2_n = len(array2)

        df1_sd = array1.std()

        df2_sd = array2.std()

        df1_mean = array1.mean()

        df2_mean = array2.mean()

        diff = df1_mean - df2_mean

        sd_pooled = ((array1.var() + array2.var())/2)**0.5

        l = levene(array1, array2)

        l_stat = l[0]

        l_sig = l[1]

        ev = False

        if l_sig > .05:

            ev = True

        t_test = stats.ttest_ind(array1, array2, equal_var=ev)

        t = t_test[0]

        p = t_test[1]

        d = cohens_d(array1, array2)

        g = glasses_delta(array1, array2)

        h = hedges_g(array1, array2)

        matrix.append([col, df1_n, df2_n, df1_mean, df2_mean, diff, sd_pooled, d, t, p, g, h, df1_sd, df2_sd, l_stat, l_sig])

 

    cols = ['Item', f'{df1_name} N', f'{df2_name} N', f'{df1_name} Mean', f'{df2_name} Mean', 'Mean Diff', 'SD Pooled', 'Cohen\'s D', 'T-Stat', 'P Value', 'Glass\'s Delta', 'Hedges\' G', f'{df1_name} SD', f'{df2_name} SD', 'Levene\'s Stat', 'Levene Sig']

 

    t_df = pd.DataFrame(matrix, columns=cols)

 

    if not stylized:

 

        return t_df

 

    two_decimal_cols = cols[3:]

    two_decimal_cols.remove('P Value')

    two_decimal_cols.remove('Levene Sig')

    highlighted_rows = (t_df['P Value'] <= 0.05).map({ True: 'background-color: rgba(241,235,156)', False: 'background-color: white'})

    bolded_ps = (t_df['P Value'] <= 0.05).map({ True: 'font-weight: bold', False: ''})

    if round_two:

        styled_df = t_df.style.format('{:.2f}', subset=two_decimal_cols).format('{:.4f}', subset=['P Value', 'Levene Sig']).apply(lambda _: bolded_ps, subset=['Item', 'P Value'])

    else:

        styled_df = t_df.style.format('{:.12f}', subset=two_decimal_cols).format('{:.4f}', subset=['P Value', 'Levene Sig']).apply(lambda _: bolded_ps, subset=['Item', 'P Value'])

    table_format = [

        {'selector': 'thead',

        'props': [('background-color', 'rgb(241,235,156)')]},

        {'selector': 'td:nth-child(8)',

        'props': [('width', '250px')]},

        {'selector': 'th:nth-child(8)',

        'props': [('text-align', 'center')]}

        ]

    styled_df = styled_df.set_table_styles(table_format)

    styled_df = styled_df.hide_index()

    styled_df = styled_df.bar('Cohen\'s D', color='#d65f5f', align='zero', vmin=-1.25, vmax=1.25, width=100)

 

    return styled_df

                            

def group_means(df, grouping_var, grand=True, grand_name='Grand Average', decimals=2, transpose=True):

    df = df.copy()

    df_mean_by_group = df.groupby(grouping_var).mean()

    if grand:

        df_grand = df.select_dtypes(np.number).mean().rename(grand_name)

        df_mean_by_group = df_mean_by_group.append(df_grand)

    output = df_mean_by_group.apply(lambda x: round(x,decimals))

    if transpose:

        output = output.T

    return output

             

def chi_square_stats(series1, series2):

    contingency_no_margins = pd.crosstab(series1, series2)

    chi2, p_value, dof, exp_freqs = chi2_contingency(contingency_no_margins)

    return {'ChiSquare': chi2, 'P Value': p_value, 'Df': dof, 'Expected Frequencies': exp_freqs}

 

def contingency_table(series1, series2, percent=False, title=None, figsize=(12,8)):

    x2 = chi_square_stats(series1, series2)

    x2_stat = round(x2['ChiSquare'], 4)

    df = x2['Df']

    p = round(x2['P Value'], 4)

    text = f'Chi Square Statistic: {x2_stat}\nDegrees of Freedom: {df}\nP Value = {p}'

    plt.figure(figsize=figsize).text(.125,.9, text, fontsize=14)

    if percent:

        contingency = pd.crosstab(series1, series2, normalize='index')

        fmt='.2f'

    else:

        contingency = pd.crosstab(series1, series2)

        fmt='d'

    xlabel = contingency.columns.name

    ylabel =  contingency.index.name

    xticklabels = list(contingency.columns)

    yticklabels = list(contingency.index)

    p=sns.heatmap(contingency, annot=True, cmap='YlGnBu', fmt=fmt) # use fmt='d' for integers

    p.set_title(title, fontsize=14, pad=30)

    p.set_xlabel(xlabel, fontsize=14, labelpad=30)

    p.set_xticklabels(xticklabels, fontsize=14)

    p.set_ylabel(ylabel, fontsize=14, labelpad=30)

    p.set_yticklabels(yticklabels, fontsize=14)

    plt.show()

 

# Regression

def regression_png(x, y, x_label, y_label, size = (10, 10), text_loc=(50, -10)):

    title = f'Regressing {y_label} on {x_label}'

    (slope, intercept, rvalue, pvalue, stderr) = linregress(x, y)

    regress_values = x * slope + intercept

    line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))

    sns.regplot(x, y)

    sns.set(rc={'figure.figsize': size})

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.title(title)

    text = f"r = {pearsonr(x, y)[0]:.2f}, p = {pearsonr(x, y)[1]:.4f}, r_squared = {rvalue**2:.2f}, n = {len(y)}"

    plt.text(text_loc[0], text_loc[1], text)

    fname = title.replace(':\n', '_').replace(' ', '_') + '.png'

    plt.savefig(fname)

 

  

#Features: Provides stats and statistical model test

def multiple_regression(xs, y): # xs is a dataframe, y is a series; prints output and returns predictions

    xs = sm.add_constant(xs) # adding a constant

    model = sm.OLS(y, xs).fit()

    predictions = model.predict(xs)

    print_model = model.summary()

    print(print_model)

    return predictions

 

def regression_3d_visualization(dataframe, x1, x2, y, size = (10,10), x1label='X Label', x2label='Y Label', ylabel='Z Label'):

    df = dataframe[[x1, x2, y]].reset_index(drop=True)

    x1r, x2r, yr = x1, x2, y

    if ' ' in x1:

        x1r = x1.replace(' ', '')

        df.rename(columns={x1: x1r}, inplace=True)

    if ' ' in x2:

        x2r = x2.replace(' ', '')

        df.rename(columns={x2: x2r}, inplace=True)

    if ' ' in y:

        yr = y.replace(' ', '')

        df.rename(columns={y: yr}, inplace=True)

    model = smf.ols(formula=f'{yr} ~ {x1r} + {x2r}', data=df)

    results = model.fit()

    results.params

 

    x_dim, y_dim = np.meshgrid(np.linspace(df[x1r].min(), df[x1r].max(), 100), np.linspace(df[x2r].min(), df[x2r].max(), 100))

    xs = pd.DataFrame({x1r: x_dim.ravel(), x2r: y_dim.ravel()})

    predicted_y = results.predict(exog=xs)

    predicted_y=np.array(predicted_y)

 

    fig = plt.figure(figsize=size, facecolor='b')

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df[x1r], df[x2r], df[yr], c='red', marker='o', alpha=0.5)

    ax.plot_surface(x_dim, y_dim, predicted_y.reshape(x_dim.shape), color='b', alpha=0.3)

    ax.set_xlabel(x1label)

    ax.set_ylabel(x2label)

    ax.set_zlabel(ylabel)

    plt.show()

 

def corrColor(x):

    try:

        val = float(x)

    except ValueError:

        return None

    absval = abs(val)*.75

    if val < 0:

        return f'background-color: rgba(255,0,0,{absval})'

    return f'background-color: rgba(0,0,255,{absval})'

 

 

corr_table_css = """

<style>

    .my_table {

    border-collapse: collapse;

    border: none;

    table-layout: fixed;

    font-size: 1em;

    line-height: 25px;

   

    }

    .my_table th {

    position: sticky;

    top: 0;

    background-color: white;

    z-index: 2;

    }

   

    .my_table td:first-child {

    position: sticky;

    left: 0;

    background-color: white;

    z-index: 1;

    text-align: right;

    font-weight: bold;

    min-width: 300px

    }

 

    .my_table th:first-child {

    left: 0;

    z-index: 3;

    text-align: right;

    font-weight: bold;

    min-width: 300px

    }

   

    .my_table th:not(:first-child), .my_table td:not(:first-child){

    min-width: 35px;

    max-width: 35px;

    text-align: right;

    }

   

    .my_table th:not(:first-child){

        text-align: center;

        }

       

    .my_table td:not(:first-child){

        padding-right: 25px;

        }

</style>

"""

 

stickytop_table_css = """

<style>

    .my_table th {

    position: sticky;

    top: 0;

    background-color: white;

    z-index: 2;

    }

</style>

"""