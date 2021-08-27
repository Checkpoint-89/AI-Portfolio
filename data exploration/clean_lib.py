# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:31:57 2020

@author: cdiet
"""
import numpy as np
import pandas as pd

DEBUG = False

def get_var_info(data, how_few = 0, what = 'all'):
    
    from numpy import percentile 
    
    # Define var_info
    var_info = pd.DataFrame(columns = ['var_rank', 'var_name','uniques', 'uniques_rate', 'size',\
                                       'count', 'nans', 'nan_rate', 'check', 'dtype', 'type', 'type_guess',\
                                       'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurtosis'])
    
    # Assert whether there are duplicated variable names
    dups = data.columns.duplicated()
    dups = data.columns[dups]
    assert len(dups) == 0, f"Function 'get_var_info': duplicated variable names\n{list(dups)}" 
    
    #Loop over the variables
    for i, c in enumerate(data.columns):
        
        d = data[c]
        
        # Get number of unique values, size (which includes NaN values), and count of non-NaN values
        uniques = d.nunique()
        size = d.size
        count = d.count()
        
        # Calculate ratio of uniques values with regards to total values excl. NaN
        if count != 0:
            uniques_rate = uniques / count
        else:
            uniques_rate = np.nan
        nan = d.isna().sum()
        
         # Calculate ratio of NaN values with regards to total values incl. NaN       
        if size != 0:
            nan_rate = nan / size
        else:
            nan_rate = np.nan
        check = (size ==  count + nan)
        
        # Get the type of the values
        dtype = data[c].dtype
        
        # Characterize the type of the values: Numerical or Qualitative
        if nan_rate == 1:
            typ = np.nan
        elif dtype != 'object':
            typ = 'Numerical'
        else:
            typ = 'Qualitative'

        # If Numerical, calculate basic statistics
        if typ == 'Numerical':
            d_notna = d[d.notna()]
            mean = d.mean()
            median = d.median()
            std = d.std()
            mini = d.min()
            p25 = percentile(d_notna,25)
            p50 = percentile(d_notna,50)
            p75 = percentile(d_notna,75)
            maxi = d.max()
            skew = d.skew()
            kurtosis = d.kurtosis()
        else:
            mean, median, std, mini, p25, p50, p75, maxi, skew, kurtosis = (np.nan,) * 10
        
        # Fill output
        var_info.loc[i] = [i, c, uniques, uniques_rate, size, count, nan, nan_rate, check, dtype, typ, typ,\
                           mean, median, std, mini, p25, p50, p75, maxi, skew, kurtosis]

    # Guess whether Qualitative variables are Categorical
    if how_few > 0:
        var_names = clean_variables(data, 'few_uniques', var_info = var_info, drop = False, how_few = how_few, typ = None, verbose = False)
        var_info.loc[var_info['var_name'].isin(var_names), 'type_guess'] = 'Categorical'
        
    if what == 'all':
        return(var_info)
    
    if what == 'variables':
        var_info = var_info[['var_name','uniques', 'uniques_rate','size',\
                                       'count', 'nans', 'nan_rate', 'dtype', 'type', 'type_guess']]
    if what == 'stats':
        var_info = var_info.loc[var_info['type'] == 'Numerical', ['var_name', 'count', 'mean', 'median', 'std', 'min',\
                                 '25%', '50%', '75%', 'max', 'skew', 'kurtosis']]
    return(var_info)


def drop_variables(data, to_drop):
    
    print(f"\nShape before drop: {data.shape}")
    d = data.drop(to_drop, axis = 1)
    print(f"Shape after drop: {d.shape}")
    return(d)


def clean_variables(data, how, *args, **kwargs):
    
    d = data.copy()
    
    how_l = ['few_notna', 'one_unique', 'few_uniques', 'low_variance', 'extrems', 'missing', 'pareto', 'pareto_multiplex']
    assert how in how_l, f"Function 'clean_variables': invalid argument 'how' = {how}"
    
    if how == 'few_notna':
        var_info = kwargs['var_info']
        how_few = kwargs['how_few']
        drop = kwargs['drop']
        typ = kwargs['typ']
        verbose = kwargs['verbose']
        d = cv_few_notna(d, var_info, drop, how_few, typ, verbose)
    
    if how == 'one_unique':
        var_info = kwargs['var_info']
        drop = kwargs['drop']
        d = cv_one_unique(d, var_info, drop)
        
    if how == 'few_uniques':
        var_info = kwargs['var_info']
        how_few = kwargs['how_few']
        drop = kwargs['drop']
        typ = kwargs['typ']
        verbose = kwargs['verbose']
        d = cv_few_uniques(d, var_info, drop, how_few, typ, verbose)
    
    if how == 'low_variance':
        var_info = kwargs['var_info']
        how_low = kwargs['how_low']
        drop = kwargs['drop']
        d = cv_low_variance(d, var_info, drop, how_low)
        
    if how == 'extrems':
        var_info = kwargs['var_info']
        limits = kwargs['limits']
        correct =  kwargs['correct']
        correction = kwargs['correction']
        d = cv_extrems(d, var_info, limits, correct, correction)  
        
    if how == 'missing':
        var_info = kwargs['var_info']
        variables = kwargs['variables']
        correct =  kwargs['correct']
        correction = kwargs['correction']
        d = cv_missing(d, var_info, variables, correct, correction)   
    
    if how == 'pareto':
        variables = kwargs['variables']
        th = kwargs['th']
        cat_name = kwargs['cat_name']
        verbose = kwargs['verbose']
        d = cv_pareto(d, variables, th, cat_name, verbose)

    if how == 'pareto_multiplex':
        variables = kwargs['variables']
        th = kwargs['th']
        cat_name = kwargs['cat_name']
        verbose = kwargs['verbose']
        d = cv_pareto_multiplex(d, variables, th, cat_name, verbose)
        
    return(d)


def cv_few_notna(data, var_info, drop = False, how_few = 0.01, typ = None, verbose = False):
    
    def p_cv_few_notna(verbose = False):
        
        if verbose == False:
            return()
        
        print("\n")
        print("*"*30)
        print("'cv_few_notna' report")
        print("\nThose are the specific inputs you have passed to this function, check it:")
        print(f"'drop' = {drop}")
        print(f"'how_few' = {how_few}")
        print(f"'typ' = {typ}")
        print(f"'verbose' = {verbose}")
     
        if typ == None:
            print(f"\n'cv_few_notna' handles Qualitative AND Numerical variables")
        if typ == 'Qualitative':
            print(f"\n'cv_few_notna' handles Qualitative variables ONLY")
        if typ == 'Numerical':
            print(f"\n'cv_few_notna' handles Numerical variables ONLY") 
            
        if (typ == None) or (typ == 'Qualitative'):
            print(f"\n# of qualitative variables where the filling rate is <= {how_few * 100}% : {len(to_drop_qual)}")
            print(to_drop_qual)
        if (typ == None) or (typ == 'Numerical'):
            print(f"\n# of numerical variables where the filling rate is <= {how_few * 100}% : {len(to_drop_num)}")
            print(to_drop_num)
            
        if drop == True:
            print(f"\n'cv_few_notna' has dropped the targeted (qualitative and / or numerical) variables with a low filling rate")
            print(f"new shape is: {d.shape}")     
            print("\n'cv_few_notna' end of report")
            print("*"*30)
        else:
            print(f"\nAs requested, no variable was dropped, shape remains: {data.shape}")
            print("\n'cv_few_notna' end of report")
            print("*"*30)
            
    typ_l = ['Qualitative', 'Numerical', None]
    assert typ in typ_l, f"Function 'cv_few_notna': invalid argument 'typ' = {typ}"
    
    to_drop_qual = [var_info.loc[i,'var_name'] for i in var_info.index\
               if (var_info.loc[i,'nan_rate'] > (1-how_few)) and (var_info.loc[i,'type'] == 'Qualitative')]
    
    to_drop_num = [var_info.loc[i,'var_name'] for i in var_info.index\
               if (var_info.loc[i,'nan_rate'] > (1-how_few)) and (var_info.loc[i,'type'] == 'Numerical')]
    
    if typ == None:
        to_drop = to_drop_qual + to_drop_num

    if typ == 'Qualitative':
        to_drop = to_drop_qual
        
    if typ == 'Numerical':
        to_drop = to_drop_num
    
    if drop == True:
        d = drop_variables(data, to_drop)
        p_cv_few_notna(verbose)
        return(d, to_drop)
    else:
        d = data
        p_cv_few_notna(verbose)
        return(d, to_drop)    
    

def cv_one_unique(data, var_info, drop = False):
    
    to_drop = [var_info.loc[i,'var_name'] for i in var_info.index if var_info.loc[i,'uniques'] in [0,1]]
    void = [var_info.loc[i,'var_name'] for i in var_info.index if var_info.loc[i,'uniques'] == 0]

    print("\n")
    print("*"*30)
    print("'cv_one_unique' report")
    print("\nThose are the specific inputs you have passed to this function, check it:")
    print(f"'drop' = {drop}")
    print(f"\n# of variables with one unique value: {len(to_drop)}")
    print(to_drop)
    
    print(f"\nOut of which {len(void)} fully empty")
    print(void)
    
    if drop == True:
        d = drop_variables(data, to_drop)
        print(f"\n'cv_one_unique' has dropped the variables with only one unique value, new shape is: {d.shape}")
        print("\n'cv_one_unique' end of report")
        print("*"*30)
        return(d, to_drop)
    else:
        d = data
        print(f"\nAs requested, no variable was dropped, shape remains: {data.shape}")
        print("\n'cv_one_unique' end of report")
        print("*"*30)
        return(d, to_drop)



    
def cv_few_uniques(data, var_info, drop = False, how_few = 0.01, typ = None, verbose = False):
    
    def p_cv_few_uniques(verbose = False):
        
        if verbose == False:
            return()
        
        print("\n")
        print("*"*30)
        print("'cv_few_uniques' report")
        print("\nThose are the specific inputs you have passed to this function, check it:")
        print(f"'drop' = {drop}")
        print(f"'how_few' = {how_few}")
        print(f"'typ' = {typ}")
        print(f"\n# of qualitative variables where #uniques/#values is less than {how_few * 100}% : {len(few_cat)}")
        print(few_cat)
        print(f"\n# of numerical variables where #uniques/#values is less than {how_few * 100}% : {len(few_num)}")
        print(few_num)
        
        if typ == None:
            print(f"\n'cv_few_uniques' impacts Qualitative AND Numerical variables")
        if typ == 'Qualitative':
            print(f"\n'cv_few_uniques' impacts Qualitative variables ONLY")
        if typ == 'Numerical':
            print(f"\n'cv_few_uniques' returns impacts Numerical variables ONLY") 
            
        if drop == True:
            print(f"\n'cv_few_uniques' has dropped the targeted (qualitative and / or numerical) low variance variables")
            print("new shape is: {d.shape}")     
            print("\n'cv_few_uniques' end of report")
            print("*"*30)
        else:
            print(f"As requested, no variable was dropped, shape remains: {data.shape}")
            print("\n'cv_few_uniques' end of report")
            print("*"*30)
            
    typ_l = ['Qualitative', 'Numerical', None]
    assert typ in typ_l, f"Function 'cv_few_uniques': invalid argument 'typ' = {typ}"
    
    few_cat = [var_info.loc[i,'var_name'] for i in var_info.index\
               if (var_info.loc[i,'uniques_rate'] < how_few) and (var_info.loc[i,'type'] == 'Qualitative')]
    
    few_num = [var_info.loc[i,'var_name'] for i in var_info.index\
               if (var_info.loc[i,'uniques_rate'] < how_few) and (var_info.loc[i,'type'] == 'Numerical')]
    
    if typ == None:
        few_return = few_cat + few_num

    if typ == 'Qualitative':
        few_return = few_cat
        
    if typ == 'Numerical':
        few_return = few_cat
    
    if drop == True:
        d = drop_variables(data, few_return)
        p_cv_few_uniques(verbose)
        return(d)
    else:
        d = few_return
        p_cv_few_uniques(verbose)
        return(d)


def cv_low_variance(data, var_info, drop = False, how_low = 0.01):
    

    low_num = [var_info.loc[i,'var_name'] for i in var_info.index\
               if (var_info.loc[i,'std'] < how_low) and (var_info.loc[i,'type'] == 'Numerical')]
     
    print("\n")
    print("*"*30)
    print("'cv_low_variance' report")
    print(f"'drop' = {drop}")
    print(f"'how_low' = {how_low}")
    print(f"\n# of numerical variables with a std less than {how_low} : {len(low_num)}")
    print(low_num)

    if drop == True:
        d = drop_variables(data, low_num)
        print(f"\n'cv_low_variance' has dropped the targeted low variance variables, new shape is: {d.shape}")
        print("\n'cv_low_variance' end of report")
        print("*"*30)
        return(d)
    else:
        d = low_num
        print(f"\n'cv_low_variance' returns the names of the targeted low variance variables")
        print(f"As requested, no variable was dropped, shape remains: {data.shape}")
        print("\n'cv_few_uniques' end of report")
        print("*"*30)
        return(d)

def cv_extrems(data, var_info, limits, correct, correction):

    assert correct in [True, False], f"Function 'cv_extrems': invalid argument 'correct' = {correct}"
    for i, lim in enumerate(limits):
        assert len(lim) == 4, f"Function 'cv_extrems':\
        #{i}th element of argument 'limits' has {len(lim)} element(s), 3 expected"
        assert lim[0] in var_info['var_name'].values, f"Function 'cv_extrems':\
        #{i}th element of argument 'limits' has an invalid argument '{lim[0]}'"
        if type(lim[1]) != str and type(lim[2]) != str:
            assert lim[1] <= lim[2], f"Function 'cv_extrems':\
            #{i}th element of argument 'limits' is invalid: v1 = {lim[1]} > v2 = {lim[2]}, should be v1 <= v2"
        assert lim[3] in {'inclusive', 'exclusive'}, f"Function 'cv_extrems':\
        #{i}th element of argument 'limits' is invalid: it is '{lim[3]}'; should be either 'inclusive' or 'exclusive'"
     
    def p_cv_extrem():
        print("\n")
        print("*"*30)
        print("'cv_extrems' report")
        print("\nThose are the specific inputs you have passed to this function, check it:")
        print(f"'limits' = {limits}")
        print(f"'correct' = {correct}")
        print(f"'correction' = {correction}") 
        print(f"\nFor each variable, this is the number of extreme data to correct\n")
        print(log)
        
        if correct == True:
            if correction == 'drop':
                print(f"\nThe corresponding lines were all dropped")
            else:
                print(f"\nThose values were all replaced by '{correction}'")
            print(f"\n'cv_extrems' returns the corrected DataFrame and the above log")
        else:
            print(f"\nAs requested, no variable was corrected")
            print(f"\n'cv_extrems' returns the same DataFrame and the above log")
 
        if DEBUG == True:
            print(f"\nSanity check: the same log would now look as such:\n")
            print(log2)
            
        print("\n'cv_extrems' end of report")
        print("*"*30)
            
    def cv_extrem_main(data, limits, log, correct):
        for i, (var_name, inf, sup, typ) in enumerate(limits):
            if type(inf) == str:
                inf = data[inf]
            if type(sup) == str:
                sup = data[sup]
            if typ == 'exclusive':
                ind_inf = data[var_name] <= inf
                ind_sup = data[var_name] >= sup
            else:
                ind_inf = data[var_name] < inf
                ind_sup = data[var_name] > sup
            nb_inf = ind_inf.sum()
            nb_sup = ind_sup.sum()
            nb_total = nb_inf + nb_sup
            log.loc[i] = [var_name, nb_inf, nb_sup, nb_total]
            if correct == True:
                if correction == 'drop':
                    ind = ind_inf | ind_sup
                    ind = data[ind].index
                    data.drop(ind, inplace = True)
                else:
                    data.loc[ind_inf, var_name ] = correction
                    data.loc[ind_sup, var_name] = correction
        return(data)

    log = pd.DataFrame(columns = ['var_name', 'extr_inf', 'extr_sup', 'extr_total'])
    data = cv_extrem_main(data, limits, log, correct)
    
    if correct == True:
        log2 = pd.DataFrame(columns = ['var_name', 'extr_inf', 'extr_sup', 'extr_total'])
        data = cv_extrem_main(data, limits, log2, correct = False)
        p_cv_extrem()
        return(data, log)
    else:
        log2 = pd.DataFrame(columns = ['var_name', 'extr_inf', 'extr_sup', 'extr_total'])
        data = cv_extrem_main(data, limits, log2, correct = False)
        p_cv_extrem()
        return(data, log)    
    
    
def cv_pareto(data, variables, th, cat_name, verbose):
    
    from pandas.api.types import CategoricalDtype
    
    assert ((th >= 0) and (th <= 1)) or ((th >= 0) and type(th) == int), f"Function 'cv_pareto': invalid argument 'th' = {th}"
    
    def p_cv_pareto(verbose = False):
        
        if verbose == False:
            return()
    
        print("\n")
        print("*"*30)
        print("'cv_pareto' report")
        print("\nThose are the specific inputs you have passed to this function, check it:")
        print(f"'variables' = {variables}")
        print(f"'th' = {th}")
        print(f"'cat_name' = {cat_name}")
        
        print(f"\n'cv_pareto' modifies the columns 'variables'  in the DataFrame")
        print(f"and returns the transformed DataFrame, the pareto data and the following log:\n")
        print(log)
        print("\n'cv_pareto' end of report")
        print("*"*30) 
       
    pareto_log  = {} # Dictionnary of the type {'var1': pareto distribution for 'var1', etc}
    log = pd.DataFrame(columns = ['var_name', 'categories', 'merged', 'remains',\
                                  'population', 'ratio_not_merged', 'target_is'])

    for i, var in enumerate(variables):
        
        # Create the Pareto distribution which also orders indexes
        pareto = data.groupby(var).size()
        pareto = pareto.sort_values(ascending = False)
        pareto = pareto.cumsum()/pareto.sum()
        
        # th > 1 means that the threashhold is a number of categories and not a percentage
        if th > 1:
            if len(pareto) >= th:
                th_ = pareto.iloc[th-2]
            else:
                th_ = pareto.iloc[len(pareto)-1]
        else:
            th_ = th
                
        # Create indexes referencing variables below and above the given theashold
        index_sup = pareto[(pareto <= th_)].index
        index_inf = pareto[pareto > th_].index

    
        # Create a category named cat_name in index_sup_new if:
        # - the threashold is not 100%,
        # - index_inf is not empty, meaning the threashold has been met
        if (th != 1) and (index_inf.empty == False) and (cat_name not in pareto):
            index_sup_new = index_sup.append(pd.Index([cat_name]))
        else:
            index_sup_new = index_sup
       

        # Calculate some measures
        pop_sup = data[var].isin(index_sup).sum()
        pop_inf = data[var].isin(index_inf).sum()
        pop_tot = pop_sup + pop_inf
        pop_rate = pop_sup / pop_tot
        cat_sup = len(index_sup)
        cat_inf = len(index_inf)
        cat_tot = cat_sup + cat_inf
        
        # Merge categories
        if th !=1:
            data.loc[data[var].isin(index_inf), var] = cat_name 
        
        # Make var a categorical variable
        cat_type = CategoricalDtype(categories = index_sup_new, ordered=True)
        data[var] = data[var].astype(cat_type)
        
        # Log measures
        log.loc[i] = [var, cat_tot, cat_inf, cat_sup, pop_tot, pop_rate, str(th * 100)+" %"]
        
        # Create the new Pareto distribution which also orders indexes
        pareto = data.groupby(var).size()
        pareto = pareto.sort_values(ascending = False)
        pareto = round(pareto/pareto.sum() * 100,2)
        pareto = pareto.reset_index()
        pareto.columns = [var, '%']
        pareto[var] = pareto[var].astype(cat_type)
        pareto = pareto.sort_values(by = var)
        pareto_log[var] = pareto  
        
    p_cv_pareto(verbose)
    
    return(data, log, pareto_log)


def cv_pareto_multiplex(data, variables, th, cat_name, verbose):
    
    from pandas.api.types import CategoricalDtype
    
    assert ((th >= 0) and (th <= 1)) or ((th >= 0) and type(th) == int), f"Function 'cv_pareto_multiplex': invalid argument 'th' = {th}"
    
    # Print function
    def p_cv_pareto_multiplex(verbose = False):
        
        if verbose == False:
            return()
    
        print("\n")
        print("*"*30)
        print("'cv_pareto_multiplex' report")
        print("\nThose are the specific inputs you have passed to this function, check it:")
        print(f"'variables' = {variables}")
        print(f"'th' = {th}")
        print(f"'cat_name' = {cat_name}")
        
        print(f"\n'cv_pareto_multiplex' modifies the columns 'variables'  in the DataFrame")
        print(f"and returns the transformed DataFrame, the pareto data and the following log:\n")
        print(log)
        print("\n'cv_pareto_multiplex' end of report")
        print("*"*30) 
       
    # Log variables
    pareto_log  = {} # Dictionnary of the type {'var1': pareto distribution for 'var1', etc}
    log = pd.DataFrame(columns = ['var_name', 'categories', 'merged', 'remains',\
                                  'population', 'ratio_not_merged', 'target_is'])

    # Loops over the input variables
    for i, var in enumerate(variables):
        
        # Demultiplexing.
        # For each individual, extracts the given 'var' column and demultiplexes its words within 'df_words'. 
        # Individuals with nans values for the column 'var' are filtered out.
        df = data.loc[data[var].notna(), var]
        len_df = len(df)
        df_words = []
        individual_id = []
        for i, words in enumerate(df):
            words_split = words.split(",")
            df_words.extend(words_split)
            individual_id.extend([i] * len(words_split))
        df_words = pd.DataFrame({var: df_words, 'ind': individual_id})
        
        # Removes duplicates.
        # The intent is to count the number of individuals for which a given string pattern occurs.
        # The count shall be the same whether the pattern occurs one or several times for the same individual.
        df_words.drop_duplicates(inplace = True)

        # Computes the Pareto distribution of the occurence of the words within 'df_words'
        # NB: groupby().size() accounts for nans and returns a Series,
        # while count() would not account for nans and return a DataFrame.
        pareto = df_words.groupby(var).size()
        pareto = pareto.sort_values(ascending = False)      
        pareto = pareto.cumsum()/pareto.sum()

        # th > 1 means that the threashhold is a number of categories and not a percentage
        if th > 1:
            if len(pareto) >= th:
                th_ = pareto.iloc[th-2]
            else:
                th_ = pareto.iloc[len(pareto)-1]
        else:
            th_ = th
                
        # Create indexes referencing variables below and above the given theashold      
        index_sup = pareto[pareto <= th_].index
        index_inf = pareto[pareto > th_].index

        if (th != 1) and (index_inf.empty == False):
            index_sup_new = index_sup.append(pd.Index([cat_name]))
        else:
            index_sup_new = index_sup


            
        # Merge categories
        if th !=1:
            df_words.loc[df_words[var].isin(index_inf), var] = cat_name 
        
        # Make var a categorical variable
        cat_type = CategoricalDtype(categories = index_sup_new, ordered=True)
        df_words[var] = df_words[var].astype(cat_type)
        
        # Remove duplicates (several 'cat_name' elements can be present per line, we don't want that)
        df_words.drop_duplicates(inplace = True)
        
        # Calculate some measures
        pop_sup = df_words[var].isin(index_sup).sum()
        pop_inf = df_words[var].isin(index_inf).sum()
        pop_tot = pop_sup + pop_inf
        pop_rate = pop_sup / pop_tot
        cat_sup = len(index_sup)
        cat_inf = len(index_inf)
        cat_tot = cat_sup + cat_inf
        
        # Log measures
        log.loc[i] = [var, cat_tot, cat_inf, cat_sup, pop_tot, pop_rate, str(th * 100)+" %"]
        
        # Create the new Pareto distribution which also orders indexes
        pareto = df_words.groupby(var).size()
        pareto = pareto.sort_values(ascending = False)
        pareto = round(pareto/len_df * 100,2)
        pareto = pareto.reset_index()
        pareto.columns = [var, '%']
        pareto[var] = pareto[var].astype(cat_type)
        pareto = pareto.sort_values(by = var)
        pareto_log[var] = pareto  
        
    p_cv_pareto_multiplex(verbose)
    
    return(data, log, pareto_log)




def cv_missing(data, var_info, variables, correct, correction):
    
    missing = variables

    log = pd.DataFrame(columns = ['var_name', 'NaNs', 'nan_ratio %'])
    
    print(var_info.columns)
    
    print("\n")
    print("*"*30)
    print("'cv_missing' report")
    print("\nThose are the specific inputs you have passed to this function, check it:")
    print(f"'variables' = {variables}")
    print(f"'correct' = {correct}")
    print(f"'correction' = {correction}")

    d = data
    
    for i, c in enumerate(missing):
        nan_before = d[c].isna().sum()
        nan_ratio = float(var_info.loc[var_info['var_name'] == c ,'nan_rate'] * 100)
        log.loc[i] = [c, nan_before, nan_ratio]
        
    print(f"\nThose are the variables treated:")
    print(missing)
    
    print(f"\nFor each variable, this is the number of missing data to correct\n")
    print(log)
             
    if correct == True:
        
        for i, c in enumerate(missing):
            d.loc[d[c].isna(), c] = correction
            
        print(f"\n'cv_missing' has replaced all NaN values by '{correction}'")
        print(f"\n'cv_missing' returns the corrected DataFrame and the above log")
        print(f"\nSanity check: the same log would now look as such:\n")
        
        log2 = pd.DataFrame(columns = ['var_name', 'NaNs', 'nan_ratio %'])
        for i, c in enumerate(missing):
            nan_before = d[c].isna().sum()
            nan_ratio = d[c].isna().sum() / d[c].size * 100
            log2.loc[i] = [c, nan_before, nan_ratio]
        print(log2)
        print("\n'cv_missing' end of report")
        print("*"*30)
        return(d, log)
    
    else:
        print(f"\nAs requested, no variable was corrected")
        print(f"\n'cv_missing' returns the same DataFrame and the above log")
        print(f"\nSanity check: the same log would now look as such:\n")
        log2 = pd.DataFrame(columns = ['var_name', 'NaNs', 'nan_ratio %'])
        for i, c in enumerate(missing):
            nan_before = d[c].isna().sum()
            nan_ratio = d[c].isna().sum() / d[c].size * 100
            log2.loc[i] = [c, nan_before, nan_ratio]
        print(log2)        
        print("\n'cv_missing' end of report")
        print("*"*30)
        return(d, log)

    
def get_outliers_helper(data, data_stat, var_name, cut_off_coef = 1.5, method = 'iqr', verbose = False):
    from numpy import percentile 
    
    #data et data
    
    outliers = []
    no_outliers = []
    
    if method == 'iqr':
        if data.loc[data[var_name].notna(), var_name].size == 0:
            q25, q75, iqr, lower, upper = (np.nan,) * 5
        else:
            q25 = percentile(data.loc[data[var_name].notna(),var_name], 25)
            q75 = percentile(data.loc[data[var_name].notna(),var_name], 75)
            iqr = q75 - q25
            cut_off = cut_off_coef * iqr
            lower, upper = q25 - cut_off, q75 + cut_off
    
    if method == 'std':
        data_mean, data_std = data_stat[var_name].mean(), data_stat[var_name].std()
        cut_off = cut_off_coef * data_std
        lower, upper = data_mean - cut_off, data_mean + cut_off
    
    ind = (data[var_name] < lower) | (data[var_name] > upper)
    ind2 = ~ind
    ind = data[ind].index
    outliers.extend(ind)
    ind2 = data[ind2].index
    no_outliers.extend(ind2)
    

    n_non_outliers_without_nan = data.loc[no_outliers, var_name].count()
    n_outliers = data.loc[outliers, var_name].count()
    n_nan = data[var_name].isna().sum()
    pop_with_nan = len(data)
    pop_without_nan = data[var_name].count()
    ratio = float(n_outliers / pop_without_nan * 100)
    
    if verbose and method == 'iqr':

        print("\n")
        print("*"*30)
        print("'get_outliers_index' report")
        print(f"\n'get_outliers_index' finds outliers of the variable {var_name} with the {method} method")
        print(f"\nThe cutoff coefficient applied is {cut_off_coef}")  
        print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))        
        print('Outliers are outside the range %.3f, %.3f' % (lower, upper))
        print('\nPopulation including NaN: %d' % pop_with_nan)
        print('Population excluding NaN: %d' % pop_without_nan)       
        print('NaN: %d' % n_nan)       
        print('Non-outliers without NaN: %d' % n_non_outliers_without_nan)
        print('Outliers: %d' % n_outliers)
        print(f'% outliers: {ratio}')
        print(f"\n'get_outliers_index' returns the index of the outliers and non-outliers along with the above information")
        print(f"\nSanity check:\n")
        print('Outliers + Non-outliers without NaN + NaN = Population including NaN: %r' \
              % (pop_with_nan == n_outliers + n_non_outliers_without_nan + n_nan))
        print('Outliers + Non-outliers without NaN = Population excluding NaN: %r' \
              % (pop_without_nan == n_outliers + n_non_outliers_without_nan))
        print('Population excluding NaN + NaN = Population including NaN: %r' \
              % (pop_without_nan + n_nan == pop_with_nan))
        print("\n'get_outliers_index' end of report")
        print("*"*30)
        
    if verbose and method == 'std':
        print("\n")
        print("*"*30)
        print("'get_outliers_index' report")
        print(f"\n'get_outliers_index' finds outliers of the variable {var_name} with the {method} method")
        print(f"\nThe cutoff coefficient applied is {cut_off_coef}")  
        print('Mean=%.3f, Std=%.3f' % (data_mean, data_std))
        print('Outliers are outside the range %.3f, %.3f' % (lower, upper))
        print('\nPopulation including NaN: %d' % pop_with_nan)
        print('Population excluding NaN: %d' % pop_without_nan)       
        print('NaN: %d' % n_nan)       
        print('Non-outliers without NaN: %d' % n_non_outliers_without_nan)
        print('Outliers: %d' % n_outliers)
        print(f'% outliers: {ratio}')
        print(f"\n'get_outliers_index' returns the index of the outliers and non-outliers along with the above information")
        print(f"\nSanity check:\n")
        print('Outliers + Non-outliers without NaN + NaN = Population including NaN: %r' \
              % (pop_with_nan == n_outliers + n_non_outliers_without_nan + n_nan))
        print('Outliers + Non-outliers without NaN = Population excluding NaN: %r' \
              % (pop_without_nan == n_outliers + n_non_outliers_without_nan))
        print('Population excluding NaN + NaN = Population including NaN: %r' \
              % (pop_without_nan + n_nan == pop_with_nan))
        print("\n'get_outliers_index' end of report")
        print("*"*30)
        
    return(outliers, no_outliers, [n_outliers, n_non_outliers_without_nan, lower, upper])


def get_ouliers_index(data, data_stat, var_name, by = None, cut_off_coef = 1.5, method = 'iqr', verbose = False):

    outliers_ind = {}
    no_outliers_ind = {}
    
    log = pd.DataFrame(columns = ['id', 'category', 'size', 'nan', 'non_outliers_without_nan', 'outliers', '% outliers', 'lower', 'upper'],\
                       dtype = float)

    if verbose:
        print("\n")
        print("*"*30)
        print("'get_ouliers_index' report")
        print(f"\n'get_ouliers_index' finds outliers of the variable '{var_name}'")
        print(f"in each category '{by}' with the '{method}' method")
        print(f"\nThe cutoff coefficient applied is {cut_off_coef}")  
        
    if by == None:
            by_categories = ['all']
            cat = 'all'
            d_stat = data_stat
            d = data
            out, no_out, log_i = get_outliers_helper(d, d_stat, var_name, cut_off_coef, method, verbose = False)
            outliers_ind[cat] = out
            no_outliers_ind[cat] = no_out
            log.loc[0] = [0, cat, len(d), d[var_name].isna().sum(),\
                          log_i[1], log_i[0], log_i[0]/d[var_name].notna().sum() * 100, log_i[2], log_i[3]]
    else:
        by_categories = data[by].unique()
        
        for i,cat in enumerate(by_categories):

            d_stat = data_stat[data_stat[by].isin([cat])]
            d = data[data[by].isin([cat])]

            out, no_out, log_i = get_outliers_helper(d, d_stat, var_name, cut_off_coef, method, verbose = False)
            outliers_ind[cat] = out
            no_outliers_ind[cat] = no_out
            log.loc[i] = [i, cat, len(d), d[var_name].isna().sum(),\
                          log_i[1], log_i[0], log_i[0]/d[var_name].notna().sum() * 100, log_i[2], log_i[3]]
    
    log = log.sort_values(['size'], ascending = False)

    n_non_outliers_without_nan1 = log['non_outliers_without_nan'].sum()
    n_non_outliers_with_nan2 = sum([len(ind) for ind in no_outliers_ind.values()]) 
    n_outliers1 = log['outliers'].sum()
    n_outliers2 = sum([len(ind) for ind in outliers_ind.values()])    
    n_nan1 = log['nan'].sum()
    n_nan2 = len(data.loc[data[var_name].isna(), var_name])
    pop_with_nan1 = log['size'].sum()
    pop_with_nan2 = len(data[var_name])
    pop_without_nan = data[var_name].count()    
    ratio = float(n_outliers1 / pop_without_nan * 100)
    
    
    if verbose:
        print('\nPopulation including Nan: %d' % pop_with_nan1)
        print('Population including Nan (2): %d' % pop_with_nan2)
        print('Population excluding Nan: %d' % pop_without_nan)
        print('Nan: %d' % n_nan1)
        print('Nan (2): %d' % n_nan2)
        print('Non-outliers with NaN: %d' % n_non_outliers_with_nan2)
        print('Non-outliers without NaN: %d' % n_non_outliers_without_nan1)
        print('Outliers: %d' % n_outliers1)
        print('Outliers (2): %d' % n_outliers2)
        print(f'% outliers: {ratio}')
        print(f"\n'get_ouliers_per_cat_index' returns:")
        print(f"the index of the outliers and non-outliers along with the above information and more for each category")
        print(f"\nSanity check:\n")
        print('Population excluding NaN + NaN = Population including NaN: %r'\
              % (pop_without_nan + n_nan1 == pop_with_nan1))
        print('Outliers + Non-outliers without NaN + NaN = Population including NaN: %r'\
              % (n_outliers1 + n_non_outliers_without_nan1 + n_nan1 == pop_with_nan1))
        print('NaN 1 = NaN 2: %r'\
              % (n_nan1 == n_nan2))  
        print('Non-outliers without NaN 1 + NaN 1 = Non-outliers with NaN: %r'\
              % (n_non_outliers_without_nan1 + n_nan1== n_non_outliers_with_nan2)) 
        print('Outliers 1 = Outliers 2: %r'\
              % (n_outliers1 == n_outliers2)) 
        print("\n'get_ouliers_index' end of report")
        print("*"*30)
    else:
        print(f'% outliers: {ratio}')

    return(outliers_ind, no_outliers_ind, log, by_categories)