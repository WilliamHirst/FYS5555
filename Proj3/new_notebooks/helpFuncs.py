def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]

def checkFeatures(df_reference,df_subject):
    col_ref = df_reference.columns
    col_sub = df_subject.columns
    drop_from_ref = []
    drop_from_sub = []
    if len(col_sub) > len(col_ref):
        for cs in col_sub:
            if not cs in col_ref:
                print("INFO \t Dropping %s from subject" %cs)
                drop_from_sub.append(cs)
    if len(col_ref) > len(col_sub):
        for cs in col_ref:
            if not cs in col_sub:
                print("INFO \t Dropping %s from reference" %cs)
                drop_from_ref.append(cs)

    df_subject.drop(drop_from_sub,axis='columns',inplace=True)
    df_reference.drop(drop_from_ref,axis='columns',inplace=True)
    return df_reference,df_subject

def plotScaleFeatures(X,df,var,nbins,isRef=1,setlog=True):

    nf = -1
    nf = np.where(dataframe_subject.columns.values == var)[0][0]

    if nf < 0:
        print("ERROR \t Could not find feature %s in dataframe"%(var))
        return -1
    
    a = np.array([X.item((i,nf)) for i in range(X.shape[0])])
    figAE, axsAE = plt.subplots(2, 1)
    axAE1, axAE2 = axsAE.ravel()

    ns1, bins1, patches1 = axAE1.hist(df[var], bins=nbins, facecolor='blue' if isRef else 'green', histtype='stepfilled',label='ref' if isRef else 'subj')
    ns2, bins2, patches2 = axAE2.hist(a,       bins=nbins, facecolor='blue' if isRef else 'green', histtype='stepfilled',label='ref (scaled)' if isRef else 'subj (scaled)')

    axAE1.legend()
    axAE2.legend()

    if setlog:
        try:
            axAE1.set_yscale('log')
            axAE2.set_yscale('log')
        except:
            print("WARNING \t Can not use log scale!")

    axAE2.set_xlabel(var)

    if isRef:
        figAE.savefig("fig/plot_ref_feat_and_scaled_%s.png"%var)
    else:
        figAE.savefig("fig/plot_sub_feat_and_scaled_%s.png"%var)

def makeHist(df1, var, nbins, df2=0, setlog = True):
    figAE, axsAE = plt.subplots(2, 1,gridspec_kw={'height_ratios': [3, 1]})
    axAE1, axAE2 = axsAE.ravel()

    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)

    max_sub = -1e5
    min_sub = 1e5
    if not type(df2) == int:
        max_sub = max(df2[var])
        min_sub = min(df2[var])
    max_ref = max(df1[var])
    min_ref = min(df1[var])

    min_bin = min(min_sub,min_ref)
    max_bin = max(max_sub,max_ref)

    #print(min_bin,max_bin)
    
    if min_bin != max_bin:
        bins = np.linspace(min_bin, max_bin, nbins)
    else:
        if min_bin > 0 and max_bin > 0:
            bins = np.linspace(min_bin-(min_bin*5), max_bin+(max_bin*5), 10)
        else:
            bins = np.linspace(min_bin+(min_bin*5), max_bin-(max_bin*5), 10)

    #print(bins)
    plt.figure(0)
    abs_ns1, abs_bins1, abs_patches1 = plt.hist(df1[var], bins, facecolor='blue', alpha=0.4, histtype='stepfilled',label='ref.',density = False)
    abs_ns2, abs_bins2, abs_patches2 = plt.hist(df2[var], bins, facecolor='green', alpha=0.4, histtype='stepfilled',label='subj.',density = False)
    
    
    ns1, bins1, patches1 = axAE1.hist(df1[var], bins, facecolor='blue', alpha=0.4, histtype='stepfilled',label='ref.',density = True)
    #print(type(bins1))
    if not type(df2) == int:
        ns2, bins2, patches2 = axAE1.hist(df2[var], bins, facecolor='green', alpha=0.4, histtype='stepfilled',label='subj.',density = True)
    axAE1.legend()

    if setlog:
        try:
            axAE1.set_yscale('log')
        except:
            print("WARNING \t Can not use log scale!")
    
    axAE1.set_ylabel("Events")
    #axAE1.set_xlabel(var)

    #print(bins1.size)
    #print(ns1.size)
    #print(ns2.size)

    np.seterr(all='ignore')
    ratio = np.divide(ns1, ns2)#, out=np.zeros_like(ns1), where=ns2!=0)
    ratio[ratio >= 1E308] = 0
    ratio[ratio <= -1E308] = 0

    #print(ratio)
    
    rat_max = max(ratio)
    rat_min = min(ratio)

    ns1divide = np.divide(np.sqrt(abs_ns1),abs_ns1,where=(ns1 > 0))
    ns2divide = np.divide(np.sqrt(abs_ns2),abs_ns2,where=(ns2 > 0))

    error = np.multiply(np.sqrt(ns1divide+ns2divide),ratio)
    
    ratio = zero_to_nan(ratio)

    if (math.isnan(rat_max) or math.isnan(rat_min)) or (math.isinf(rat_max) or math.isinf(rat_min)):
        #print(min_bin,max_bin)
        #print(bins)
        #print(ratio)
        #print(ns1)
        #print(ns2)
        #print("WARNING \t Problems with axis limits for %s: ratio max = %.2f, min = %.2f"%(var,rat_max,rat_min))
        rat_min = 0.5
        rat_max = 2.0
        #return -1,-1,-1,-1,-1,-1
    if rat_max > 5: rat_max = 5
    axAE2.set_ylim([(rat_min - (rat_min*0.1)) if rat_min != 0 else 0.5, (rat_max+(rat_max*0.1)) if rat_max < 5 else 5.0])


    
    #axAE2 = axAE1.twinx()

    bincenter = 0.5 * (bins1[1:] + bins1[:-1])

    #print(bincenter)

    axAE2.errorbar(bincenter, ratio, yerr=error, fmt='.', color='r')
    #axAE2.hist(error, nbins)
    
    #axAE2.plot(bins1[:-1], ratio, 'bo')
    x_min, x_max = axAE1.get_xlim()

    axAE2.set_xlim(x_min, x_max)
    axAE2.set_ylabel("Ref./Subj.")
    axAE2.set_xlabel(var)
    axAE2.axhline(y=1.0, color='black', linestyle='-')
    axAE2.grid(visible=True,axis='y')

    tickint = (rat_max+0.1+rat_min)/8.
    plt.yticks(np.arange(rat_min, rat_max+0.1, tickint))
    
    figAE.savefig("fig/plot_%s.png"%var)

    return ns1, bins1, patches1, ns2, bins2, patches2
# Possibility of doing simple cuts e.g. removing all objects not passing some True/False
# requirement,
def doCut(df, ident = 'muon_', cutvar = 'muon_isSignal', cutval = 0, vb = 0):

    print("INFO \t Cutting on all %s* variables requiring %s != %i"%(ident,cutvar,cutval))

    leaf_dic = {}
    
    for c in df.columns:
        if c in feature_drop_bad: continue
        if ident in c:
            if not df.dtypes[c] == 'object':
                print("Skipping %s as type is %s"%(c,df.dtypes[c]))
                continue
            leaf_dic[c] = df[c]

    i = 0
    df[ident+"n"] = np.zeros(df.shape[0],dtype='int32')
    for j in df[cutvar]:
        if i%1000 == 0:
            print("INFO \t Done %i of %i rows" %(i,df.shape[0]))
        cutidx = np.where(j == cutval)[0].tolist()
        for key in leaf_dic.keys():
            if vb:
                print("Key = %s"%key)
                print("cutidx = ",cutidx)
                print("value  = ",np.array(leaf_dic[key].iloc[i]))
                print("Before",df[key].iloc[i])
            try:
                cutted_array = np.delete(np.array(leaf_dic[key].iloc[i]),cutidx)
            except:
                if vb or not key in feature_drop_bad:
                    print("Key = %s shows bad behavior. Adding to drop vector"%key)
                feature_drop_bad.append(key)
                continue
            df[key].iloc[i] = cutted_array
            df[ident+"n"].iloc[i] = cutted_array.size
            if vb: print("After",df[key].iloc[i])
        #if i > 5: break
        i += 1
    #df.astype({ident+"n": 'int32'})
    return df



def removeBool(df):
    for c in df.columns:
        if df.dtypes[c] == 'bool':
            df[c] = df[c].astype(int)
    return df

def removeNaN(df):
    col_before_drop_na = df.columns
    df = df.dropna(axis = 1)
    col_after_drop_na = df.columns
    for b in col_before_drop_na:
        if not b in col_after_drop_na:
            print("WARNING \t Removed column %s because contains NaN"%b)
    return df

def dvaugmentation(df,ndv,feature_drop_bad,ident = "dvtrack_", sort_after = 'pt'):

    print("INFO \t Flattening branches %s*, keeping %i object(s) sorted after %s"%(ident,ndv,sort_after))
    
    leaf_dic = {}
    for c in df.columns:
        if c in feature_drop_bad: continue
        if ident in c:
            #print(c)
            #print(df.dtypes[c])
            # If not a vector we just save it as it is, so skipping
            if not df.dtypes[c] == 'object':
                print("INFO \t Skipping %s as type is %s"%(c,df.dtypes[c]))
                continue
            leaf_dic[c] = awkward.from_iter(df[c])

    sortparam = ident+sort_after
    if not sortparam in leaf_dic.keys():
        print("ERROR \t Could not find %s to sort after, returning..."%(sortparam))
        return -1
    

    size = -1
    leaf_dic[sortparam] = awkward.fill_none(awkward.pad_none(leaf_dic[sortparam], ndv,axis=1),1e10)
    mask = np.logical_and(awkward.sum(leaf_dic[sortparam],axis=1) < 1e10, True)
    for key in leaf_dic.keys():
        #print("Key is ",key)
        leaf_dic[key] = awkward.fill_none(awkward.pad_none(leaf_dic[key], ndv,axis=1),1e10)#.fillna(-999)pt.pad(nlep).fillna(-999)
        #print(awkward.sum(leaf_dic[key],axis=1))
        #print(leaf_dic[key])
        

        leaf_dic[key]  = leaf_dic[key][mask]

        mask_size = awkward.size(leaf_dic[key],axis=0)

        if size < 0:
            size = mask_size
        elif mask_size != size:
            print("WARNING \t Mask size %i is different from %i for %s"%(size,mask_size,key))
        

    df['to_keep'] = mask

    
    df.drop( df[df['to_keep'] == False].index , inplace=True)

    # Need to update leaf_dics after having removed some of the rows above
    for c in leaf_dic.keys():
        leaf_dic[c] = awkward.from_iter(df[c])

    for i in range(1,ndv+1):
        mask = np.logical_and(leaf_dic[sortparam] == awkward.max(leaf_dic[sortparam],axis=1), awkward.max(leaf_dic[sortparam],axis=1) != None)
        for key in leaf_dic.keys():
            try:
                df['obj%i_%s'%(i,key)]  = awkward.to_numpy(leaf_dic[key][mask])
            except:
                print("WARNING \t Key = %s"%key)
                print("WARNING \t Mask = ", mask)
                print("WARNING \t Problems augumenting ", leaf_dic[key])
                continue
            
            leaf_dic[key]  = leaf_dic[key][~mask]
            
    df.drop(leaf_dic.keys(),axis='columns',inplace=True)
    
            
    return df
