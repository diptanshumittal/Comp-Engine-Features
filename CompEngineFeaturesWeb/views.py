from django.shortcuts import render,HttpResponse
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
import io
import urllib,base64
from plotly.offline import plot
import plotly.graph_objs as go
import pandas as pd
import csv
from collections import OrderedDict
from pymongo import MongoClient
from random import randint
import json 
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from operator import itemgetter
import numpy as np
import warnings
import time
import os
import pandas as pd
import warnings
from csv import reader
from tqdm import tqdm
from scipy.stats import zscore
from scipy.stats import spearmanr
from operator import itemgetter
import seaborn as sns
import sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.patches import Rectangle
import numpy as np
from RestrictedPython import safe_builtins, compile_restricted
from RestrictedPython.Eval import default_guarded_getitem
from func_timeout import func_timeout,FunctionTimedOut

warnings.filterwarnings("ignore",category=RuntimeWarning)
warnings.filterwarnings("ignore",category=UserWarning)

myclient = MongoClient(port=27017)
mydb = myclient["CompEngineFeaturesDatabase"]
mycol = mydb["Temp"]
features = []
Alltimeseries = []
alpha=0.05

for x in mycol.find({},{ "_id": 0, "HCTSA_TIMESERIES_VALUE":0, "COEF":0, "PVALUE":0}):
    features.append(x)
with open('hctsa_timeseries-data.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    li = list(csv_reader)
    for i in tqdm(li):
        Alltimeseries.append(list(map(float, i)))


#Add views here

def getfeatures(request):
    return JsonResponse({"data": features})

def exporter(request,number,fname):
    data = pd.read_csv('media/matching data.csv')
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="results.csv"'
    writer = csv.writer(response)
    writer.writerow(data.columns)
    for i in range(len(data)):
        writer.writerow(data.iloc[i])
    return response

def error404page(request,exception):
    return render(request,"error404page.html")

def index(request):
    return render(request,'index.html')

def howitworks(request):
    return render(request,'howitworks.html')

def about(request):
    return render(request,'about.html')

def contact(request):
    return render(request,'contact.html')

def contribute(request):
    return render(request,'contribute.html')

keywords = 0
hctsa = 0
def explore(request):
    alldata=[]
    for i in range(keywords.shape[0]):
        alldata.append(dict(keywords.loc[i]))

    context={'data':alldata}

    return render(request,"explore.html",context)

def apiexploremode(request,number,fname):
    plt.switch_backend('agg')

    x = mycol.find_one({'ID': number},{ "_id": 0, "NAME":0, "KEYWORDS":0, "ID":0, "HCTSA_TIMESERIES_VALUE":0})
    x["COEF"] = np.array(x["COEF"]).astype(np.float64)
    x["PVALUE"] = np.array(x["PVALUE"]).astype(np.float64)
    res = []
    for i in range(len(features)):
        if(x["PVALUE"][i]>alpha):
            continue
        res.append(features[i])
        res[-1]['COEF'] = x["COEF"][i]
        res[-1]['PVALUE'] = x["PVALUE"][i]
        res[-1]['COEF_ABS'] = abs(x["COEF"][i])
    res = pd.DataFrame(res)
    res = res.sort_values(by='COEF_ABS',ascending=False)
    DATAFRAME = res.drop(['COEF_ABS'], axis=1)
    DATAFRAME['Rank']=np.arange(1,len(DATAFRAME)+1)
    totalmatches = len(DATAFRAME)
    matched = list(DATAFRAME.iloc[:13,0])
    myfulljsondata=DATAFRAME[:100]
    res = []
    for index, row in list(myfulljsondata.iterrows()):
        res.append(dict(row))

    PairWise=pd.DataFrame()
    for i in range(13):
        temp = mycol.find_one({'ID': matched[i]},{ "HCTSA_TIMESERIES_VALUE": 1})
        PairWise[DATAFRAME.iloc[i,1]] = temp['HCTSA_TIMESERIES_VALUE']
    pairwise_corr=PairWise.corr(method="spearman").abs()
    g=sns.clustermap(pairwise_corr,method="complete",annot=True,linewidth=0.5,square=True)
    columns = list(PairWise.columns)
    N = len(columns)
    wanted_label = fname
    wanted_row = g.dendrogram_row.reordered_ind.index(columns.index(wanted_label))
    wanted_col = g.dendrogram_col.reordered_ind.index(columns.index(wanted_label))
    xywh_row = (0, wanted_row, N, 1)
    xywh_col = (wanted_col, 0, 1, N)
    for x, y, w, h in (xywh_row, xywh_col):
        g.ax_heatmap.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='Blue', lw=4, clip_on=True))
    g.ax_heatmap.tick_params(length=0)
    myfig=plt.gcf()
    buf=io.BytesIO()
    myfig.savefig(buf,format='png')
    buf.seek(0)
    string=base64.b64encode(buf.read())
    uri=urllib.parse.quote(string)

    yAxis = {}
    yAxis['ydata'] = list(PairWise[fname])
    yAxis['label'] = fname
    xAxes = []
    for i in range(1,13):
        xaxis = {}
        xaxis['label'] = DATAFRAME.iloc[i,1]
        xaxis['title'] = abs(DATAFRAME.iloc[i,3])
        xaxis['xdata'] = list(PairWise[DATAFRAME.iloc[i,1]])
        xAxes.append(xaxis)
    graph = {
        "yaxis": yAxis,
        "xAxes": xAxes
    }
    return JsonResponse({"tabledata":res, "totalmatches":totalmatches, "featurename":fname, "heatmap":uri, "graph":graph})

buffer = OrderedDict()
BUFFER_LIMIT = 5           #Set buffer limit
def exploremode(request,number,fname):
    from tqdm import tqdm
    from scipy.stats import spearmanr
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    from matplotlib.patches import Rectangle
    from operator import itemgetter
    import numpy as np
    import warnings
   

    warnings.filterwarnings("ignore",category=RuntimeWarning)
    warnings.filterwarnings("ignore",category=UserWarning)

    alpha=0.05
    New_Feature_vector_dataframe=hctsa.iloc[:,int(number)-1] 
    #print(New_Feature_vector_dataframe)
    correlatedfeatures=[]
    BestMatches = 0
    if fname in buffer.keys():
        BestMatches = buffer[fname]
        buffer.pop(fname)
        buffer[fname] = BestMatches
    else:
        for i in tqdm(range(hctsa.shape[1])):
            eachfeature=[]
            if (hctsa.iloc[:,i].isna().sum())<50:
                coef, p = spearmanr(hctsa.iloc[:,i],New_Feature_vector_dataframe.values,nan_policy="omit")
                if p < alpha:
                    eachfeature=[hctsa.columns[i],p,format(abs(coef),'.3f'),i,*keywords.iloc[i:i+1,2].values,format(coef,'.3f'),*keywords.iloc[i:i+1,0].values]
                    correlatedfeatures.append(eachfeature)
        BestMatches = sorted(correlatedfeatures, key=itemgetter(2))[::-1]
        if(len(buffer.keys())>BUFFER_LIMIT-1):
            buffer.pop(list(buffer.keys())[0])
        buffer[fname] = BestMatches

    totalmatches=len(BestMatches)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    DATAFRAME = pd.DataFrame(BestMatches)
    DATAFRAME.columns = ['Name', 'pvalue', 'Corr', 'ColumnId', 'Keywords', 'Signedcorrvalue',"ID"]
    DATAFRAME['Rank']=np.arange(1,len(BestMatches)+1)
    myfulljsondata=DATAFRAME[:100]
    jsontable=[]

    for i in tqdm(range(myfulljsondata.shape[0])):
        temp=myfulljsondata.iloc[i]
        jsontable.append(dict(temp))


    DATAFRAME=DATAFRAME[['Rank','Name','Keywords','Corr','pvalue','Signedcorrvalue',"ID"]]
    DATAFRAME.to_csv('media/matching data.csv',index=False)

    PairWise=pd.DataFrame()

    featurename=fname
    PairWise[featurename]=New_Feature_vector_dataframe.values

    for i in range(len(BestMatches[:12])):
        PairWise[BestMatches[i][0]] = hctsa.iloc[:, BestMatches[i][3]]
    
    pairwise_corr=PairWise.corr(method="spearman").abs()
    print(pairwise_corr)

    g=sns.clustermap(pairwise_corr,method="complete",annot=True,linewidth=0.5,square=True)


    columns = list(PairWise.columns)

    N = len(columns)
    wanted_label = featurename
    wanted_row = g.dendrogram_row.reordered_ind.index(columns.index(wanted_label))
    wanted_col = g.dendrogram_col.reordered_ind.index(columns.index(wanted_label))

    xywh_row = (0, wanted_row, N, 1)
    xywh_col = (wanted_col, 0, 1, N)
    for x, y, w, h in (xywh_row, xywh_col):
        print(x,y,w,h)
        g.ax_heatmap.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='Blue', lw=4, clip_on=True))
    g.ax_heatmap.tick_params(length=0)
    myfig=plt.gcf()
    buf=io.BytesIO()
    myfig.savefig(buf,format='png')


    buf.seek(0)

    string=base64.b64encode(buf.read())
    uri=urllib.parse.quote(string)

    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    
    Scatterdataframe=pd.DataFrame()

    for i in range(12):
        Scatterdataframe[i]=hctsa.iloc[:,BestMatches[i][3]]

    Scatterdataframe[13]=New_Feature_vector_dataframe

    trace0 = go.Scatter(x=Scatterdataframe.iloc[:,0].rank(), y=Scatterdataframe[13].rank(),mode="markers")
    trace1 = go.Scatter(x=Scatterdataframe.iloc[:,1].rank(), y=Scatterdataframe[13].rank(),mode="markers")
    trace2 = go.Scatter(x=Scatterdataframe.iloc[:,2].rank(), y=Scatterdataframe[13].rank(),mode="markers")
    trace3 = go.Scatter(x=Scatterdataframe.iloc[:,3].rank(), y=Scatterdataframe[13].rank(),mode="markers")
    trace4 = go.Scatter(x=Scatterdataframe.iloc[:,4].rank(), y=Scatterdataframe[13].rank(),mode="markers")
    trace5 = go.Scatter(x=Scatterdataframe.iloc[:,5].rank(), y=Scatterdataframe[13].rank(),mode="markers")
    trace6 = go.Scatter(x=Scatterdataframe.iloc[:,6].rank(), y=Scatterdataframe[13].rank(),mode="markers")
    trace7 = go.Scatter(x=Scatterdataframe.iloc[:,7].rank(), y=Scatterdataframe[13].rank(),mode="markers")
    trace8 = go.Scatter(x=Scatterdataframe.iloc[:,8].rank(), y=Scatterdataframe[13].rank(),mode="markers")
    trace9 = go.Scatter(x=Scatterdataframe.iloc[:,9].rank(), y=Scatterdataframe[13].rank(),mode="markers")
    trace10 = go.Scatter(x=Scatterdataframe.iloc[:,10].rank(), y=Scatterdataframe[13].rank(),mode="markers")
    trace11 = go.Scatter(x=Scatterdataframe.iloc[:,11].rank(), y=Scatterdataframe[13].rank(),mode="markers")

    fig=go.FigureWidget(make_subplots(rows=3,cols=4,subplot_titles=(f"Correlation = {BestMatches[0][2]}",f"Correlation = {BestMatches[1][2]}",f"Correlation = {BestMatches[2][2]}",f"Correlation = {BestMatches[3][2]}",f"Correlation = {BestMatches[4][2]}",
    f"Correlation = {BestMatches[5][2]}",f"Correlation = {BestMatches[6][2]}",f"Correlation = {BestMatches[7][2]}",f"Correlation = {BestMatches[8][2]}",f"Correlation = {BestMatches[9][2]}",
    f"Correlation = {BestMatches[10][2]}",f"Correlation = {BestMatches[11][2]}")))
    fig.update_layout(template='plotly')


    
    fig.add_trace(trace0,1,1)
    fig.add_trace(trace1,1,2)
    fig.add_trace(trace2,1,3)
    fig.add_trace(trace3,1,4)
    fig.add_trace(trace4,2,1)
    fig.add_trace(trace5,2,2)
    fig.add_trace(trace6,2,3)
    fig.add_trace(trace7,2,4)
    fig.add_trace(trace8,3,1)
    fig.add_trace(trace9,3,2)
    fig.add_trace(trace10,3,3)
    fig.add_trace(trace11,3,4)



    fig.update_xaxes(title_text=BestMatches[0][0], row=1, col=1)
    fig.update_xaxes(title_text=BestMatches[1][0], row=1, col=2)
    fig.update_xaxes(title_text=BestMatches[2][0], row=1, col=3)
    fig.update_xaxes(title_text=BestMatches[3][0], row=1, col=4)
    fig.update_xaxes(title_text=BestMatches[4][0], row=2, col=1)
    fig.update_xaxes(title_text=BestMatches[5][0], row=2, col=2)
    fig.update_xaxes(title_text=BestMatches[6][0], row=2, col=3)
    fig.update_xaxes(title_text=BestMatches[7][0], row=2, col=4)
    fig.update_xaxes(title_text=BestMatches[8][0], row=3, col=1)
    fig.update_xaxes(title_text=BestMatches[9][0], row=3, col=2)
    fig.update_xaxes(title_text=BestMatches[10][0], row=3, col=3)
    fig.update_xaxes(title_text=BestMatches[11][0], row=3, col=4)


    fig.update_yaxes(title_text=featurename,row=1,col=1)
    fig.update_yaxes(title_text=featurename,row=2,col=1)
    fig.update_yaxes(title_text=featurename,row=3,col=1)

    fig.update_layout(showlegend=False,template='ggplot2',paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(
    r=130
    ))


    graph = fig.to_html(full_html=False, default_height=1200, default_width=1200)

    context = {
        "clusterdata":uri,
        "data":jsontable,
        'graph': graph,
        "totalmatches":totalmatches,
        "featurename":featurename
            }


    return render(request,"result.html",context)

def result(request):

    class MaxCountIter:
        def __init__(self, dataset, max_count):
            self.i = iter(dataset)
            self.left = max_count

        def __iter__(self):
            return self

        def __next__(self):
            if self.left > 0:
                self.left -= 1
                return next(self.i)
            else:
                raise StopIteration()

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
            safe_modules = ["math","statistics","numpy","scipy","pandas","statsmodels","sklearn"]
            if name in safe_modules:
                globals[name] = __import__(name, globals, locals, fromlist, level)
            else:
                raise

    def _getiter(ob):
        return MaxCountIter(ob, MAX_ITER_LEN)

    def execute_user_code(byte_code, *args, **kwargs):
        def _apply(f, *a, **kw):
            return f(*a, **kw)        
        safe_builtins['__import__'] = _import
        try:
            # This is the variables we allow user code to see. @result will contain return value.
            restricted_locals = {
                "result": None,
                "args": args,
                "kwargs": kwargs,
            }
            restricted_globals = {
                "_getiter_": _getiter,
                "__builtins__": safe_builtins,
                "_getitem_": default_guarded_getitem,
                "_apply_": _apply,
                "sum":sum,
                "min":min,
                "max":max,
                "abs":abs,
                "sorted":sorted,
                "round":round,
                "type":type,
                "complex":complex
            }
            exec(byte_code, restricted_globals, restricted_locals)
            return restricted_locals["result"]
        except SyntaxError as e:
            raise
        except Exception as e:
            raise
    
    def Execute_User_Code():
        user_code += "\nresult = {0}(*args, **kwargs)".format(featurename)
        byte_code = compile_restricted(user_code, filename="<user_code>", mode="exec")
        for i in tqdm(range(len(Alltimeseries))):
            featurevalue = execute_user_code(byte_code, Alltimeseries[i])
            New_feature_vector.append(featurevalue)

    New_feature_vector=[]
    if request.method != 'POST':
        return JsonResponse({"stat":5})
    featurename=request.POST['featurename']
    featurecode=request.FILES['featurecode']
    fs= FileSystemStorage()
    modulename=fs.save(featurecode.name,featurecode).replace(".py","")
    MAX_ITER_LEN = 1000000000000000
    with open(f"media\{modulename}.py") as f:
        user_code=f.read()
        print(user_code)

    try:
        mytemplate="result.html"
        try:
            func_timeout(300,Execute_User_Code)
        except FunctionTimedOut:
            raise
        nan_fvector=int(pd.DataFrame(New_feature_vector).isna().sum())
        if int(nan_fvector)>50:
            raise SyntaxError
        allData = mycol.find({},{"NAME":1, "KEYWORDS":1, "ID":1, "HCTSA_TIMESERIES_VALUE":1})
        
        correlatedfeatures=[]
        for i in tqdm(range(len(allData))):
            eachfeature=[]
            if (pd.DataFrame(allData[i]['HCTSA_TIMESERIES_VALUE']).isna().sum()+nan_fvector)<50:
                coef, p = spearmanr( allData[i]['HCTSA_TIMESERIES_VALUE'], New_feature_vector, nan_policy="omit")
                if p < alpha:
                    eachfeature=[allData[i]['NAME'], p, format(abs(coef),'.3f'), allData[i]['ID'], allData[i]['KEYWORDS'], format(coef,'.3f')]
                    correlatedfeatures.append(eachfeature)
        BestMatches = sorted(correlatedfeatures, key=itemgetter(2))[::-1]
        totalmatches=len(BestMatches)
        DATAFRAME = pd.DataFrame(BestMatches)
        DATAFRAME.columns = ['Name', 'pvalue', 'Corr', 'ID', 'Keywords', 'Signedcorrvalue']
        DATAFRAME['Rank']=np.arange(1,len(BestMatches)+1)
        myfulljsondata=DATAFRAME[:100]
        res = []
        for index, row in list(myfulljsondata.iterrows()):
            res.append(dict(row))

        PairWise=pd.DataFrame()    
        PairWise[featurename]=New_feature_vector
        for i in range(len(BestMatches[:12])):
            PairWise[BestMatches[i][0]] =  allData[BestMatches[i][3]-1]["HCTSA_TIMESERIES_VALUE"]
        pairwise_corr=PairWise.corr(method="spearman").abs()
        g=sns.clustermap(pairwise_corr,method="complete",annot=True,linewidth=0.5,square=True)
        columns = list(PairWise.columns)
        N = len(columns)
        wanted_label = featurename
        wanted_row = g.dendrogram_row.reordered_ind.index(columns.index(wanted_label))
        wanted_col = g.dendrogram_col.reordered_ind.index(columns.index(wanted_label))
        xywh_row = (0, wanted_row, N, 1)
        xywh_col = (wanted_col, 0, 1, N)
        for x, y, w, h in (xywh_row, xywh_col):
            g.ax_heatmap.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='Blue', lw=4, clip_on=True))
        g.ax_heatmap.tick_params(length=0)
        myfig=plt.gcf()
        buf=io.BytesIO()
        myfig.savefig(buf,format='png')
        buf.seek(0)
        string=base64.b64encode(buf.read())
        uri=urllib.parse.quote(string)
        
        yAxis = {}
        yAxis['ydata'] = list(PairWise[featurename])
        yAxis['label'] = featurename
        xAxes = []
        for i in range(1,13):
            xaxis = {}
            xaxis['label'] = DATAFRAME.iloc[i,0]
            xaxis['title'] = abs(DATAFRAME.iloc[i,2])
            xaxis['xdata'] = list(PairWise[DATAFRAME.iloc[i,0]])
            xAxes.append(xaxis)
        graph = {
            "yaxis": yAxis,
            "xAxes": xAxes
        }

        return JsonResponse({"stat":1, "tabledata":res, "totalmatches":totalmatches, "featurename":featurename, "heatmap":uri, "graph":graph})
    except SyntaxError as e:
        return JsonResponse({"stat":2})
    except Exception as e:
        return JsonResponse({"stat":3})
    except FunctionTimedOut:
        return JsonResponse({"stat":4})

    '''
        # Using plotly for creating interactive graphs
        import plotly.graph_objs as go
        
        from plotly.subplots import make_subplots

        
        Scatterdataframe=pd.DataFrame()
        
        for i in range(12):
            Scatterdataframe[i]=hctsa.iloc[:,BestMatches[i][3]]
        
        Scatterdataframe[13]=New_Feature_vector_dataframe.iloc[:,0]


        # For generating  x and y axis scatter plots suing plotly

        trace0 = go.Scatter(x=Scatterdataframe.iloc[:,0].rank(), y=Scatterdataframe[13].rank(),mode="markers")
        trace1 = go.Scatter(x=Scatterdataframe.iloc[:,1].rank(), y=Scatterdataframe[13].rank(),mode="markers")
        trace2 = go.Scatter(x=Scatterdataframe.iloc[:,2].rank(), y=Scatterdataframe[13].rank(),mode="markers")
        trace3 = go.Scatter(x=Scatterdataframe.iloc[:,3].rank(), y=Scatterdataframe[13].rank(),mode="markers")
        trace4 = go.Scatter(x=Scatterdataframe.iloc[:,4].rank(), y=Scatterdataframe[13].rank(),mode="markers")
        trace5 = go.Scatter(x=Scatterdataframe.iloc[:,5].rank(), y=Scatterdataframe[13].rank(),mode="markers")
        trace6 = go.Scatter(x=Scatterdataframe.iloc[:,6].rank(), y=Scatterdataframe[13].rank(),mode="markers")
        trace7 = go.Scatter(x=Scatterdataframe.iloc[:,7].rank(), y=Scatterdataframe[13].rank(),mode="markers")
        trace8 = go.Scatter(x=Scatterdataframe.iloc[:,8].rank(), y=Scatterdataframe[13].rank(),mode="markers")
        trace9 = go.Scatter(x=Scatterdataframe.iloc[:,9].rank(), y=Scatterdataframe[13].rank(),mode="markers")
        trace10 = go.Scatter(x=Scatterdataframe.iloc[:,10].rank(), y=Scatterdataframe[13].rank(),mode="markers")
        trace11 = go.Scatter(x=Scatterdataframe.iloc[:,11].rank(), y=Scatterdataframe[13].rank(),mode="markers")
        

        # adding correlation value as title for all 12 plots

        fig=go.FigureWidget(make_subplots(rows=3,cols=4,subplot_titles=(f"Correlation = {BestMatches[0][2]}",f"Correlation = {BestMatches[1][2]}",f"Correlation = {BestMatches[2][2]}",f"Correlation = {BestMatches[3][2]}",f"Correlation = {BestMatches[4][2]}",
        f"Correlation = {BestMatches[5][2]}",f"Correlation = {BestMatches[6][2]}",f"Correlation = {BestMatches[7][2]}",f"Correlation = {BestMatches[8][2]}",f"Correlation = {BestMatches[9][2]}",
        f"Correlation = {BestMatches[10][2]}",f"Correlation = {BestMatches[11][2]}")))
        fig.update_layout(template='plotly')


        # Generating 12 scatter plots on single page


        fig.add_trace(trace0,1,1)
        fig.add_trace(trace1,1,2)
        fig.add_trace(trace2,1,3)
        fig.add_trace(trace3,1,4)
        fig.add_trace(trace4,2,1)
        fig.add_trace(trace5,2,2)
        fig.add_trace(trace6,2,3)
        fig.add_trace(trace7,2,4)
        fig.add_trace(trace8,3,1)
        fig.add_trace(trace9,3,2)
        fig.add_trace(trace10,3,3)
        fig.add_trace(trace11,3,4)


        # Updating x-axis labels for scatter plots


        fig.update_xaxes(title_text=BestMatches[0][0], row=1, col=1)
        fig.update_xaxes(title_text=BestMatches[1][0], row=1, col=2)
        fig.update_xaxes(title_text=BestMatches[2][0], row=1, col=3)
        fig.update_xaxes(title_text=BestMatches[3][0], row=1, col=4)
        fig.update_xaxes(title_text=BestMatches[4][0], row=2, col=1)
        fig.update_xaxes(title_text=BestMatches[5][0], row=2, col=2)
        fig.update_xaxes(title_text=BestMatches[6][0], row=2, col=3)
        fig.update_xaxes(title_text=BestMatches[7][0], row=2, col=4)
        fig.update_xaxes(title_text=BestMatches[8][0], row=3, col=1)
        fig.update_xaxes(title_text=BestMatches[9][0], row=3, col=2)
        fig.update_xaxes(title_text=BestMatches[10][0], row=3, col=3)
        fig.update_xaxes(title_text=BestMatches[11][0], row=3, col=4)


        fig.update_yaxes(title_text=featurename,row=1,col=1)
        fig.update_yaxes(title_text=featurename,row=2,col=1)
        fig.update_yaxes(title_text=featurename,row=3,col=1)



        # plots configuration (papercolor,margin,show legends)


        fig.update_layout(showlegend=False,template='ggplot2',paper_bgcolor='rgba(0,0,0,0)', margin=dict(r=130))
        
        # configuring graph into html for rendering.

        graph = fig.to_html(full_html=False, default_height=1200, default_width=1200)
        

        # Dictionary for passsing the variables to be rendered on result page

        context = {
            "clusterdata":uri,
            "data":jsontable,
            'graph': graph,
            "totalmatches":totalmatches,
            "featurename":featurename
            }
        '''
