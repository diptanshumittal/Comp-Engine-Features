import base64
import io
import urllib
import warnings
from collections import OrderedDict
from operator import itemgetter
from copy import deepcopy as dp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from RestrictedPython import safe_builtins, compile_restricted
from RestrictedPython.Eval import default_guarded_getitem
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.shortcuts import render, HttpResponse
from func_timeout import func_timeout, FunctionTimedOut
from matplotlib.patches import Rectangle
from pymongo import MongoClient
from scipy.stats import spearmanr
from tqdm import tqdm

plt.switch_backend('agg')
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

myclient = MongoClient(port=27017)
mydb = myclient["CompEngineFeaturesDatabase"]
mycol = mydb["FeaturesCollection"]
AllFeatures = []
Alltimeseries = []
AlltimeSeriesNames = []
AlltimeseriesCategory = []
alpha = 0.05


def getAllTimeSeries():
    col = mydb["TimeSeries"]
    for x in col.find({}, {"_id": 0}):
        Alltimeseries.append(x["TIMESERIES"])
        AlltimeSeriesNames.append(x["NAME"])
        AlltimeseriesCategory.append(x["CATEGORY"])


getAllTimeSeries()
TimeSeriesCategory = list(set(AlltimeseriesCategory))
for x in mycol.find({}, {"_id": 0, "HCTSA_TIMESERIES_VALUE": 0, "COEF": 0, "PVALUE": 0}):
    AllFeatures.append(x)

# Add views here
featureDic = {}


def getfeatures(request):
    try:
        if len(featureDic) == 0:
            for i in AllFeatures:
                tdic = {
                    "id": i["ID"],
                    "NAME": i["NAME"],
                    "KEYWORDS": i["KEYWORDS"]
                }
                featureDic[i["ID"]] = tdic
        return JsonResponse(featureDic)
    except Exception:
        return JsonResponse({"error": "Recheck the API request made"})


def gettimeseries(request, timeseriesname):
    try:
        number = AlltimeSeriesNames.index(timeseriesname)
        dic = {
            "name": AlltimeSeriesNames[number],
            "ydata": Alltimeseries[number],
            "xdata": np.linspace(1, len(Alltimeseries[number]), len(Alltimeseries[number])).tolist()
        }
        return JsonResponse(dic)
    except Exception:
        return JsonResponse({"error": "Recheck the API request made"})


def error404page(request, exception):
    return render(request, "error404page.html")


def index(request):
    return render(request, 'index.html')


def howitworks(request):
    return render(request, 'howitworks.html')


def about(request):
    return render(request, 'about.html')


def contact(request):
    return render(request, 'contact.html')


def contribute(request):
    return render(request, 'contribute.html')


keywords = pd.read_csv('hctsa_features.csv')
hctsa = pd.read_csv('hctsa_datamatrix.csv')


def explore(request):
    alldata = []
    for i in range(keywords.shape[0]):
        alldata.append(dict(keywords.loc[i]))

    context = {'data': alldata}

    return render(request, "explore.html", context)


def splittimeseries(arr):
    dic = {}
    for i in range(len(AlltimeseriesCategory)):
        if AlltimeseriesCategory[i] in dic:
            dic[AlltimeseriesCategory[i]].append(arr[i])
        else:
            dic[AlltimeseriesCategory[i]] = []
            dic[AlltimeseriesCategory[i]].append(arr[i])
    res = []
    for i in TimeSeriesCategory:
        res.append(dic[i])
    return res


def apiNetwork(request, fid, nodes):
    try:
        nodes = min(20, int(nodes))
        x = mycol.find_one({'ID': int(fid)}, {"_id": 0, "NAME": 0, "KEYWORDS": 0, "ID": 0, "HCTSA_TIMESERIES_VALUE": 0})
        x["COEF"] = np.array(x["COEF"]).astype(np.float64)
        x["PVALUE"] = np.array(x["PVALUE"]).astype(np.float64)
        res = []
        for i in range(len(AllFeatures)):
            if (x["PVALUE"][i] > alpha):
                continue
            res.append(dp(AllFeatures[i]))
            res[-1]['COEF'] = x["COEF"][i]
            res[-1]['PVALUE'] = x["PVALUE"][i]
            res[-1]['COEF_ABS'] = abs(x["COEF"][i])
        DATAFRAME = pd.DataFrame(res)
        DATAFRAME = DATAFRAME.sort_values(by='COEF_ABS', ascending=False)
        PairWise = pd.DataFrame()
        for i in range(min(nodes, len(DATAFRAME))):
            temp = mycol.find_one({'ID': int(DATAFRAME.iloc[i, 0])}, {"HCTSA_TIMESERIES_VALUE": 1})
            PairWise[DATAFRAME.iloc[i, 1]] = temp['HCTSA_TIMESERIES_VALUE']
        PairWise = PairWise.fillna(0)
        pairwise_corr = PairWise.corr(method="spearman")
        names = list(pairwise_corr.columns)
        networkGraph = {
            'nodes': [],
            'edges': []
        }
        for i in range(len(names)):
            networkGraph['nodes'].append({
                'id': i,
                'title': int(DATAFRAME['ID'].iloc[i])
            })
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                networkGraph['edges'].append({
                    'to': j,
                    'from': i,
                    'label': format(pairwise_corr[names[i]][names[j]], '.3f')
                })
        return JsonResponse(
            {"networkGraph": networkGraph})
    except Exception:
        return JsonResponse({"error": "Recheck the API request made"})


def apiexploremode(request, number, fname):
    try:
        plt.switch_backend('agg')

        x = mycol.find_one({'ID': int(number)},
                           {"_id": 0, "NAME": 0, "KEYWORDS": 0, "ID": 0, "HCTSA_TIMESERIES_VALUE": 0})
        x["COEF"] = np.array(x["COEF"]).astype(np.float64)
        x["PVALUE"] = np.array(x["PVALUE"]).astype(np.float64)
        res = []
        for i in range(len(AllFeatures)):
            if (x["PVALUE"][i] > alpha):
                continue
            res.append(dp(AllFeatures[i]))
            res[-1]['COEF'] = x["COEF"][i]
            res[-1]['PVALUE'] = x["PVALUE"][i]
            res[-1]['COEF_ABS'] = abs(x["COEF"][i])
        res = pd.DataFrame(res)
        res = res.sort_values(by='COEF_ABS', ascending=False)
        DATAFRAME = res.drop(['COEF_ABS'], axis=1)
        DATAFRAME = DATAFRAME.fillna(0)
        DATAFRAME['Rank'] = np.arange(1, len(DATAFRAME) + 1)
        PairWise = pd.DataFrame()
        for i in range(13):
            temp = mycol.find_one({'ID': int(DATAFRAME.iloc[i, 0])}, {"HCTSA_TIMESERIES_VALUE": 1})
            PairWise[DATAFRAME.iloc[i, 1]] = temp['HCTSA_TIMESERIES_VALUE']
        PairWise = PairWise.fillna(0)
        pairwise_corr = PairWise.corr(method="spearman").abs()
        g = sns.clustermap(pairwise_corr, method="complete", annot=True, linewidth=0.5, square=True)
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
        myfig = plt.gcf()
        buf = io.BytesIO()
        myfig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        scatterPlotsData = {
            'xaxis': {
                'xdata': splittimeseries(PairWise[fname].rank()),
                'xtit': fname
            },
            'yaxes': []
        }
        TimeseriesNamesDivided = splittimeseries(AlltimeSeriesNames)
        for i in range(1, 13):
            gr = {
                'title': "Correlation = " + str(DATAFRAME.iloc[i, 3]),
                'ytit': DATAFRAME.iloc[i, 1],
                'ydata': splittimeseries(PairWise[DATAFRAME.iloc[i, 1]].rank())
            }
            scatterPlotsData['yaxes'].append(gr)
        pairwise_corr = PairWise.corr(method="spearman")
        names = list(pairwise_corr.columns)
        networkGraph = {
            'nodes': [],
            'edges': []
        }
        for i in range(len(names)):
            networkGraph['nodes'].append({
                'id': i,
                'title': int(DATAFRAME['ID'].iloc[i])
            })
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                networkGraph['edges'].append({
                    'to': j,
                    'from': i,
                    'label': format(pairwise_corr[names[i]][names[j]], '.3f')
                })

        totalmatches = len(DATAFRAME)
        DATAFRAME["id"] = DATAFRAME["ID"].astype("int64")
        DATAFRAME = DATAFRAME.drop(['NAME'], axis=1)
        DATAFRAME = DATAFRAME.drop(['KEYWORDS'], axis=1)
        DATAFRAME = DATAFRAME.drop(['ID'], axis=1)
        res = []
        for index, row in list(DATAFRAME.iterrows()):
            res.append(dict(row))

        return JsonResponse(
            {"tabledata": res, "totalmatches": totalmatches, "featurename": fname, "heatmap": uri,
             "scatterplotgraphs": scatterPlotsData, "timeseriesnames": TimeseriesNamesDivided,
             "timeseriescategory": TimeSeriesCategory, "networkGraph": networkGraph})
    except Exception:
        return JsonResponse({"error": "Recheck the API request made"})


buffer = OrderedDict()
BUFFER_LIMIT = 5  # Set buffer limit


def exploremode(request, number, fname):
    plt.switch_backend('agg')

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    alpha = 0.05
    New_Feature_vector_dataframe = hctsa.iloc[:, int(number) - 1]
    # print(New_Feature_vector_dataframe)
    correlatedfeatures = []
    BestMatches = 0
    if fname in buffer.keys():
        BestMatches = buffer[fname]
        buffer.pop(fname)
        buffer[fname] = BestMatches
    else:
        for i in tqdm(range(hctsa.shape[1])):
            eachfeature = []
            if (hctsa.iloc[:, i].isna().sum()) < 50:
                coef, p = spearmanr(hctsa.iloc[:, i], New_Feature_vector_dataframe.values, nan_policy="omit")
                if p < alpha:
                    eachfeature = [hctsa.columns[i], p, format(abs(coef), '.3f'), i, *keywords.iloc[i:i + 1, 2].values,
                                   format(coef, '.3f'), *keywords.iloc[i:i + 1, 0].values]
                    correlatedfeatures.append(eachfeature)
        BestMatches = sorted(correlatedfeatures, key=itemgetter(2))[::-1]
        if (len(buffer.keys()) > BUFFER_LIMIT - 1):
            buffer.pop(list(buffer.keys())[0])
        buffer[fname] = BestMatches

    totalmatches = len(BestMatches)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    DATAFRAME = pd.DataFrame(BestMatches)
    DATAFRAME.columns = ['Name', 'pvalue', 'Corr', 'ColumnId', 'Keywords', 'Signedcorrvalue', "ID"]
    DATAFRAME['Rank'] = np.arange(1, len(BestMatches) + 1)
    myfulljsondata = DATAFRAME[:100]
    jsontable = []

    for i in tqdm(range(myfulljsondata.shape[0])):
        temp = myfulljsondata.iloc[i]
        jsontable.append(dict(temp))

    DATAFRAME = DATAFRAME[['Rank', 'Name', 'Keywords', 'Corr', 'pvalue', 'Signedcorrvalue', "ID"]]
    DATAFRAME.to_csv('media/matching data.csv', index=False)

    PairWise = pd.DataFrame()

    featurename = fname
    PairWise[featurename] = New_Feature_vector_dataframe.values

    for i in range(len(BestMatches[:12])):
        PairWise[BestMatches[i][0]] = hctsa.iloc[:, BestMatches[i][3]]

    pairwise_corr = PairWise.corr(method="spearman").abs()
    print(pairwise_corr)

    g = sns.clustermap(pairwise_corr, method="complete", annot=True, linewidth=0.5, square=True)

    columns = list(PairWise.columns)

    N = len(columns)
    wanted_label = featurename
    wanted_row = g.dendrogram_row.reordered_ind.index(columns.index(wanted_label))
    wanted_col = g.dendrogram_col.reordered_ind.index(columns.index(wanted_label))

    xywh_row = (0, wanted_row, N, 1)
    xywh_col = (wanted_col, 0, 1, N)
    for x, y, w, h in (xywh_row, xywh_col):
        print(x, y, w, h)
        g.ax_heatmap.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='Blue', lw=4, clip_on=True))
    g.ax_heatmap.tick_params(length=0)
    myfig = plt.gcf()
    buf = io.BytesIO()
    myfig.savefig(buf, format='png')

    buf.seek(0)

    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)

    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

    Scatterdataframe = pd.DataFrame()

    for i in range(12):
        Scatterdataframe[i] = hctsa.iloc[:, BestMatches[i][3]]

    Scatterdataframe[13] = New_Feature_vector_dataframe

    trace0 = go.Scatter(x=Scatterdataframe.iloc[:, 0].rank(), y=Scatterdataframe[13].rank(), mode="markers")
    trace1 = go.Scatter(x=Scatterdataframe.iloc[:, 1].rank(), y=Scatterdataframe[13].rank(), mode="markers")
    trace2 = go.Scatter(x=Scatterdataframe.iloc[:, 2].rank(), y=Scatterdataframe[13].rank(), mode="markers")
    trace3 = go.Scatter(x=Scatterdataframe.iloc[:, 3].rank(), y=Scatterdataframe[13].rank(), mode="markers")
    trace4 = go.Scatter(x=Scatterdataframe.iloc[:, 4].rank(), y=Scatterdataframe[13].rank(), mode="markers")
    trace5 = go.Scatter(x=Scatterdataframe.iloc[:, 5].rank(), y=Scatterdataframe[13].rank(), mode="markers")
    trace6 = go.Scatter(x=Scatterdataframe.iloc[:, 6].rank(), y=Scatterdataframe[13].rank(), mode="markers")
    trace7 = go.Scatter(x=Scatterdataframe.iloc[:, 7].rank(), y=Scatterdataframe[13].rank(), mode="markers")
    trace8 = go.Scatter(x=Scatterdataframe.iloc[:, 8].rank(), y=Scatterdataframe[13].rank(), mode="markers")
    trace9 = go.Scatter(x=Scatterdataframe.iloc[:, 9].rank(), y=Scatterdataframe[13].rank(), mode="markers")
    trace10 = go.Scatter(x=Scatterdataframe.iloc[:, 10].rank(), y=Scatterdataframe[13].rank(), mode="markers")
    trace11 = go.Scatter(x=Scatterdataframe.iloc[:, 11].rank(), y=Scatterdataframe[13].rank(), mode="markers")

    fig = go.FigureWidget(make_subplots(rows=3, cols=4, subplot_titles=(
        f"Correlation = {BestMatches[0][2]}", f"Correlation = {BestMatches[1][2]}",
        f"Correlation = {BestMatches[2][2]}",
        f"Correlation = {BestMatches[3][2]}", f"Correlation = {BestMatches[4][2]}",
        f"Correlation = {BestMatches[5][2]}", f"Correlation = {BestMatches[6][2]}",
        f"Correlation = {BestMatches[7][2]}",
        f"Correlation = {BestMatches[8][2]}", f"Correlation = {BestMatches[9][2]}",
        f"Correlation = {BestMatches[10][2]}",
        f"Correlation = {BestMatches[11][2]}")))
    fig.update_layout(template='plotly')

    fig.add_trace(trace0, 1, 1)
    fig.add_trace(trace1, 1, 2)
    fig.add_trace(trace2, 1, 3)
    fig.add_trace(trace3, 1, 4)
    fig.add_trace(trace4, 2, 1)
    fig.add_trace(trace5, 2, 2)
    fig.add_trace(trace6, 2, 3)
    fig.add_trace(trace7, 2, 4)
    fig.add_trace(trace8, 3, 1)
    fig.add_trace(trace9, 3, 2)
    fig.add_trace(trace10, 3, 3)
    fig.add_trace(trace11, 3, 4)

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

    fig.update_yaxes(title_text=featurename, row=1, col=1)
    fig.update_yaxes(title_text=featurename, row=2, col=1)
    fig.update_yaxes(title_text=featurename, row=3, col=1)

    fig.update_layout(showlegend=False, template='ggplot2', paper_bgcolor='rgba(0,0,0,0)',
                      margin=dict(
                          r=130
                      ))

    graph = fig.to_html(full_html=False, default_height=1200, default_width=1200)

    context = {
        "clusterdata": uri,
        "data": jsontable,
        'graph': graph,
        "totalmatches": totalmatches,
        "featurename": featurename
    }

    return render(request, "result.html", context)


def apiresult(request):
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
        safe_modules = ["math", "statistics", "numpy", "scipy", "pandas", "statsmodels", "sklearn"]
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
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "sorted": sorted,
                "round": round,
                "type": type,
                "complex": complex
            }
            exec(byte_code, restricted_globals, restricted_locals)
            return restricted_locals["result"]
        except SyntaxError as e:
            raise
        except Exception as e:
            raise

    def Execute_User_Code():
        user_code = usercode
        user_code += "\nresult = {0}(*args, **kwargs)".format(featurename)
        byte_code = compile_restricted(user_code, filename="<user_code>", mode="exec")
        for i in range(len(Alltimeseries)):
            featurevalue = execute_user_code(byte_code, Alltimeseries[i])
            New_feature_vector.append(featurevalue)

    try:
        New_feature_vector = []
        if request.method != 'POST':
            return JsonResponse({"stat": 5})
        featurename = request.POST['featurename']
        featurecode = request.FILES['featurecode']
        fs = FileSystemStorage()
        modulename = fs.save(featurecode.name, featurecode).replace(".py", "")
        MAX_ITER_LEN = 1000000000000000
        with open(f"media\{modulename}.py") as f:
            usercode = f.read()
        try:
            func_timeout(300, Execute_User_Code)
        except FunctionTimedOut:
            raise
        nan_fvector = int(pd.DataFrame(New_feature_vector).isna().sum())
        if int(nan_fvector) > 50:
            raise SyntaxError
        allData = mycol.find({}, {"NAME": 1, "KEYWORDS": 1, "ID": 1, "HCTSA_TIMESERIES_VALUE": 1})
        correlatedfeatures = []
        for data in allData:
            eachfeature = []
            if (int(pd.DataFrame(data['HCTSA_TIMESERIES_VALUE']).isna().sum()) + nan_fvector) < 50:
                coef, p = spearmanr(data['HCTSA_TIMESERIES_VALUE'], New_feature_vector, nan_policy="omit")
                if p < alpha:
                    eachfeature = [data['NAME'], p, format(abs(coef), '.3f'), int(data['ID']), data['KEYWORDS'],
                                   format(coef, '.3f')]
                    correlatedfeatures.append(eachfeature)
        BestMatches = sorted(correlatedfeatures, key=itemgetter(2))[::-1]
        DATAFRAME = pd.DataFrame(BestMatches)
        DATAFRAME.columns = ['NAME', 'PVALUE', 'ABSCOEF', 'ID', 'KEYWORDS', 'COEF']
        DATAFRAME['Rank'] = np.arange(1, len(BestMatches) + 1)
        PairWise = pd.DataFrame()
        PairWise[featurename] = New_feature_vector
        for i in range(12):
            temp = mycol.find_one({'ID': int(DATAFRAME.iloc[i, 3])}, {"HCTSA_TIMESERIES_VALUE": 1})
            PairWise[DATAFRAME.iloc[i, 0]] = temp['HCTSA_TIMESERIES_VALUE']
        pairwise_corr = PairWise.corr(method="spearman").abs()
        g = sns.clustermap(pairwise_corr, method="complete", annot=True, linewidth=0.5, square=True)
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
        myfig = plt.gcf()
        buf = io.BytesIO()
        myfig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        scatterPlotsData = {
            'xaxis': {
                'xdata': splittimeseries(PairWise[featurename].rank()),
                'xtit': featurename
            },
            'yaxes': []
        }
        TimeseriesNamesDivided = splittimeseries(AlltimeSeriesNames)
        for i in range(12):
            gr = {
                'title': "Correlation = " + str(DATAFRAME.iloc[i, 5]),
                'ytit': DATAFRAME.iloc[i, 0],
                'ydata': splittimeseries(PairWise[DATAFRAME.iloc[i, 0]].rank())
            }
            scatterPlotsData['yaxes'].append(gr)

        pairwise_corr = PairWise.corr(method="spearman")
        names = list(pairwise_corr.columns)
        networkGraph = {
            'nodes': [],
            'edges': []
        }
        for i in range(len(names)):
            networkGraph['nodes'].append({
                'id': i,
                'title': int(DATAFRAME['ID'].iloc[i])
            })
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                networkGraph['edges'].append({
                    'to': j,
                    'from': i,
                    'label': format(pairwise_corr[names[i]][names[j]], '.3f')
                })

        totalmatches = len(DATAFRAME)
        DATAFRAME["id"] = DATAFRAME["ID"].astype("int64")
        DATAFRAME = DATAFRAME.drop(['NAME'], axis=1)
        DATAFRAME = DATAFRAME.drop(['KEYWORDS'], axis=1)
        DATAFRAME = DATAFRAME.drop(['ABSCOEF'], axis=1)
        DATAFRAME = DATAFRAME.drop(['ID'], axis=1)
        res = []
        for index, row in list(DATAFRAME.iterrows()):
            res.append(dict(row))

        return JsonResponse(
            {"stat": 1, "tabledata": res, "totalmatches": totalmatches, "featurename": featurename, "heatmap": uri,
             "timeseriesnames": TimeseriesNamesDivided, "timeseriescategory": TimeSeriesCategory,
             "networkGraph": networkGraph, "scatterplotgraphs": scatterPlotsData})
    except SyntaxError as e:
        return JsonResponse({"stat": 2})
    except Exception as e:
        return JsonResponse({"stat": 3})
    except FunctionTimedOut:
        return JsonResponse({"stat": 4})


def contribute(request):
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
        safe_modules = ["math", "statistics", "numpy", "scipy", "pandas", "statsmodels", "sklearn"]
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
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "sorted": sorted,
                "round": round,
                "type": type,
                "complex": complex
            }
            exec(byte_code, restricted_globals, restricted_locals)
            return restricted_locals["result"]
        except SyntaxError as e:
            raise
        except Exception as e:
            raise

    def Execute_User_Code():
        user_code = usercode
        user_code += "\nresult = {0}(*args, **kwargs)".format(featurename)
        byte_code = compile_restricted(user_code, filename="<user_code>", mode="exec")
        for i in range(len(Alltimeseries)):
            featurevalue = execute_user_code(byte_code, Alltimeseries[i])
            New_feature_vector.append(featurevalue)

    New_feature_vector = []
    if request.method != 'POST':
        return JsonResponse({"stat": 5})
    featurename = request.POST['featurename']
    featurecode = request.FILES['featurecode']
    keywords = request.POST['keywords']
    fs = FileSystemStorage()
    modulename = fs.save(featurecode.name, featurecode).replace(".py", "")
    MAX_ITER_LEN = 1000000000000000
    with open(f"media\{modulename}.py") as f:
        usercode = f.read()

    try:
        try:
            func_timeout(300, Execute_User_Code)
        except FunctionTimedOut:
            raise
        nan_fvector = int(pd.DataFrame(New_feature_vector).isna().sum())
        if int(nan_fvector) > 50:
            raise SyntaxError
        allData = mycol.find({}, {"NAME": 1, "KEYWORDS": 1, "ID": 1, "HCTSA_TIMESERIES_VALUE": 1})
        pvalues = []
        coef = []
        print("herer")
        for data in allData:
            p, corr = 0, 0
            if (int(pd.DataFrame(data['HCTSA_TIMESERIES_VALUE']).isna().sum()) + nan_fvector) < 50:
                corr, p = spearmanr(data['HCTSA_TIMESERIES_VALUE'], New_feature_vector, nan_policy="omit")
            coef.append(str(format(corr, '.3f')))
            pvalues.append(str(format(p, '.3f')))
        ID = len(features) + 1

        dic = {
            "ID": ID,
            "NAME": featurename,
            "KEYWORDS": keywords,
            "HCTSA_TIMESERIES_VALUE": New_feature_vector,
            "COEF": coef,
            "PVALUE": pvalues
        }
        print(dic)
        features.append(dic)
        mycol.insert_one(dic)
        return JsonResponse({"stat": 1})
    except SyntaxError as e:
        return JsonResponse({"stat": 2})
    except Exception as e:
        print(e)
        return JsonResponse({"stat": 3})
    except FunctionTimedOut:
        return JsonResponse({"stat": 4})


def result(request):
    import pandas as pd
    import warnings
    from csv import reader
    from tqdm import tqdm
    from scipy.stats import spearmanr
    from operator import itemgetter
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    from matplotlib.patches import Rectangle
    import numpy as np
    from RestrictedPython import safe_builtins, compile_restricted
    from RestrictedPython.Eval import default_guarded_getitem
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    Alltimeseries = []
    missing = []
    finalfeatures = []
    New_feature_vector = []

    # Reading code file and function name

    if request.method == 'POST':

        # Taking user's input

        featurename = request.POST['featurename']
        featurecode = request.FILES['featurecode']
        fs = FileSystemStorage()

        modulename = fs.save(featurecode.name, featurecode).replace(".py", "")
        print(modulename)

        MAX_ITER_LEN = 1000000000000000

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
            safe_modules = ["math", "statistics", "numpy", "scipy", "pandas", "statsmodels", "sklearn"]
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
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "sorted": sorted,
                    "round": round,
                    "type": type,
                    "complex": complex
                }

                exec(byte_code, restricted_globals, restricted_locals)

                return restricted_locals["result"]
            except SyntaxError as e:
                raise
            except Exception as e:
                raise

        #  Reading all required files - timeseries data, hctsa datamatrix and keywords 

        with open('hctsa_timeseries-data.csv', 'r') as read_obj:
            csv_reader = reader(read_obj)
            li = list(csv_reader)
            for i in tqdm(li):
                Alltimeseries.append(list(map(float, i)))

        # Reading hctsa datamatrix

        hctsa = pd.read_csv('hctsa_datamatrix.csv')
        keywords = pd.read_csv('hctsa_features.csv')

        # Reading user's code as string
        with open(f"media\{modulename}.py") as f:
            usercode = f.read()
            print(usercode)

        def usercodeexec():
            for i in tqdm(range(len(Alltimeseries))):
                featurevalue = execute_user_code(usercode, featurename, Alltimeseries[i])
                New_feature_vector.append(featurevalue)

        from func_timeout import func_timeout, FunctionTimedOut

        def Execute_User_Code():
            user_code = usercode
            user_code += "\nresult = {0}(*args, **kwargs)".format(featurename)
            byte_code = compile_restricted(user_code, filename="<user_code>", mode="exec")
            for i in tqdm(range(len(Alltimeseries))):
                featurevalue = execute_user_code(byte_code, Alltimeseries[i])
                New_feature_vector.append(featurevalue)

        # Passing timeseries data to user's uploaded function

        try:
            mytemplate = "result.html"
            try:

                func_timeout(300, Execute_User_Code)
            except FunctionTimedOut:
                raise

            #  For handling too many Nan values

            if int(pd.DataFrame(New_feature_vector).isna().sum()) > 50:
                raise SyntaxError

            # Removing hctsa datamatrix's column having more than 70% missing values

            for i in tqdm(range(len(hctsa.columns))):
                if (hctsa.iloc[:, i].isna().sum() * 100 / hctsa.iloc[:, i].shape[0]) > 70:
                    missing.append(hctsa.columns[i])
                else:
                    #  Column number of hctsa datamatrix which have less <70% missing values are stored in finalfeatures list.

                    finalfeatures.append(i)

            #   Comparing features

            alpha = 0.05
            New_Feature_vector_dataframe = pd.DataFrame(New_feature_vector)

            #   Calculating total Nan values in feature vector 

            nan_fvector = int(New_Feature_vector_dataframe.isna().sum())

            correlatedfeatures = []

            for i in tqdm(finalfeatures):
                eachfeature = []
                if (hctsa.iloc[:, i].isna().sum() + nan_fvector) < 50:
                    coef, p = spearmanr(hctsa.iloc[:, i], New_Feature_vector_dataframe.values, nan_policy="omit")
                    if p < alpha:
                        eachfeature = [hctsa.columns[i], p, format(abs(coef), '.3f'), i,
                                       *keywords.iloc[i:i + 1, 2].values, format(coef, '.3f')]
                        correlatedfeatures.append(eachfeature)

            BestMatches = sorted(correlatedfeatures, key=itemgetter(2))[::-1]

            totalmatches = len(BestMatches)

            # for displaying all row columns in one screen

            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)

            # Preparing Dataframe for visualization

            DATAFRAME = pd.DataFrame(BestMatches)
            DATAFRAME.columns = ['Name', 'pvalue', 'Corr', 'ColumnId', 'Keywords', 'Signedcorrvalue']
            DATAFRAME['Rank'] = np.arange(1, len(BestMatches) + 1)

            myfulljsondata = DATAFRAME[:100]
            jsontable = []

            for i in tqdm(range(myfulljsondata.shape[0])):
                temp = myfulljsondata.iloc[i]
                jsontable.append(dict(temp))

            # Creating csv file for downloading by the user

            DATAFRAME = DATAFRAME[['Rank', 'Name', 'Keywords', 'Corr', 'pvalue', 'Signedcorrvalue']]
            DATAFRAME.to_csv('media/matching data.csv', index=False)

            # visualization

            PairWise = pd.DataFrame()

            # appending user's feature vector to pairwise datframe

            PairWise[featurename] = New_feature_vector

            for i in range(len(BestMatches[:12])):
                PairWise[BestMatches[i][0]] = hctsa.iloc[:, BestMatches[i][3]]

            pairwise_corr = PairWise.corr(method="spearman").abs()

            g = sns.clustermap(pairwise_corr, method="complete", annot=True, linewidth=0.5, square=True)

            # Highlighting user's row and column in pairwise correlation with patch

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
            myfig = plt.gcf()
            buf = io.BytesIO()
            myfig.savefig(buf, format='png')

            buf.seek(0)

            string = base64.b64encode(buf.read())

            uri = urllib.parse.quote(string)

            # Using plotly for creating interactive graphs

            import plotly.graph_objs as go

            from plotly.subplots import make_subplots

            Scatterdataframe = pd.DataFrame()

            for i in range(12):
                Scatterdataframe[i] = hctsa.iloc[:, BestMatches[i][3]]

            Scatterdataframe[13] = New_Feature_vector_dataframe.iloc[:, 0]

            # For generating  x and y axis scatter plots suing plotly

            trace0 = go.Scatter(x=Scatterdataframe.iloc[:, 0].rank(), y=Scatterdataframe[13].rank(), mode="markers")
            trace1 = go.Scatter(x=Scatterdataframe.iloc[:, 1].rank(), y=Scatterdataframe[13].rank(), mode="markers")
            trace2 = go.Scatter(x=Scatterdataframe.iloc[:, 2].rank(), y=Scatterdataframe[13].rank(), mode="markers")
            trace3 = go.Scatter(x=Scatterdataframe.iloc[:, 3].rank(), y=Scatterdataframe[13].rank(), mode="markers")
            trace4 = go.Scatter(x=Scatterdataframe.iloc[:, 4].rank(), y=Scatterdataframe[13].rank(), mode="markers")
            trace5 = go.Scatter(x=Scatterdataframe.iloc[:, 5].rank(), y=Scatterdataframe[13].rank(), mode="markers")
            trace6 = go.Scatter(x=Scatterdataframe.iloc[:, 6].rank(), y=Scatterdataframe[13].rank(), mode="markers")
            trace7 = go.Scatter(x=Scatterdataframe.iloc[:, 7].rank(), y=Scatterdataframe[13].rank(), mode="markers")
            trace8 = go.Scatter(x=Scatterdataframe.iloc[:, 8].rank(), y=Scatterdataframe[13].rank(), mode="markers")
            trace9 = go.Scatter(x=Scatterdataframe.iloc[:, 9].rank(), y=Scatterdataframe[13].rank(), mode="markers")
            trace10 = go.Scatter(x=Scatterdataframe.iloc[:, 10].rank(), y=Scatterdataframe[13].rank(), mode="markers")
            trace11 = go.Scatter(x=Scatterdataframe.iloc[:, 11].rank(), y=Scatterdataframe[13].rank(), mode="markers")

            # adding correlation value as title for all 12 plots

            fig = go.FigureWidget(make_subplots(rows=3, cols=4, subplot_titles=(
                f"Correlation = {BestMatches[0][2]}", f"Correlation = {BestMatches[1][2]}",
                f"Correlation = {BestMatches[2][2]}", f"Correlation = {BestMatches[3][2]}",
                f"Correlation = {BestMatches[4][2]}",
                f"Correlation = {BestMatches[5][2]}", f"Correlation = {BestMatches[6][2]}",
                f"Correlation = {BestMatches[7][2]}", f"Correlation = {BestMatches[8][2]}",
                f"Correlation = {BestMatches[9][2]}",
                f"Correlation = {BestMatches[10][2]}", f"Correlation = {BestMatches[11][2]}")))
            fig.update_layout(template='plotly')

            # Generating 12 scatter plots on single page

            fig.add_trace(trace0, 1, 1)
            fig.add_trace(trace1, 1, 2)
            fig.add_trace(trace2, 1, 3)
            fig.add_trace(trace3, 1, 4)
            fig.add_trace(trace4, 2, 1)
            fig.add_trace(trace5, 2, 2)
            fig.add_trace(trace6, 2, 3)
            fig.add_trace(trace7, 2, 4)
            fig.add_trace(trace8, 3, 1)
            fig.add_trace(trace9, 3, 2)
            fig.add_trace(trace10, 3, 3)
            fig.add_trace(trace11, 3, 4)

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

            fig.update_yaxes(title_text=featurename, row=1, col=1)
            fig.update_yaxes(title_text=featurename, row=2, col=1)
            fig.update_yaxes(title_text=featurename, row=3, col=1)

            # plots configuration (papercolor,margin,show legends)

            fig.update_layout(showlegend=False, template='ggplot2', paper_bgcolor='rgba(0,0,0,0)',
                              margin=dict(
                                  r=130
                              ))

            # configuring graph into html for rendering.

            graph = fig.to_html(full_html=False, default_height=1200, default_width=1200)

            # Dictionary for passsing the variables to be rendered on result page

            context = {
                "clusterdata": uri,
                "data": jsontable,
                'graph': graph,
                "totalmatches": totalmatches,
                "featurename": featurename
            }



        # change the template so as to redirect user to syntax page if syntax error encounters

        except SyntaxError as e:
            print(e)
            context = {}
            mytemplate = "syntaxerror.html"


        # change the template so as to redirect user to warning page if unwanted imports encounters

        except Exception as e:
            print(e)
            context = {}
            mytemplate = "warningpage.html"

        except FunctionTimedOut:
            context = {}
            mytemplate = "timeout.html"

        return render(request, mytemplate, context)
