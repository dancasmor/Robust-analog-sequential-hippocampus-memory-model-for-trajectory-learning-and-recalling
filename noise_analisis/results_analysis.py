import pandas as pd
import statistics
import matplotlib.pyplot as plt


def load_excell_data(filename):
    df = pd.read_excel(filename)
    data = df.to_dict()
    return data


def delete_nan_values(data):
    dataOut = {}
    for key, value in data.items():
        if not(pd.isna(value)):
            dataOut[key] = value
    return dataOut


def get_data_processed(excelldata):
    """
    Converts the data in the excel file into processed results at the output

    :param excelldata: data to be processed
    :return: experimentMetaData, resultsByFrequency, resultsByFrequencyAndPathLength
            experimentMetaData = {"experiment": var, "subexperiment": var, "noiseEffect": var, "numRepetitions": var,
                                  "numExperiments": var, "frequencyRates": var, "pathInfo": var}
            resultsByFrequency = {f0:resf0, f1:resf1, f2:resf2, ...}
                fi = frequency rate
                resfi = results for frequency rate fi: resfi = {rep0: resfir0, rep1: resfir1, ...}
                    repi = repetition id
                    resfiri = results for frequency rate fi and repetition id i: resfiri = {"numRecalls": var, "numRecallHits": var,
                                                                                            "numPaths": var, "numPathHits": var,
                                                                                            "recallHitRate": list, "pathHitRate": list,
                                                                                            "pathID": list, "numRecallsList": list}
    """

    # Get meta info of experiment
    frequencyRates = list(set(excelldata["poisson_freq"].values()))
    numExperiments = len(excelldata["rep_id"])
    # pathLength = list(set(delete_nan_values(excelldata["path_steps"]).values()))
    pathIDs = list(delete_nan_values(excelldata["path_id"]).values())
    pathSteps = list(delete_nan_values(excelldata["path_steps"]).values())
    pathInfo = {}
    for i in range(len(pathIDs)):
        pathInfo.update({pathIDs[i]: pathSteps[i]})
    experimentMetaData = {"experiment": excelldata["experiment"][0], "subexperiment": excelldata["subexperiment"][0],
                          "noiseEffect": excelldata["noise_effect"][0], "numRepetitions": excelldata["num_repetitions"][0],
                          "numExperiments": numExperiments, "frequencyRates": frequencyRates, "pathInfo": pathInfo}

    # Prepare results variables
    resultsByFrequency = {}
    for freq in frequencyRates:
        resultsByFrequency.update({freq: {}})
        for rep in range(1, int(excelldata["num_repetitions"][0])+1):
            resultsByFrequency[freq].update({rep: {"numRecalls": 0, "numRecallHits": 0,
                                                   "numPaths": 0, "numPathHits": 0,
                                                   "recallHitRate": [], "pathHitRate": [], "pathID": [],
                                                   "numRecallsList": []}})
    # Process information
    for id in range(numExperiments):
        resultsForFixedFreq = resultsByFrequency[excelldata["poisson_freq"][id]][excelldata["rep_id"][id]]
        resultsForFixedFreq["numRecalls"] = resultsForFixedFreq["numRecalls"] + excelldata["num_recall_hit"][id] + \
                                            excelldata["num_recall_miss"][id]
        resultsForFixedFreq["numRecallHits"] = resultsForFixedFreq["numRecallHits"] + excelldata["num_recall_hit"][id]
        resultsForFixedFreq["numPaths"] = resultsForFixedFreq["numPaths"] + 1
        resultsForFixedFreq["numPathHits"] = resultsForFixedFreq["numPathHits"] + excelldata["path_hit"][id]
        resultsForFixedFreq["recallHitRate"].append(excelldata["num_recall_hit"][id] /
                                                    (excelldata["num_recall_hit"][id] + excelldata["num_recall_miss"][id]))
        resultsForFixedFreq["pathHitRate"].append(excelldata["path_hit"][id])
        resultsForFixedFreq["pathID"].append(excelldata["path_id_rep"][id])
        resultsForFixedFreq["numRecallsList"].append(excelldata["num_recall_hit"][id] + excelldata["num_recall_miss"][id])

    # Return data (resultsByFrequency order keys by freq)
    return experimentMetaData, dict(sorted(resultsByFrequency.items()))


def get_metrics(resultsByParameter, pathInfo):
    """
    Get metrics from data
    :param resultsByParameter: data to get metrics separated by a parameter
    :param pathInfo: experiment path Info
    :return: metricsByParameter
            metricsByParameter = {param: {"recallHitMean": var, "recallHitStd": var, "pathHitMean": var,
                                          "pathHitStd": var, "totalRecall": var, "metricsByPathLength": metricsByParameter*}}
                    "metricsByPathLength": metricsByParameter* -> {pathLength: {"recallHitMean": var,
                                                                        "recallHitStd": var, "pathHitMean": var,
                                                                        "pathHitStd": var, "totalRecall": var,
                                                                        "numberOfPath": var}
    """
    # Get path lengths
    pathLengths = list(set(list(pathInfo.values())))
    # Create output variable
    metricsByParameter = {}
    for param in resultsByParameter.keys():
        metricsByParameter.update({param: {"recallHitMean": 0, "recallHitStd": 0, "pathHitMean": 0,
                                           "pathHitStd": 0, "totalRecall": 0, "metricsByPathLength": {}}})
        for pathLength in pathLengths:
            metricsByParameter[param]["metricsByPathLength"].update({pathLength: {"recallHitMean": 0, "recallHitStd": 0,
                                                                                  "pathHitMean": 0, "pathHitStd": 0,
                                                                                  "totalRecall": 0, "numberOfPath": 0}})
    # Get metrics
    for freq, dataByFreq in resultsByParameter.items():
        recallHit, recallTotal, pathHit, pathHitTotal = [0]*4
        recallHitRate, pathHitRate = [], []
        recallHitRateByPathLength, pathHitRateByPathLength = [[] for i in range(len(pathLengths))], [[] for i in range(len(pathLengths))]
        recallTotalByPathLength = [0 for i in range(len(pathLengths))]
        numberOfPathByLength = [0 for i in range(len(pathLengths))]
        # Get data from all repetitions
        for rep, dataByRep in dataByFreq.items():
            recallHit = recallHit + dataByRep["numRecallHits"]
            recallTotal = recallTotal + dataByRep["numRecalls"]
            pathHit = pathHit + dataByRep["numPathHits"]
            pathHitTotal = pathHitTotal + dataByRep["numPaths"]
            recallHitRate = recallHitRate + dataByRep["recallHitRate"]
            pathHitRate = pathHitRate + dataByRep["pathHitRate"]
            # Order data by path length
            for index, ID in enumerate(dataByRep["pathID"]):
                pathLengthsIndex = pathLengths.index(pathInfo[ID])
                recallHitRateByPathLength[pathLengthsIndex].append(dataByRep["recallHitRate"][index])
                pathHitRateByPathLength[pathLengthsIndex].append(dataByRep["pathHitRate"][index])
                recallTotalByPathLength[pathLengthsIndex] = recallTotalByPathLength[pathLengthsIndex] + dataByRep["numRecallsList"][index]
                numberOfPathByLength[pathLengthsIndex] = numberOfPathByLength[pathLengthsIndex] + 1
        # Total recalls
        metricsByParameter[freq]["totalRecall"] = recallTotal
        # Mean
        metricsByParameter[freq]["recallHitMean"] = recallHit / recallTotal
        metricsByParameter[freq]["pathHitMean"] = pathHit / pathHitTotal
        # Std deviation
        metricsByParameter[freq]["recallHitStd"] = statistics.stdev(recallHitRate, metricsByParameter[freq]["recallHitMean"])
        metricsByParameter[freq]["pathHitStd"] = statistics.stdev(pathHitRate, metricsByParameter[freq]["pathHitMean"])

        # Metrics for each path length
        for index, length in enumerate(pathLengths):
            metricsByLength = metricsByParameter[freq]["metricsByPathLength"][length]
            # Total recalls
            metricsByLength["totalRecall"] = recallTotalByPathLength[index]
            # Number of path with this length
            metricsByLength["numberOfPath"] = numberOfPathByLength[index]
            # Mean
            metricsByLength["recallHitMean"] = statistics.mean(recallHitRateByPathLength[index])
            metricsByLength["pathHitMean"] = statistics.mean(pathHitRateByPathLength[index])
            # Std deviation
            metricsByLength["recallHitStd"] = statistics.stdev(recallHitRateByPathLength[index])
            metricsByLength["pathHitStd"] = statistics.stdev(pathHitRateByPathLength[index])
    return metricsByParameter


def create_plot_1_parameter(xData, yData, upErrData, lowErrData, labels, color, plotName, plot, yTicks=True):
    """
    Create the plot with the input data for only 1 parameter (std or mean)

    :param xData: x axis values
    :param yData: y axis values
    :param upErrData: up error values
    :param lowErrData: low error values
    :param labels: x and y axis labels
    :param color: color of points and connection line (no error lines)
    :param plotName: name of the output plot file
    :param plot: if plot or not the figure
    :param yTicks: if put yTicks or set it default
    :return:
    """
    plt.figure(figsize=(8, 8), dpi=400)
    plt.xlabel(labels["x"])
    plt.ylabel(labels["y"])

    plt.errorbar(xData, yData, [lowErrData, upErrData], capsize=3, linestyle="none", fmt='none', ecolor="red")
    plt.plot(xData, yData, color=color, marker="o")

    plt.xticks(xData)
    if yTicks:
        plt.yticks([x * 0.1 for x in range(0, 11)])

    plt.savefig("results/analysis/" + plotName + ".png", bbox_inches='tight')
    if plot:
        plt.show()
    plt.close()


def create_plot_several_parameter(xData, yData, upErrData, lowErrData, dataLabels, axisLabels, color, plotName, plot, allExperiments=False, yTicks=True):
    """
    Create the plot with the input data for several parameters

    :param xData: (list) x axis values
    :param yData: (list) y axis values
    :param upErrData: (list) up error values
    :param lowErrData: (list) low error values
    :param dataLabels: (list) label of each parameter data
    :param axisLabels: x and y axis labels
    :param color: (list) color of points and connection line (no error lines)
    :param plotName: name of the output plot file
    :param plot: if plot or not the figure
    :param allExperiments: if plot is for all experiments (True) or only for single experiments (False)
    :param yTicks: if put yTicks or set it default
    :return:
    """
    xticks = []
    for id in range(len(xData)):
        xticks = xticks + xData[id]

    if allExperiments:
        barWidth = max(xticks) * 0.04
        plt.figure(figsize=(18, 14), dpi=400)
    else:
        barWidth = max(xticks) * 0.02
        plt.figure(figsize=(8, 8), dpi=400)
    plt.xlabel(axisLabels["x"])
    plt.ylabel(axisLabels["y"])

    for id in range(len(xData)):
        if allExperiments:
            xDataTmp = [x + id * barWidth + index * len(xData) * barWidth for index, x in enumerate(xData[id])]
        else:
            xDataTmp = [x + id * barWidth for x in xData[id]]
        # plt.plot(xDataTmp, yData[id], color=color[id], marker="o", linestyle='None')
        plt.errorbar(xDataTmp, yData[id], [lowErrData[id], upErrData[id]], capsize=3, linestyle="none", fmt='o', c="red", ecolor="red")
        plt.bar(xDataTmp, yData[id], color=color[id], linestyle='None', width=barWidth, label=dataLabels[id])

    xticks = list(set(xticks))
    xticks.sort()
    if allExperiments:
        xtickspos = [x - barWidth / 2 + (len(xData) * barWidth) / 2 + index * len(xData) * barWidth for index, x in enumerate(xticks)]
    else:
        xtickspos = [x + ((len(xData)-1)*barWidth)/len(xData) for x in xticks]
    plt.xticks(xtickspos, xticks)
    if yTicks:
        plt.yticks([x * 0.1 for x in range(0, 11)])

    if allExperiments:
        plt.legend(bbox_to_anchor=(1.025, 1.025), framealpha=1)
    else:
        plt.legend(bbox_to_anchor=(0.65, 1.025), ncol=len(xData), framealpha=1)

    plt.savefig("results/analysis/" + plotName + ".png", bbox_inches='tight')
    if plot:
        plt.show()
    plt.close()


def create_plot_2_parameter(xData, yData, upErrData, lowErrData, dataLabels, axisLabels, color, plotName, plot, yTicks=True):
    """
    Create the plot with the input data for only 1 parameter (std or mean)

    :param xData: (list 2-values) x axis values
    :param yData: (list 2-values) y axis values
    :param upErrData: (list 2-values) up error values
    :param lowErrData: (list 2-values) low error values
    :param dataLabels: (list 2-values) label of each parameter data
    :param axisLabels: x, y1 and y2 axis labels
    :param color: (list 2-values) color of points and connection line (no error lines)
    :param plotName: name of the output plot file
    :param plot: if plot or not the figure
    :param yTicks: if put yTicks or set it default
    :return:
    """
    barWidth = max(xData[0]) * 0.02

    fig, ax1 = plt.subplots(figsize=(8, 8), dpi=400)
    plt.xlabel(axisLabels["x"])

    ax2 = ax1.twinx()

    ax1.errorbar(xData[0], yData[0], [lowErrData[0], upErrData[0]], capsize=3, linestyle="none", fmt='none', ecolor="red")
    plt1 = ax1.plot(xData[0], yData[0], color=color[0], marker="o", label=dataLabels[0])

    # xDataTmp = [x + barWidth for x in xData[1]]
    ax2.errorbar(xData[1], yData[1], [lowErrData[1], upErrData[1]], capsize=3, linestyle="none", fmt='none', ecolor="red")
    plt2 = ax2.plot(xData[1], yData[1], color=color[1], marker="o", label=dataLabels[1])

    ax1.set_ylabel(axisLabels["y1"])
    ax2.set_ylabel(axisLabels["y2"])

    xtickspos = [x + ((len(xData) - 1) * barWidth) / len(xData) for x in xData[0]]
    plt.xticks(xtickspos, xData[0])
    ax1.yaxis.set_ticks([x * 0.1 for x in range(0, 11)])
    ax2.yaxis.set_ticks([x for x in range(0, max(yData[1]), 50)])

    # plt.legend(bbox_to_anchor=(0.65, 1.025), ncol=len(xData), framealpha=1)
    plts = plt1 + plt2
    labs = [p.get_label() for p in plts]
    ax1.legend(plts, labs, bbox_to_anchor=(0.65, 1.05))

    plt.savefig("results/analysis/" + plotName + ".png", bbox_inches='tight')
    if plot:
        plt.show()
    plt.close()


def generate_individual_plots(metricsByFrequency, phaseName, phase, noiseEffectName, noiseEffect, plot, pathInfo):
    # Get path lengths
    pathLengths = list(set(list(pathInfo.values())))
    # Process metrics data to plot it
    xDataRecall, yDataRecall, upErrDataRecall, lowErrDataRecall = [], [], [], []
    xDataPath, yDataPath, upErrDataPath, lowErrDataPath = [], [], [], []
    xDataNumRecall, yDataNumRecall = [], []
    xDataRecallByPathLength, yDataRecallByPathLength, upErrDataRecallByPathLength, lowErrDataRecallByPathLength = \
        [[] for i in pathLengths], [[] for i in pathLengths], [[] for i in pathLengths], [[] for i in pathLengths]
    xDataPathByPathLength, yDataPathByPathLength, upErrDataPathByPathLength, lowErrDataPathByPathLength = \
        [[] for i in pathLengths], [[] for i in pathLengths], [[] for i in pathLengths], [[] for i in pathLengths]
    xDataNumRecallByPathLength, yDataNumRecallByPathLength = [[] for i in pathLengths], [[] for i in pathLengths]
    for freq, metrics in metricsByFrequency.items():
        # Num recall
        xDataNumRecall.append(freq)
        yDataNumRecall.append(metrics["totalRecall"])
        # Recall
        xDataRecall.append(freq)
        yDataRecall.append(metrics["recallHitMean"])
        if metrics["recallHitMean"] + metrics["recallHitStd"] > 1:
            upErrDataRecall.append(1-metrics["recallHitMean"])
        else:
            upErrDataRecall.append(metrics["recallHitStd"])
        if metrics["recallHitMean"] - metrics["recallHitStd"] < 0:
            lowErrDataRecall.append(0+metrics["recallHitMean"])
        else:
            lowErrDataRecall.append(metrics["recallHitStd"])
        # Path
        xDataPath.append(freq)
        yDataPath.append(metrics["pathHitMean"])
        if metrics["pathHitMean"] + metrics["pathHitStd"] > 1:
            upErrDataPath.append(1 - metrics["pathHitMean"])
        else:
            upErrDataPath.append(metrics["pathHitStd"])
        if metrics["pathHitMean"] - metrics["pathHitStd"] < 0:
            lowErrDataPath.append(0 + metrics["pathHitMean"])
        else:
            lowErrDataPath.append(metrics["pathHitStd"])

        # Data for each path length
        for length, metricsByPathLength in metrics["metricsByPathLength"].items():
            # Num recall
            xDataNumRecallByPathLength[pathLengths.index(length)].append(freq)
            yDataNumRecallByPathLength[pathLengths.index(length)].append(metricsByPathLength["totalRecall"]/metricsByPathLength["numberOfPath"])
            # Recall
            xDataRecallByPathLength[pathLengths.index(length)].append(freq)
            yDataRecallByPathLength[pathLengths.index(length)].append(metricsByPathLength["recallHitMean"])
            if metricsByPathLength["recallHitMean"] + metricsByPathLength["recallHitStd"] > 1:
                upErrDataRecallByPathLength[pathLengths.index(length)].append(1 - metricsByPathLength["recallHitMean"])
            else:
                upErrDataRecallByPathLength[pathLengths.index(length)].append(metricsByPathLength["recallHitStd"])
            if metricsByPathLength["recallHitMean"] - metricsByPathLength["recallHitStd"] < 0:
                lowErrDataRecallByPathLength[pathLengths.index(length)].append(0 + metricsByPathLength["recallHitMean"])
            else:
                lowErrDataRecallByPathLength[pathLengths.index(length)].append(metricsByPathLength["recallHitStd"])
            # Path
            xDataPathByPathLength[pathLengths.index(length)].append(freq)
            yDataPathByPathLength[pathLengths.index(length)].append(metricsByPathLength["pathHitMean"])
            if metricsByPathLength["pathHitMean"] + metricsByPathLength["pathHitStd"] > 1:
                upErrDataPathByPathLength[pathLengths.index(length)].append(
                    1 - metricsByPathLength["pathHitMean"])
            else:
                upErrDataPathByPathLength[pathLengths.index(length)].append(metricsByPathLength["pathHitStd"])
            if metricsByPathLength["pathHitMean"] - metricsByPathLength["pathHitStd"] < 0:
                lowErrDataPathByPathLength[pathLengths.index(length)].append(
                    0 + metricsByPathLength["pathHitMean"])
            else:
                lowErrDataPathByPathLength[pathLengths.index(length)].append(metricsByPathLength["pathHitStd"])

    # Get plots
    #   + Recall hit
    plotName = str(phaseName[phase]) + "_" + str(noiseEffectName[noiseEffect]) + "_recall_hit_rate"
    colorRecall = "#7FB3D5"
    create_plot_1_parameter(xDataRecall, yDataRecall, upErrDataRecall, lowErrDataRecall,
             {"x": "Poisson generator frequency rate (Hz)", "y": "Recall hit rate"},
             colorRecall, plotName, plot)
    #   + Path hit
    plotName = str(phaseName[phase]) + "_" + str(noiseEffectName[noiseEffect]) + "_path_hit_rate"
    colorPath = "#BB8FCE"
    create_plot_1_parameter(xDataPath, yDataPath, upErrDataPath, lowErrDataPath,
             {"x": "Poisson generator frequency rate (Hz)", "y": "Path hit rate"},
             colorPath, plotName, plot)
    #   + Recall and Hit parameters
    plotName = str(phaseName[phase]) + "_" + str(noiseEffectName[noiseEffect]) + "_both_hit_rate"
    color = [colorRecall, colorPath]
    create_plot_several_parameter([xDataRecall, xDataPath], [yDataRecall, yDataPath], [upErrDataRecall, upErrDataPath],
                            [lowErrDataRecall, lowErrDataPath], ["Recall", "Path"],
                            {"x": "Poisson generator frequency rate (Hz)", "y": "Hit rate"}, color, plotName, plot)
    #   + Num recalls
    plotName = str(phaseName[phase]) + "_" + str(noiseEffectName[noiseEffect]) + "_num_recalls"
    colorNumRecall = "#F39C12"
    create_plot_1_parameter(xDataNumRecall, yDataNumRecall, [0]*len(xDataNumRecall), [0]*len(xDataNumRecall),
                            {"x": "Poisson generator frequency rate (Hz)", "y": "Number of recall operations"},
                            colorNumRecall, plotName, plot, False)
    #   + Num recall and recall hit parameters
    plotName = str(phaseName[phase]) + "_" + str(noiseEffectName[noiseEffect]) + "_recallhit_recallnum"
    color = [colorRecall, colorNumRecall]
    create_plot_2_parameter([xDataRecall, xDataNumRecall], [yDataRecall, yDataNumRecall], [upErrDataRecall, [0]*len(xDataNumRecall)],
                            [lowErrDataRecall, [0]*len(xDataNumRecall)], ["Recall hit", "Num Recall"],
                            {"x": "Poisson generator frequency rate (Hz)", "y1": "Hit rate", "y2": "Num recall"},
                            color, plotName, plot)
    #   + Recall hit for all path length
    plotName = str(phaseName[phase]) + "_" + str(noiseEffectName[noiseEffect]) + "_recall_hit_rate_by_path_length"
    dataLabels = []
    for length in pathLengths:
        dataLabels.append("pl="+str(length))
    color = ["#7D3C98", "#138D75", "#B9770E", "#A93226"]
    create_plot_several_parameter(xDataRecallByPathLength, yDataRecallByPathLength, upErrDataRecallByPathLength,
                                  lowErrDataRecallByPathLength, dataLabels,
                                  {"x": "Poisson generator frequency rate (Hz)", "y": "Recall hit rate"}, color,
                                  plotName, plot, allExperiments=True)
    #   + Recall hit for all path length
    plotName = str(phaseName[phase]) + "_" + str(noiseEffectName[noiseEffect]) + "_path_hit_rate_by_path_length"
    create_plot_several_parameter(xDataPathByPathLength, yDataPathByPathLength, upErrDataPathByPathLength,
                                  lowErrDataPathByPathLength, dataLabels,
                                  {"x": "Poisson generator frequency rate (Hz)", "y": "Path hit rate"}, color,
                                  plotName, plot, allExperiments=True)
    #   + Num recalls for all path length
    plotName = str(phaseName[phase]) + "_" + str(noiseEffectName[noiseEffect]) + "_num_recalls_by_path_length"
    create_plot_several_parameter(xDataNumRecallByPathLength, yDataNumRecallByPathLength,
                                  [[0] * len(xDataNumRecall) for i in range(len(pathLengths))],
                                  [[0] * len(xDataNumRecall) for i in range(len(pathLengths))], dataLabels,
                                  {"x": "Poisson generator frequency rate (Hz)", "y": "Number of recall operations"},
                                  color, plotName, plot, allExperiments=True, yTicks=False)
    metricsProcessed = {"xDataRecall": xDataRecall, "yDataRecall": yDataRecall, "upErrDataRecall": upErrDataRecall,
                        "lowErrDataRecall": lowErrDataRecall, "xDataPath": xDataPath, "yDataPath": yDataPath,
                        "upErrDataPath": upErrDataPath, "lowErrDataPath": lowErrDataPath,
                        "xDataNumRecall": xDataNumRecall, "yDataNumRecall": yDataNumRecall}
    return metricsProcessed


def generate_general_plots(allMetricsProcessed, plot):
    # Format information
    # color = ["#D98880", "#566573", "#BB8FCE", "#D5DBDB", "#73C6B6", "#F8C471", "#F7DC6F", "#A9CCE3", "#F5B7B1"]
    color = ["#7D3C98", "#A569BD", "#D2B4DE", "#138D75", "#45B39D", "#A2D9CE", "#B9770E", "#F39C12", "#F8C471"]
    xDataNumRecall, yDataNumRecall, upErrDataNumRecall, lowErrDataNumRecall = [], [], [], []
    xDataRecall, yDataRecall, upErrDataRecall, lowErrDataRecall, dataLabels = [], [], [], [], []
    xDataPath, yDataPath, upErrDataPath, lowErrDataPath = [], [], [], []
    for label, metrics in allMetricsProcessed.items():
        if label == "Learn-Cont":
            # Num recall
            previousxDataList = list(allMetricsProcessed.values())[0]["xDataNumRecall"]
            xDataNumRecall.append(previousxDataList)
            yDataNumRecall.append([metrics["yDataNumRecall"][0]]*len(previousxDataList))
            upErrDataNumRecall.append([0] * len(previousxDataList))
            lowErrDataNumRecall.append([0] * len(previousxDataList))
            # Recall
            previousxDataList = list(allMetricsProcessed.values())[0]["xDataRecall"]
            xDataRecall.append(previousxDataList)
            yDataRecall.append([metrics["yDataRecall"][0]]*len(previousxDataList))
            upErrDataRecall.append([metrics["upErrDataRecall"][0]]*len(previousxDataList))
            lowErrDataRecall.append([metrics["lowErrDataRecall"][0]]*len(previousxDataList))
            # Path
            previousxDataList = list(allMetricsProcessed.values())[0]["xDataPath"]
            xDataPath.append(previousxDataList)
            yDataPath.append([metrics["yDataPath"][0]] * len(previousxDataList))
            upErrDataPath.append([metrics["upErrDataPath"][0]] * len(previousxDataList))
            lowErrDataPath.append([metrics["lowErrDataPath"][0]] * len(previousxDataList))
        else:
            # Num recall
            xDataNumRecall.append(metrics["xDataNumRecall"])
            yDataNumRecall.append(metrics["yDataNumRecall"])
            upErrDataNumRecall.append([0]*len(metrics["xDataNumRecall"]))
            lowErrDataNumRecall.append([0]*len(metrics["xDataNumRecall"]))
            # Recall
            xDataRecall.append(metrics["xDataRecall"])
            yDataRecall.append(metrics["yDataRecall"])
            upErrDataRecall.append(metrics["upErrDataRecall"])
            lowErrDataRecall.append(metrics["lowErrDataRecall"])
            # Path
            xDataPath.append(metrics["xDataPath"])
            yDataPath.append(metrics["yDataPath"])
            upErrDataPath.append(metrics["upErrDataPath"])
            lowErrDataPath.append(metrics["lowErrDataPath"])
        # Both
        dataLabels.append(label)
    # Num recall plot
    plotName = "_all_experiments_num_recall"
    create_plot_several_parameter(xDataNumRecall, yDataNumRecall, upErrDataNumRecall, lowErrDataNumRecall, dataLabels,
                                  {"x": "Poisson generator frequency rate (Hz)", "y": "Number of recall operations"},
                                  color, plotName, plot, True, False)
    # Recall plot
    plotName = "_all_experiments_recall_hit_rate"
    create_plot_several_parameter(xDataRecall, yDataRecall, upErrDataRecall, lowErrDataRecall, dataLabels,
                                  {"x": "Poisson generator frequency rate (Hz)", "y": "Recall hit rate"}, color,
                                  plotName, plot, True)
    # Path plot
    plotName = "_all_experiments_path_hit_rate"
    create_plot_several_parameter(xDataPath, yDataPath, upErrDataPath, lowErrDataPath, dataLabels,
                                  {"x": "Poisson generator frequency rate (Hz)", "y": "Path hit rate"}, color,
                                  plotName, plot, True)


def main(debugLevel):
    # Experiments parameters:
    #   + Phase affected by the noise
    phaseName = ["A_learn_only", "B_recall_only", "C_learn_and_recall"]
    prettyPhaseName = ["Learn", "Recall", "Both"]
    #   + Noise effect
    noiseEffectName = ["noise_not", "noise_cue", "noise_cont", "noise_both"]
    prettyNoiseEffectName = ["Not", "Cue", "Cont", "Both"]

    # Debug info
    plot = False
    if debugLevel >= 2:
        plot = True

    # Data analysis
    #   + Phase affected by the noise: 0 = learn, 1 = recall, 2 = both
    allMetricsProcessed = {}
    for phase in [0, 1, 2]:
        #   + Noise effect: 0 = no noise, 1 = cue, 2 = cont, 3 = both
        for noiseEffect in [1, 2, 3]:
            if debugLevel >= 1:
                print("* " + str(phaseName[phase]) + "-" + str(noiseEffectName[noiseEffect]))
            # Base path to files
            basePath = "../1_noise_in_complete_map/" + str(phaseName[phase]) + "/" + str(noiseEffectName[noiseEffect]) + "/"
            # Excell file name
            excellname = basePath + "results.xlsx"
            # Load excell data
            excelldata = load_excell_data(excellname)

            # Get data processed
            experimentMetaData, resultsByFrequency = get_data_processed(excelldata)

            # Get metrics: general by frequency
            metricsByFrequency = get_metrics(resultsByFrequency, experimentMetaData["pathInfo"])

            if debugLevel >= 3:
                print(experimentMetaData)
                print(resultsByFrequency)
                print(metricsByFrequency)

            # Plot data:
            #   * For each experiment a plot for mean/std recall and path hit
            metricsProcessed = generate_individual_plots(metricsByFrequency, phaseName, phase, noiseEffectName, noiseEffect, plot, experimentMetaData["pathInfo"])
            allMetricsProcessed.update({str(prettyPhaseName[phase])+"-"+str(prettyNoiseEffectName[noiseEffect]): metricsProcessed})

    # Plot data:
    # 1 plot for recall hit rate of all experiments, 1 for the same but for path hit rate and 1 plot for recall length
    generate_general_plots(allMetricsProcessed, plot)

    if debugLevel >= 1:
        print("Finished")


if __name__ == '__main__':
    # Debug level: 0 = none, 1 = soft, 2 = soft + plot, 3 = hard
    debugLevel = 1
    main(debugLevel)

