import matplotlib.pyplot as plt
from pathlib import Path
import copy


def load_file_data(filename):
    file = open(filename, "r")
    data = eval(file.read())
    file.close()
    return data


def save_file_data(filename, data):
    file = open(filename, "w")
    file.write(str(data))
    file.close()


def get_data_formatted(phaseFolderName, phaseFileName, noiseEffectFolderName, poissonRates, experimentPhaseName):
    """
    Get the data from all the experiments file formatted

    :param phaseFolderName: phase affected by the noise
    :param phaseFileName: noise effect
    :param noiseEffectFolderName: noise effect folder names
    :param poissonRates: noise poisson frequency
    :param experimentPhaseName: phase when the data is recorded
    :return: results: all the data from the experiments files formatted
        results = {experimenti: resultsExperimenti}
            experimenti = experiment id (int)
            resultsExperimenti = {"metadata": metadata, "dataByFrequency": dataByFrequency}
                metadata = {"phase": int, "noiseEffect": int, "phaseName": str, "noiseEffectName": str, "freq": list}
                dataByFrequency = {freqi: dataFreqi}
                    freqi = frequency of poisson noise
                    dataFreqi = {"learn": dataL, "recall": dataR}
                        datai = {popi: numSpikes}
                            popi = population name tha generated the spikes
                            numSpikes = number of spikes generated by the popi
    """

    # Data analysis
    results = {}
    popNames = ["NOISE", "IN", "DG", "CA3cue", "CA3cont", "CA1", "EC"]
    experimentID = 0
    #   + Phase affected by the noise: 0 = learn, 1 = recall, 2 = both
    for phase in [0, 1, 2]:
        #   + Noise effect: 0 = no noise, 1 = cue, 2 = cont, 3 = both
        for noiseEffect in [0, 1, 2, 3]:
            dataByFrequency = {}
            #   + Frequency rate of poisson generators
            for freqID in range(len(poissonRates[noiseEffect][phase])):
                dataFreqi = {}
                #   + Experimentation phase of the data
                for experimentPhase in [0, 1]:
                    datai = {}
                    #   + Repetition of the experiment: from 1 to 5
                    for repID in [1, 2, 3, 4, 5]:
                        # Get name of the file with the data
                        filename = "../1_noise_in_complete_map/" + str(phaseFolderName[phase]) + "/" + \
                                   str(noiseEffectFolderName[noiseEffect]) + "/events_sequential_noise_1_" + \
                                   phaseFileName[phase] + "_" + noiseEffectFolderName[noiseEffect].replace("_", "") + \
                                   "_freq" + str(poissonRates[noiseEffect][phase][freqID]) + "_rep" + str(repID) + \
                                   "_" + experimentPhaseName[experimentPhase] + ".txt"
                        # Get the data from the file
                        rawData = load_file_data(filename)["timestamp_ms"]
                        # From the raw data, get the amount of spikes for each population
                        for popIndex, spikes in enumerate(rawData):
                            popName = popNames[popIndex]
                            if popName in datai.keys():
                                datai[popName] = datai[popName] + len(spikes)
                            else:
                                datai[popName] = len(spikes)
                    # Divide for the number of repetition
                    for freq in datai.keys():
                        datai[freq] = datai[freq] / 5
                    dataFreqi.update({experimentPhaseName[experimentPhase]: datai})
                dataByFrequency.update({poissonRates[noiseEffect][phase][freqID]: dataFreqi})
            # Inset the data in the ouput variable
            metadata = {"phase": phase, "noiseEffect": noiseEffect, "phaseName": phaseFolderName[phase],
                        "noiseEffectName": noiseEffectFolderName[noiseEffect], "freq": poissonRates[noiseEffect][phase]}
            results.update({experimentID: {"metadata": metadata, "dataByFrequency": dataByFrequency}})
            experimentID = experimentID + 1
    return results


def get_metrics(results):
    """
    Get metrics for all experiments
    :param results:
    :return: metrics: metrics parameters for all experiments
        metrics = = {experimenti: metricsExperimenti}
            experimenti = experiment id (int)
            metricsExperimenti = {"metadata": metadata, "generalMetrics": generalMetrics, "metricsByFrequency": metricsByFrequency}
                metadata = {"phase": int, "noiseEffect": int, "phaseName": str, "noiseEffectName": str, "freq": list}
                generalMetrics = {"learn": generalMetricsL, "recall": generalMetricsR}
                    generalMetricsi = {"baseInputActivity": float, "baseNetworkActivity": float, "noiseInNetworkNoiseInInputRateMean": float}
                metricsByFrequency = {freqi: metricsFreqi}
                    freqi = frequency of poisson noise
                    metricsFreqi = {"learn": metricsL, "recall": metricsiR}
                            metricsi = {"noise": float, "inputActivity": float, "networkActivity": float,
                                    "networkNoise": float, "inputNoise": float,
                                    "inputActivityNoisyBaseRate": float, "networkActivityNoisyBaseRate" ~SNR: float,
                                    "noiseInInputInputInformationRate": float, "noiseInNetworkNetworkInformationRate": float,
                                    "noiseInNetworkNoiseInInputRate": float}
    """

    # Get general metrics
    generalMetrics = {}
    phaseIndex = {"learn": 0, "recall": 1}
    numCases = 0
    baseInputActivity = [0, 0]
    baseNetworkActivity = [0, 0]
    for experimentID, resultsExperimenti in results.items():
        if resultsExperimenti["metadata"]["noiseEffect"] == 0:
            numCases = numCases + 1
            for freqi, dataFreqi in resultsExperimenti["dataByFrequency"].items():
                for phase, datai in dataFreqi.items():
                    for pop, activity in datai.items():
                        baseNetworkActivity[phaseIndex[phase]] = baseNetworkActivity[phaseIndex[phase]] + activity
                        if pop == "IN":
                            baseInputActivity[phaseIndex[phase]] = baseInputActivity[phaseIndex[phase]] + activity
    baseInputActivity = [x / numCases for x in baseInputActivity]
    baseNetworkActivity = [x / numCases for x in baseNetworkActivity]
    generalMetrics.update({"learn": {"baseInputActivity": baseInputActivity[phaseIndex["learn"]], "baseNetworkActivity": baseNetworkActivity[phaseIndex["learn"]]}})
    generalMetrics.update({"recall": {"baseInputActivity": baseInputActivity[phaseIndex["recall"]], "baseNetworkActivity": baseNetworkActivity[phaseIndex["recall"]]}})

    # Get metrics for each experiment (that has noise)
    metrics = {}
    for experimentID, resultsExperimenti in results.items():
        if not(resultsExperimenti["metadata"]["noiseEffect"] == 0):
            metricsExperimenti = {"metadata": resultsExperimenti["metadata"], "generalMetrics": copy.deepcopy(generalMetrics), "metricsByFrequency": {}}
            # Get metrics by frequency
            metricsByFrequency = {}
            for freqi, dataFreqi in resultsExperimenti["dataByFrequency"].items():
                metricsFreqi = {}
                for phase, datai in dataFreqi.items():
                    # Base metrics
                    metricsi = {"noise": datai["NOISE"], "inputActivity": datai["IN"]}
                    networkActivity = 0
                    for activity in datai.values():
                        networkActivity = networkActivity + activity
                    metricsi.update({"networkActivity": networkActivity})
                    # Noise only metrics
                    metricsi.update({"networkNoise": networkActivity - generalMetrics[phase]["baseNetworkActivity"],
                                     "inputNoise": datai["IN"] - generalMetrics[phase]["baseInputActivity"]})
                    # Rate metrics
                    metricsi.update({"inputActivityNoisyBaseRate": datai["IN"] / generalMetrics[phase]["baseInputActivity"],
                                     "networkActivityNoisyBaseRate": networkActivity / generalMetrics[phase]["baseNetworkActivity"],
                                     "noiseInInputInputInformationRate": metricsi["inputNoise"] / generalMetrics[phase]["baseInputActivity"],
                                     "noiseInNetworkNetworkInformationRate": metricsi["networkNoise"] / generalMetrics[phase]["baseNetworkActivity"],
                                     "noiseInNetworkNoiseInInputRate": metricsi["networkNoise"] / metricsi["inputNoise"]})

                    metricsFreqi.update({phase: metricsi})
                metricsByFrequency.update({freqi: metricsFreqi})
            metricsExperimenti["metricsByFrequency"] = metricsByFrequency

            # Add all metrics to the output variable
            metrics.update({experimentID: metricsExperimenti})

    # Get general metrics that depend on metrics for each experiment: noiseInNetworkNoiseInInputRateMean
    for experimentID, metricsExperimenti in metrics.items():
        noiseInNetworkNoiseInInputRateMeanLearn, noiseInNetworkNoiseInInputRateMeanRecall = 0, 0
        numCases = 0
        metricsByFrequency = metricsExperimenti["metricsByFrequency"]
        for freqi, metricsFreqi in metricsByFrequency.items():
            numCases = numCases + 1
            noiseInNetworkNoiseInInputRateMeanLearn = noiseInNetworkNoiseInInputRateMeanLearn + \
                                                      metricsFreqi["learn"]["noiseInNetworkNoiseInInputRate"]
            noiseInNetworkNoiseInInputRateMeanRecall = metricsFreqi["recall"]["noiseInNetworkNoiseInInputRate"] + \
                                                       noiseInNetworkNoiseInInputRateMeanRecall
        metricsExperimenti["generalMetrics"]["learn"]["noiseInNetworkNoiseInInputRateMean"] = noiseInNetworkNoiseInInputRateMeanLearn / numCases
        metricsExperimenti["generalMetrics"]["recall"]["noiseInNetworkNoiseInInputRateMean"] = noiseInNetworkNoiseInInputRateMeanRecall / numCases

    return metrics


def get_plots(metrics, phaseName, noiseEffectName, plot):
    metricsSelected = ["noiseInInputInputInformationRate", "noiseInNetworkNetworkInformationRate", "noiseInNetworkNoiseInInputRate"]
    metricsSelectedPretty = ["Noise in Input / Input Information", "Noise in Network / Network Information",
                             "Noise in Network / Noise in Input"]
    # Prepare data for individual plots
    xDataLearns, yDataLearns, xDataRecalls, yDataRecalls = {}, {}, {}, {}
    index = 0
    for experimentID, metricsExperimenti in metrics.items():
        metricsByFrequency = metricsExperimenti["metricsByFrequency"]
        xDataLearnExp, yDataLearnExp, xDataRecallExp, yDataRecallExp = {}, {}, {}, {}
        for metricSelected in metricsSelected:
            xDataLearn, yDataLearn, xDataRecall, yDataRecall = [], [], [], []
            for freqi, metricsFreqi in metricsByFrequency.items():
                metricsL = metricsFreqi["learn"]
                metricsiR = metricsFreqi["recall"]
                xDataLearn.append(freqi)
                yDataLearn.append(metricsL[metricSelected])
                xDataRecall.append(freqi)
                yDataRecall.append(metricsiR[metricSelected])
            xDataLearnExp.update({metricSelected: xDataLearn})
            yDataLearnExp.update({metricSelected: yDataLearn})
            xDataRecallExp.update({metricSelected: xDataRecall})
            yDataRecallExp.update({metricSelected: yDataRecall})
        xDataLearns.update({index: xDataLearnExp})
        yDataLearns.update({index: yDataLearnExp})
        xDataRecalls.update({index: xDataRecallExp})
        yDataRecalls.update({index: yDataRecallExp})
        index = index + 1

    # Get plots for each experiment
    color = "#7FB3D5"
    labels = {"x": "Poisson generator frequency rate (Hz)", "y": "Rate"}
    index = 0
    for experimentID, metricsExperimenti in metrics.items():
        # For each selected metric
        for indexMetric, metricSelected in enumerate(metricsSelected):
            plotNameL = str(phaseName[metricsExperimenti["metadata"]["phase"]]) + "_" + \
                       str(noiseEffectName[metricsExperimenti["metadata"]["noiseEffect"]]) + "_learn_" + metricSelected
            plotNameR = str(phaseName[metricsExperimenti["metadata"]["phase"]]) + "_" + \
                        str(noiseEffectName[metricsExperimenti["metadata"]["noiseEffect"]]) + "_recall_" + metricSelected
            title = metricsSelectedPretty[indexMetric]
            create_plot_1_parameter(xDataLearns[index][metricSelected], yDataLearns[index][metricSelected],
                                    labels, title, color, plotNameL, plot, yTicks=False)
            create_plot_1_parameter(xDataRecalls[index][metricSelected], yDataRecalls[index][metricSelected],
                                    labels, title, color, plotNameR, plot, yTicks=False)
        index = index + 1

    # Processes data for global plots
    xDataLearns, xDataRecalls = {}, {}
    numCases = 0
    for experimentID, metricsExperimenti in metrics.items():
        if not(100 in metricsExperimenti["metadata"]["freq"]):
            numCases = numCases + 1
            metricsByFrequency = metricsExperimenti["metricsByFrequency"]
            for freqi, metricsFreqi in metricsByFrequency.items():
                metricsL = metricsFreqi["learn"]
                metricsiR = metricsFreqi["recall"]
                if freqi in xDataLearns.keys():
                    xDataLearns[freqi] = xDataLearns[freqi] + metricsL["noiseInNetworkNetworkInformationRate"]
                else:
                    xDataLearns[freqi] = metricsL["noiseInNetworkNetworkInformationRate"]
                if freqi in xDataRecalls.keys():
                    xDataRecalls[freqi] = xDataRecalls[freqi] + metricsiR["noiseInNetworkNetworkInformationRate"]
                else:
                    xDataRecalls[freqi] = metricsiR["noiseInNetworkNetworkInformationRate"]
    for var in [xDataLearns, xDataRecalls]:
        for key in var.keys():
            var[key] = var[key] / numCases

    # Get global plots
    title = "Noise in Network / Network Information"
    plotName = "_all_noiseInNetworkNetworkInformationRate"
    create_plot_1_parameter(list(xDataLearns.keys()), list(xDataLearns.values()), labels, title,
                            color, plotName+str("_learn"), plot, yTicks=False)
    create_plot_1_parameter(list(xDataRecalls.keys()), list(xDataRecalls.values()), labels, title,
                            color, plotName+str("_recall"), plot, yTicks=False)


def create_plot_1_parameter(xData, yData, labels, title, color, plotName, plot, yTicks=True):
    """
    Create the plot with the input data for only 1 parameter (std or mean)

    :param xData: x axis values
    :param yData: y axis values
    :param labels: x and y axis labels
    :param title: title of the plot
    :param color: color of points and connection line (no error lines)
    :param plotName: name of the output plot file
    :param plot: if plot or not the figure
    :param yTicks: if put yTicks or set it default
    :return:
    """
    plt.figure(figsize=(8, 8), dpi=400)
    plt.xlabel(labels["x"])
    plt.ylabel(labels["y"])

    plt.plot(xData, yData, color=color, marker="o")

    plt.xticks(xData)
    if yTicks:
        plt.yticks([x * 0.1 for x in range(0, 11)])

    plt.title(title)
    plt.savefig("analysis_results/noise/" + plotName + ".png", bbox_inches='tight')
    if plot:
        plt.show()
    plt.close()


def print_metrics(metrics):
    for experimentID, metricsExperimenti in metrics.items():
        print("     + " + str(metricsExperimenti["metadata"]["phaseName"]) + "-" + str(metricsExperimenti["metadata"]["noiseEffectName"]))
        print("         - (learn) Noise in Network / Noise in Input: " + str(metricsExperimenti["generalMetrics"]["learn"]["noiseInNetworkNoiseInInputRateMean"]))
        print("         - (recall) Noise in Network / Noise in Input: " + str(metricsExperimenti["generalMetrics"]["recall"]["noiseInNetworkNoiseInInputRateMean"]))


def main(debugLevel, getPlots):
    # Experiments parameters:
    #   + Phase affected by the noise
    phaseFolderName = ["A_learn_only", "B_recall_only", "C_learn_and_recall"]
    phaseFileName = ["A", "B", "C"]
    #   + Noise effect
    noiseEffectFolderName = ["noise_not", "noise_cue", "noise_cont", "noise_both"]
    prettyNoiseEffectName = ["Not", "Cue", "Cont", "Both"]
    #   + Noise poisson frequency
    poissonRates = [[[0], [0], [0]],
                    [[0.5, 1, 2, 3, 4, 10], [0.5, 1, 2, 3, 4, 10], [0.5, 1, 2, 3, 4, 10]],
                    [[1, 10, 50, 75, 100, 110, 150], [0.5, 1, 2, 3, 4, 10], [0.5, 1, 2, 3, 4, 10]],
                    [[0.5, 1, 2, 3, 4, 10], [0.5, 1, 2, 3, 4, 10], [0.5, 1, 2, 3, 4, 10]]]
    #   + Phase when the data is recorded
    experimentPhaseName = ["learn", "recall"]

    # Debug info
    plot = False
    if debugLevel >= 2:
        plot = True

    # Check if data has been processed before
    metricsfilename = "results/noise/metrics.txt"
    my_file = Path(metricsfilename)
    if my_file.is_file():
        # Open and load the metrics
        if debugLevel >= 1:
            print("* Open metrics file")
        metrics = load_file_data(metricsfilename)
    else:
        # Get the data from the file formatted
        if debugLevel >= 1:
            print("* Raw data")
        results = get_data_formatted(phaseFolderName, phaseFileName, noiseEffectFolderName, poissonRates, experimentPhaseName)
        if debugLevel >= 3:
            print(results)

        # Get metrics
        if debugLevel >= 1:
            print("* Metrics")
        metrics = get_metrics(results)
        if debugLevel >= 3:
            print(metrics)

        # Save metrics for the next time
        save_file_data(metricsfilename, metrics)

    # Plots
    if getPlots:
        if debugLevel >= 1:
            print("* Plots")
        get_plots(metrics, phaseFileName, prettyNoiseEffectName, plot)

    # Print some metrics
    if debugLevel >= 1:
        print("* Print metrics")
    print_metrics(metrics)

    if debugLevel >= 1:
        print("Finished")


if __name__ == '__main__':
    # Debug level: 0 = none, 1 = soft, 2 = soft + plot, 3 = hard
    debugLevel = 1
    # If generate and save plots (True) or not (False)
    getPlots = False
    main(debugLevel, getPlots)

