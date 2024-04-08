# Analog sequential hippocampus memory model for trajectory learning and recalling: a robustness analysis overview

<h2 name="Description">Description</h2>
<p align="justify">
Code on which the paper entitled "Analog sequential hippocampus memory model for trajectory learning and recalling: a robustness analysis overview" is based, sent to a journal and awaiting review.
</p>
<p align="justify">
A fully functional analog spike-based implementation of a sequential memory model bio-inspired by the hippocampus implemented on the <a href="https://ieeexplore.ieee.org/document/8094868">DYNAP-SE</a> hardware platform using the technology of the Spiking Neuronal Network (SNN) is presented. The code is written in Python and makes use of the Samna library and their adaptation for DYNAP-SE called <a href="https://code.ini.uzh.ch/ncs/libs/dynap-se1">dynap-se1</a>. This model has been applied to robotic navigation for learning and recalling trajectories. In addition, the tolerance and robustness of the system to sources of random input noise has been analysed. The necessary scripts to replicate the tests and plots carried out in the paper are included, together with data and plots of the tests.
</p>
<p align="justify">
Please go to section <a href="#CiteThisWork">cite this work</a> to learn how to properly reference the works cited here.
</p>


<h2>Table of contents</h2>
<p align="justify">
<ul>
<li><a href="#Description">Description</a></li>
<li><a href="#Article">Article</a></li>
<li><a href="#Instalation">Instalation</a></li>
<li><a href="#Usage">Usage</a></li>
<li><a href="#RepositoryContent">Repository content</a></li>
<li><a href="#CiteThisWork">Cite this work</a></li>
<li><a href="#Credits">Credits</a></li>
<li><a href="#License">License</a></li>
</ul>
</p>


<h2 name="Article">Article</h2>
<p align="justify">
<strong>Title</strong>: Analog sequential hippocampus memory model for trajectory learning and recalling: a robustness analysis overview

<p align="justify"><strong>Abstract</strong>: The rapid expansion of information systems in all areas of society demands more powerful, efficient and low energy consumption computing systems. Neuromorphic engineering has emerged as a solution that attempts to mimic the brain to incorporate its capabilities to solve complex problems in a computationally and energy-efficient way in real time. Within neuromorphic computing, building systems to efficiently store the information is still a challenge. Among all the brain regions, the hippocampus stands out as a short-term memory capable of learning and recalling large amounts of information quickly and efficiently. In this work, we propose a spike-based bio-inspired hippocampus sequential memory model that makes use of the benefits of analog computing and Spiking Neural Networks: noise robustness, improved real-time operation and energy efficiency. This model is applied to robotic navigation in order to learn and recall trajectories that lead to a goal position within a known grid environment. The model was implemented on the special-purpose Spiking Neural Networks mixed-signal DYNAP-SE hardware platform. Through extensive experimentation together with an extensive analysis of the model's behaviour in the presence of external noise sources, its correct functioning was demonstrated, proving the robustness and consistency of the proposed neuromorphic sequential memory system.</p>

<strong>Keywords</strong>: Hippocampus model, analog sequential memory, robustness analysis, Spiking Neural Networks, Neuromorphic engineering, DYNAP-SE

<strong>Author</strong>: Daniel Casanueva-Morato

<strong>Contact</strong>: dcasanueva@us.es
</p>

<h2 name="Instalation">Instalation</h2>
<p align="justify">
<ol>
	<li>Have or have access to the DYNAP-SE hardware platform
	<li>Python version 3.8.10</li>
	<li>Python libraries:</li>
	<ul>
		<li><strong>samna</strong> 0.18.0.0</li>
		<li><strong>dynap-se1</strong> available in the <a href="https://code.ini.uzh.ch/ncs/libs/dynap-se1">gitlab repository</a></li>
		<li><strong>ctxctl_contrib</strong> available in the <a href="https://gitlab.com/neuroinf/ctxctl_contrib">gitlab repository</a></li>
		<li><strong>numpy</strong> 1.21.4</li>
		<li><strong>matplotlib</strong> 3.5.0</li>
		<li><strong>pandas</strong> 2.0.3</li>
	</ul>
</ol>
</p>

<h2 name="RepositoryContent">Repository content</h3>
<p align="justify">
<ul>
	<li><p align="justify"><a href="sequential_memory.ipynb">sequential_memory.ipynb</a>: python notebook containing the definition of the complete sequential memory model and tests to verify its basic functioning. The configuration of the STDP mechanism of this model is contained in the <a href="triplet_stdp_params_sequential.json">triplet_stdp_params_sequential.json</a> file.</p></li>
	<li><p align="justify"><a href="sequential_memory_noise_1_A_only_learn.ipynb">sequential_memory_noise_1_A_only_learn.ipynb</a>, <a href="sequential_memory_noise_1_B_only_recall.ipynb">sequential_memory_noise_1_B_only_recall.ipynb</a> and <a href="sequential_memory_noise_1_C_both_phases.ipynb">sequential_memory_noise_1_C_both_phases.ipynb</a>: python notebooks containing the definition of the complete sequential memory model together with the random noise generators based on a Poisson distribution.  For each notebook, a set of tests of the network under noise is carried out for different phases: learning only, recall only and both phases respectively. The configuration of the STDP mechanism of this model is contained in the <a href="triplet_stdp_params_sequential_noise.json">triplet_stdp_params_sequential_noise.json</a> file.</p></li>
	<li><p align="justify"><a href="results/">results</a> folder: contains the figures (.png) generated by all the tests of the different models, as well as files with the trace of modifications in the synaptic weight of CA3 during the operations performed (trace_.txt) and spikes generated by the network (events_.txt) during these tests. In the event file, the following can be found for each spike generated in the network: the time instant at which it occurred (timestamp_ms), the id of the neuron that generated it at the global level of the network (neuron_ids) as formatted at the local level of the population to which it belongs (neuron_ids_formated) and the tag associated with said neuron (event_tag) formed by the population to which the neuron that produces the spike belongs plus its local id. This can be seen for the model without noise in <a href="results/sequential_memory/">sequential_memory</a> folder and for the model with noise (and its different test cases) in <a href="results/sequential_memory_with_noise/">sequential_memory_with_noise</a>.</li>
	<li><p align="justify"><a href="noise_analisis/">noise_analisis</a> folder: contains the script used to analyse the noise applied to the network (<a href="noise_analisis/noise_analysis.py">noise_analysis.py</a>) and the script used to analyse the results of the network as a consequence of this noise (<a href="noise_analisis/results_analysis.py">results_analysis.py</a>). In addition, it includes the <a href="noise_analisis/results/">results</a> folder where the figures generated by both analyses can be found at an individual level for each test case and at a global level as a summary.</p></li>
</ul>
</p>


<h2 name="Usage">Usage</h2>
<p align="justify">
To run the different experiments, it is necessary to install all the libraries indicated in the <a href="#Instalation">instalation</a> section, to have a local or online tool for running notebooks and to have access to a DYNAP-SE board. Each cell of each notebook comments to a greater or lesser extent on what is happening in it. In general terms: connecting to the board, declaring the functions to be used during the definition of the network, defining the neural network itself, defining the learning mechanism, configuring the parameters of neurons and synapses per core of the board, elaborating and applying the tests to the model, taking and formatting the results network data and creating the figures with the data taken as a result of the test.
</p>

<p align="justify">
For this code to work, it is necessary to modify the local path to the "ctxctl_contrib" library in the first cell and the path to the STDP triplet mechanism parameter file in the configuration cell of this mechanism. To configure the test case to be performed, the following parameters can be varied in the network model definition cell: "exp_id" to indicate whether to perform learning or recall, "poisson_freq" to indicate the frequency of each Poisson generator, "noise_target" to indicate the part of the memory that will be affected by the noise and "rep_id" to indicate the number of repetitions of the experiment.
</p>


<h2 name="CiteThisWork">Cite this work</h2>
<p align="justify">
Work in progress...
</p>


<h2 name="Credits">Credits</h2>
<p align="justify">
The author of the original idea is Daniel Casanueva-Morato while working on a research project of the <a href="http://www.rtc.us.es/">RTC Group</a>.

This work is part of the project SANEVEC TED2021-130825B-I00, funded by the Ministerio de Ciencia e Innovaci\'on (MCIN), Agencia Estatal de  Investigaci\'on (AEI) of Spain, MCIN/AEI/10.13039/501100011033, and by the European Union NextGenerationEU/PRTR, and was partially supported by project PID2019-105556GB-C33 funded by MCIN/ AEI /10.13039/501100011033. D. C.-M. was supported by a "Formaci\'{o}n de Profesorado Universitario" Scholarship and by "Ayudas complementarias de movilidad" from the Spanish Ministry of Education, Culture and Sport.

D.~Casanueva-Morato would like to thank Giacomo Indiveri and his group for hosting him during a three-month internship between 1st June 2023 and 31th August 2023, during which the idea of this paper was originated and most of the results presented in this work were obtained.
</p>


<h2 name="License">License</h2>
<p align="justify">
This project is licensed under the GPL License - see the <a href="https://github.com/dancasmor/Robust-analog-sequential-hippocampus-memory-model-for-trajectory-learning-and-recalling/blob/main/LICENSE">LICENSE.md</a> file for details.
</p>
<p align="justify">
Copyright © 2023 Daniel Casanueva-Morato<br>  
<a href="mailto:dcasanueva@us.es">dcasanueva@us.es</a>
</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
