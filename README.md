# BCI - Final Project

This work is prepared to meet the requirements of the course 11120ISA557300: Brain Computer Interfaces: Fundamentals and Application, instructed by Prof. Chun-Hsiang Chuang

## Authors

- Didier Salazar 利葉 111065427
- Edwin Sanjaya 陳潤烈 110065710
- Gabriela Herrera 凱碧 111065421

## Table of contents

- [Introduction](#introduction)
- [Demo Video](#demo-video)
- [Dataset](#dataset)
- [Model Framework](#model-framework)
- [Validation](#validation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Introduction

Brain-computer interfaces (BCIs) are systems that use EEG (Electroencephalography) data to enable users to interact with external devices and varied technologies. The user's neural activity can be analyzed under different conditions and stimuli to find correlations and identify related neural signals that can be translated to instructions. Recent advancements in eye tracking technologies and research on integrating gaze information with blinking patterns have shown promise in enhancing users' ability to intuitively select options when controlling the device or technology. In line with these advancements, we propose a project to develop a system that analyzes neural activity related to the voluntary blinking, which can be utilized as a mechanism in interaction with computers.

This project focuses on leveraging the brain-computer interface technology to develop a system that can consistently identify voluntarily blinking with the aim of enhancing accessibility by providing an alternative method for option selection. To achieve this our main objectives are:

- Identify the features in the EEG signal which can be reliable for eye blinking classification.
- Develop the pre-processing framework to enhance the signal quality of the EEG dataset while eliminating the noise.
- Develop a machine learning model to classify different types of eye blinking.
- Evaluating and fine-tuning the accuracy of our system in classifying the different neural signals into voluntary blinking and involuntary blinking.

## Demo Video

_video here_

## Dataset

For the purpose of this project, we utilized the dataset developed by Suguru Kanoga, Masaki Nakanishi, and Yasue Mitsukura as documented in their research paper titled "Assessing the effects of voluntary and involuntary eyeblinks in independent components of electroencephalogram" [[1]](#references). To acquire the dataset, we communicated with the authors and obtained the dataset "EyeblinkDataset" directly from their Google Drive.

The data was collected from 14 channels (Fp1, Fp2, F3, F4, T3, C3, Cz, C4, T4, P3, Pz, P4, O1, and O2) according to the 10–20 system + 1 EOG signal.
<br>
<img src="https://upload.wikimedia.org/wikipedia/commons/7/70/21_electrodes_of_International_10-20_system_for_EEG.svg" alt="Electrodes 10–20 system" width="250px" height="auto">
<br>

The dataset corresponds to twenty subjects (14 males and 6 females, mean age: 22.75±1.45 years, 14 right and 6 left eye dominants). As detailed in the paper[[1]](#references), the signals were acquired using active electrodes made of sintered Ag/Ag–Cl material (manufactured by g.tec Medical Engineering GmbH, Austria) were used, with their metallic tips securely attached to the scalp. Two surface Ag/Ag–Cl electrodes (Blue Sensor P, Ambu Corp., Denmark) were placed at the superior and inferior orbital rims of the left eye to record the vertical EOG signal. The left mastoid and Fz served as the reference and ground electrodes, respectively. To ensure accurate signal capture, the EEG and EOG data were band-pass filtered from 0.5 Hz to 60 Hz using a Butterworth filter. The signals were then digitized at a sampling rate of 256 Hz using the g.USBamp system. The first 5 seconds of recorded data were discarded as they were deemed unreliable. To reduce skin resistance and ensure good electrode-skin contact, all electrodes were coated with an electrolyte called g.GAMMAgel.

The experiments to collect the data are specified in the paper and a useful diagram has also been included here. For voluntary eyeblinks, an audio stimulus was used, and participants were instructed to blink within 1 second of hearing the beep sound while focusing on a fixation point. The study comprised three sessions with 20 trials each, separated by rest periods. The sound presentation intervals were selected to capture sustained effects on EEG signals while minimizing interference. In the case of involuntary eyeblinks, three different sounds ("A," "S," and "D") at specific frequencies and volume were employed. Participants placed their left fingertips on corresponding keyboard keys, responding to the presented sound. Feedback and performance rates were provided after 20 trials, aiming for a 90% accuracy rate. Natural blinking was allowed, and three sessions were conducted.

![Diagram of a trial for voluntary and involuntary eyeblink](https://ars.els-cdn.com/content/image/1-s2.0-S0925231216001569-gr1.jpg)

In our analysis of the EEG data, we conducted time and frequency domain analyses. For the time domain, we plotted the EEG signals for each channel, showing their amplitude over time. In the frequency domain, we computed the FFT of the signals to visualize their magnitude spectra. Additionally, we calculated the event-related spectral perturbation (ERSP) using spectrogram calculations. We plotted the average ERSP over time and frequency, as well as the ERSP of a single epoch. These analyses provided insights into the characteristics and variations of the EEG signals.
<br>
<img src="./imgs/dataset/time_frequency.png" alt="Time and frequency analysis" min-width="300px" height="auto">
<br>
<img src="./imgs/dataset/ERSP.png" alt="ERSP" min-width="300px" height="auto">
<br>

#### Quality Evaluation: literature survey and analysis

The dataset provide a lot of contribution in the BCI research with 27 citations, some of the notable contribution for other research are:

- [Simultaneous Eye Blink Characterization and Elimination From Low-Channel Prefrontal EEG Signals Enhances Driver Drowsiness Detection](https://ieeexplore.ieee.org/abstract/document/9484745) as real dataset for generating an algorithm for eye blink detection and elimination
- [EEGdenoiseNet: a benchmark dataset for deep learning solutions of EEG denoising](https://iopscience.iop.org/article/10.1088/1741-2552/ac2bf8) where the dataset become a the part of EEGdenoiseNet, a dataset suitable for deep learning based EEG denoising research)
- [Machine learning classifier for eye-blink artifact detection](https://www.sciencedirect.com/science/article/pii/S2772528622000772) as real dataset for comparative analysis between machine-learning classifiers on eye-blink detection
- And other [citations](https://scholar.google.com/scholar?oi=bibs&cites=5737901810583805697) as well

In addition, the main author of the paper has a high credibility:

- Ph.D. degree in Engineering, Graduate School of Integrated Design Engineering, Keio University, Japan
- 7+ years of research experience
- Google Scholar Page: https://scholar.google.com/citations?user=69k7XzYAAAAJ

#### Analyzing the hidden independent components within EEG using ICA with ICLabel

The dataset was created by the original authors of the paper through a series of processing steps. They initially filtered the recorded EOG signals using a low-pass filter (Butterworth, cutoff frequency: 8.0 Hz) to reduce cerebral activities. Then, they detected the first positive peaks of blinks in the filtered EOG signals using a hard threshold (threshold value: 50 μV) and visually inspected these peaks to confirm their validity. The authors then segmented the eyeblink features and vertical EOG signals into 4-second epochs based on the point of maximum amplitude in the EOG data.

During our analysis, we applied Independent Component Analysis (ICA) to the EEG epoch data and used ICLabel for automatic labeling. It's important to note that the components we obtained were specifically related to brain and eye activities, reflecting the processing steps conducted by the original authors. So we wish to clarify, when we mention "raw" data, we are referring to the data that includes the processing performed by the original authors, without any additional processing from our side.

We compared the number of identified ICs in the dataset obtained from the "raw" data provided by the authors, as well as the dataset after applying bandpass filtering and Artifact Subspace Reconstruction (ASR) correction. We observed slight differences between these datasets. To gain further insights into the changes within the datasets, we also examined the probabilities shown by the ICA, which were influenced by our additional processing steps of bandpass filtering and ASR correction.

<table style="padding: 10px; border: solid 1px black">
  <tr>
    <td>&nbsp;</td>
    <td colspan="2" style="text-align: center; padding: 5px; font-weight: 600">
      Pre-processing
    </td>
    <td colspan="7" style="text-align: center; padding: 5px; font-weight: 600">
      Average numbers of ICs classified by ICLabel
    </td>
  </tr>
  <tr>
    <td>EEG (14 Channels & Eyeblink Dataset)</td>
    <td style="text-align: center; padding: 5px">bandpass filter</td>
    <td style="text-align: center; padding: 5px">ASR</td>
    <td style="text-align: center; padding: 5px">Brain</td>
    <td style="text-align: center; padding: 5px">Muscle</td>
    <td style="text-align: center; padding: 5px">Eye</td>
    <td style="text-align: center; padding: 5px">Heart</td>
    <td style="text-align: center; padding: 5px">Line Noise</td>
    <td style="text-align: center; padding: 5px">Channel Noise</td>
    <td style="text-align: center; padding: 5px">Other</td>
  </tr>
  <tr>
    <td>Raw</td>
    <td style="text-align: center; padding: 5px">&nbsp;</td>
    <td style="text-align: center; padding: 5px">&nbsp;</td>
    <td style="text-align: center; padding: 5px">4</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">10</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">0</td>
  </tr>
  <tr>
    <td>Filtered</td>
    <td style="text-align: center; padding: 5px">v</td>
    <td style="text-align: center; padding: 5px">&nbsp;</td>
    <td style="text-align: center; padding: 5px">4</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">10</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">0</td>
  </tr>
  <tr>
    <td>ASR-corrected</td>
    <td style="text-align: center; padding: 5px">v</td>
    <td style="text-align: center; padding: 5px">v</td>
    <td style="text-align: center; padding: 5px">6</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">6</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">2</td>
  </tr>
</table>

Upon examining the results, we can observe that applying ASR to the filtered data yields less accurate detection of independent components, with some components being categorized as "others," which is not ideal. This can be attributed to the ASR algorithm and its impact on the data. The ASR correction process has the potential to remove or weaken EEG components associated with eye activities or other artifacts, as well as alter the spatial distribution and temporal characteristics of the remaining ICs. Consequently, this leads to a decrease in the number of identified ICs related to these artifacts. Its objective is to eliminate artifacts and optimize the signal-to-noise ratio, resulting in changes to the ICs' amplitude, shape, and timing. Since the dataset provided by the authors had been pre-processed, these ARS related alterations can potentially negatively affect the interpretation and identification of the ICs.

<table style="padding:10px">
   <thead>
    <td style="text-align: center;">Raw</td>
    <td style="text-align: center;">Filtered</td>
    <td style="text-align: center;">ASR-corrected</td>
   </thead>
  <tr>
   <td><img src="./imgs/01_merged_raw_data_ICA.png"  alt="rew" width=370px height=auto></td> 
   <td><img src="./imgs/02_merged_bandpass_ICA.png" alt="filtered" width=370px height=auto></td>
   <td><img src="./imgs/03_merged_ASR_ICA.png" alt="asr-corrected" width=370px height=auto></td>
  </tr>
</table>

Based on our analysis and evaluation of the data, we have determined that our pre-processing steps do not improve the Independent Component Analysis (ICA) results. Therefore, it is concluded that the best course of action is to continue utilizing the raw data provided by the authors.

## Model Framework

_Outline the architecture and components of your BCI system. This includes the input/output mechanisms, signal preprocessing techniques, data segmentation methods, artifact removal strategies, feature extraction approaches, machine learning models utilized, and any other relevant components._

Taking into account that the dataset has already been labeled, supervised learning was used to
create a model that can classify voluntary and involuntary blinking. We explored various
models of supervised learning, specifically: Support Vector Machine (SVM), Random Forest, Recurrent
Neural Network (RNN), and Linear Discriminant Analysis (LDA). This approach provided valuable insights
since employing multiple models facilitated a comparison of their performance and helped identify the
model that best suited the data.
Our system framework is as follow:

1. Preprocessing of the data: noise was filtered out and the data was be organized homogeneously
   across all subjects using Matlab.
2. Feature engineering: the exploration of different features, such as peak amplitude, duration, blink frequency, inter-blink
   interval or time-frequency features, was conducted.Out of the features extracted, the most informative ones were used to reduce the
   dimensionality of the data and improve the performance of the machine learning algorithms.
3. Machine learning model development: the preprocessed data and selected features were used to
   generate the supervised learning models for Support Vector Machine (SVM), Random Forest,
   Recurrent Neural Network (RNN), and Linear Discriminant Analysis (LDA).
4. Evaluation: the dataset was divided into a training set and a validation set, then each model was trained on the training set and their performance was assessed using the validation set. After obtaining each model's performance, their results were compared and the most effective model was selected.

![Framework](/imgs/framework.png)

## Validation

_Describe the methods used to validate the effectiveness and reliability of your BCI system._

## Usage

_Describe the usage of their BCI model's code._

### Environment

_Explain the required environment and dependencies needed to run the code. Describe any configurable options or parameters within the code._

Used software application:
1. Jupyter Notebook: Works as a user interface for the classifier
2. Python: To run the python programs and libraries, mainly working on feature extraction and classification
3. MATLAB: To pre-process the EEG signal before processed by Python

### Configuration

### Execution

_Provide instructions on how to execute the code._

## Results

_Present a detailed comparison and analysis of your BCI system's performance against the competing methods. Include metrics such as accuracy, precision, recall, F1-score, or any other relevant evaluation metrics. Compare and contrast your BCI system with existing competing methods. Highlight the advantages and unique aspects of your system._

## References

1. Kanoga, S., Nakanishi, M., & Mitsukura, Y. (2016). Assessing the effects of voluntary and involuntary eyeblinks in independent components of electroencephalogram. Neurocomputing, 193, 20-32. [https://doi.org/10.1016/j.neucom.2016.01.057](https://doi.org/10.1016/j.neucom.2016.01.057)
