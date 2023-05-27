# bci-finalproject

This work is prepared to meet the requirements of the course 11120ISA557300: Brain Computer Interfaces: Fundamentals and Application, instructed by Prof. Chun-Hsiang Chuang

_Table of contents_

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Framework](#model-framework)
4. [Validation](#validation)
5. [Usage](#usage)
6. [Results](#results)
7. [References](#references)

## Video HERE

## Introduction

Provide an overview of your BCI system, explaining its purpose, functionality, and key features.

## Dataset

### Experimental Design/paradigm

### Collection Procedure

### hardware and software used, data size, number of channels, sampling rate, the website from which your data was collected owner, source

### Quality evaluation

#### Surveying and analyzing existing literature

#### analyzing the hidden independent components within EEG using ICA with ICLabel

Apply ICA to your EEG data then use ICLabel to automatically label the ICs and estimate the probability of each IC being either non-brain artifactual or Brain ICs. Investigate and analyze the change in the number of recognized ICs for the following EEG datasets:

1. Raw EEG data
   | EEG (? Channels & ? Datasets) | bandpass filter | ASR | Brain | Muscle | Eye | Heart | Line Noise | Channel Noise | Other |
   | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
   | raw | | | | | | | | | |
   | filtered | | | | | | | | | |
   | ASR-corrected | | | | | | | | | |
2. Filtered EEG data
   | EEG (? Channels & ? Datasets) | bandpass filter | ASR | Brain | Muscle | Eye | Heart | Line Noise | Channel Noise | Other |
   | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
   | raw | | | | | | | | | |
   | filtered | | | | | | | | | |
   | ASR-corrected | | | | | | | | | |
3. EEG data corrected using ASR.
   | EEG (? Channels & ? Datasets) | bandpass filter | ASR | Brain | Muscle | Eye | Heart | Line Noise | Channel Noise | Other |
   | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
   | raw | | | | | | | | | |
   | filtered | | | | | | | | | |
   | ASR-corrected | | | | | | | | | |

## Model Framework

Outline the architecture and components of your BCI system. This includes the input/output mechanisms, signal preprocessing techniques, data segmentation methods, artifact removal strategies, feature extraction approaches, machine learning models utilized, and any other relevant components.

## Validation

Describe the methods used to validate the effectiveness and reliability of your BCI system.

## Usage

Describe the usage of their BCI model's code. Explain the required environment and dependencies needed to run the code. Describe any configurable options or parameters within the code. Provide instructions on how to execute the code.

## Results

Present a detailed comparison and analysis of your BCI system's performance against the competing methods. Include metrics such as accuracy, precision, recall, F1-score, or any other relevant evaluation metrics. Compare and contrast your BCI system with existing competing methods. Highlight the advantages and unique aspects of your system.

## References

- Kanoga, S., Nakanishi, M., & Mitsukura, Y. (2016). Assessing the effects of voluntary and involuntary eyeblinks in independent components of electroencephalogram. Neurocomputing, 193, 20-32.
- Agarwal, Mohit & Sivakumar, Raghupathy. (2019). Blink: A Fully Automated Unsupervised Algorithm for Eye-Blink Detection in EEG Signals. 1113-1121. 10.1109/ALLERTON.2019.8919795.
- Agarwal, Mohit & Sivakumar, R.. (2020). Charge for a whole day: Extending Battery Life for BCI Wearables using a Lightweight Wake-Up Command. 1-14. 10.1145/3313831.3376738.
- Gupta, Ekansh & Agarwal, Mohit & Sivakumar, R.. (2020). Blink to Get In: Biometric Authentication for Mobile Devices using EEG Signals. 1-6. 10.1109/ICC40277.2020.9148741.
- Hwang, H., Lim, J., Jung, Y., Choi, H., Lee, S.W., & Im, C. (2012). Development of an SSVEP-based BCI spelling system adopting a QWERTY-style LED keyboard. Journal of Neuroscience Methods, 208, 59-65.
- Meena, K., Kumar, M., & Jangra, M. (2020). Controlling Mouse Motions Using Eye Tracking Using Computer Vision. 2020 4th International Conference on Intelligent Computing and Control Systems (ICICCS), 1001-1005.
