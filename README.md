# Medic-Analyze04

Medical Image Analysis for Pneumonia Detection with comparison for Sequential and Parallel Processing

## Project Overview

Medic-analyze04 is a medical image analysis project focused on pneumonia dataset processing using image preprocessing and computational comparison between sequential and parallel execution models. The project explores how image-based medical analysis can be improved using optimized computational techniques and parallel processing frameworks such as OpenMP.

The main objective is to process pneumonia-related medical images, apply preprocessing operations, and compare the runtime and efficiency of sequential versus parallel implementations. This project is particularly relevant for understanding how computational optimization impacts medical image analysis workloads, which are often resource-intensive.

## Motivation

Medical image analysis often involves large datasets and computationally expensive operations such as filtering, enhancement, and transformations. In clinical and research settings, efficient processing is essential for timely diagnosis and analysis. This project aims to study the performance differences between sequential execution and parallel execution when working with pneumonia image data.

By comparing both implementations, the project highlights the benefits of parallel computation in reducing execution time and improving scalability for image-processing tasks.

## Features

- Pneumonia dataset preprocessing
- Medical image analysis workflow
- Sequential processing implementation
- Parallel processing implementation using OpenMP
- Performance comparison between sequential and parallel execution
- Image enhancement and transformation pipeline
- Computational efficiency analysis for medical imaging workloads

## Tech Stack

This project combines several technologies and tools:

- Python
- C++
- C
- Makefile
- OpenMP
- Medical image preprocessing workflows
- Image processing and computational analysis utilities

## System Architecture

The project is designed around a workflow that includes:

1. Dataset acquisition and loading
2. Image preprocessing and quality enhancement
3. Feature extraction or transformation operations
4. Sequential execution pipeline
5. Parallel execution pipeline using OpenMP
6. Runtime comparison and evaluation
7. Analysis of speedup and performance metrics

The project compares the same processing pipeline in both sequential and parallel modes to evaluate how OpenMP improves execution speed for image-based workloads.

## Dataset

The project uses a pneumonia image dataset for medical image analysis. The dataset is used to study patterns in chest X-ray or pneumonia-related images and to evaluate the effectiveness of preprocessing and computational processing strategies.

## Workflow

The general workflow of the project is as follows:

- Load the pneumonia dataset
- Perform image preprocessing steps such as resizing, normalization, filtering, or enhancement
- Process images through the computational pipeline
- Execute the analysis sequentially
- Re-run the same pipeline in parallel using OpenMP
- Compare time efficiency and performance
- Evaluate practical benefits of parallel processing

## Installation

To run this project locally:

```bash
git clone https://github.com/Anjalieee/Medic-analyze04.git
cd Medic-analyze04
make deps 
make
make test
